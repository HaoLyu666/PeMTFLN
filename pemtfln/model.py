from __future__ import annotations

import torch
from torch import nn

from .layers import BERTTimeEmbedding, MLP
from .mamba import Mamba, MambaConfig


class Encoder(nn.Module):
    """Multi-scale trajectory feature encoder.

    The module attribute names intentionally follow the original research code
    so released checkpoints such as ``epoch20_e.tar`` can be loaded directly.
    """

    def __init__(self, args: dict) -> None:
        super().__init__()
        self.device = args["device"]
        self.encoder_size = args["encoder_size"]
        self.n_head = args["n_head"]
        self.in_length = args["in_length"]
        self.out_length = args["out_length"]
        self.f_length = args["f_length"]
        self.train_flag = args["train_flag"]
        self.batch_size = args["batch_size"]
        self.dropout = args["dropout"]
        self.transformer_layer = args["transformer_layer"]
        self.num_mc = args["num_mc"]
        self.veh_num = args["veh_num"]
        self.para_length = args["para_length"]

        self.encoder = nn.Linear(self.f_length, self.encoder_size)
        self.relu = nn.ReLU()
        self.mamba_Config = MambaConfig(d_model=self.encoder_size, n_layers=1, d_state=8)
        self.mamba = Mamba(self.mamba_Config)
        self.var_encoder = MLP(
            f_in=self.encoder_size,
            f_out=self.encoder_size * 2,
            activation="tanh",
            hidden_dim=self.encoder_size * 4,
            hidden_layers=3,
            dropout=self.dropout,
        )

        self.pe_emb = BERTTimeEmbedding(
            max_position_embeddings=self.veh_num * 6,
            embedding_dim=self.encoder_size,
        )
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.encoder_size,
            nhead=self.n_head,
            dim_feedforward=self.encoder_size * 4,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.TransformerEncoderLayer,
            num_layers=self.transformer_layer,
        )
        self.te_emb = BERTTimeEmbedding(
            max_position_embeddings=self.para_length,
            embedding_dim=self.encoder_size,
        )
        self.TransformerDecoderLayer = nn.TransformerDecoderLayer(
            d_model=self.encoder_size,
            nhead=self.n_head,
            dim_feedforward=self.encoder_size * 4,
            batch_first=True,
        )
        self.nat = nn.TransformerDecoder(
            self.TransformerDecoderLayer,
            num_layers=self.transformer_layer,
        )
        self.decoder = nn.Linear(self.encoder_size, 3)
        self.softplus = nn.Softplus()

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history = history.reshape(-1, self.in_length, self.f_length)
        history_encoded = self.relu(self.encoder(history))
        history_mamba = self.mamba(history_encoded)

        vehicle_features = history_mamba.reshape(
            -1,
            self.veh_num,
            self.in_length,
            self.encoder_size,
        )[:, :, -1, :]
        variational_features = self.var_encoder(vehicle_features)
        mu = variational_features[:, :, : self.encoder_size]
        log_var = variational_features[:, :, self.encoder_size :]

        if self.num_mc:
            history_mamba = history_mamba.repeat([self.num_mc, 1, 1])
            sampled = torch.cat(
                [self.reparameterize(mu, log_var) for _ in range(self.num_mc)],
                dim=0,
            )
        else:
            sampled = self.reparameterize(mu, log_var)

        vehicle_tokens = self.pe_emb(sampled) + sampled
        vehicle_mask = nn.Transformer.generate_square_subsequent_mask(vehicle_tokens.size(1)).to(vehicle_tokens.device)
        vehicle_context = self.transformer_encoder(vehicle_tokens, vehicle_mask, is_causal=True)
        decoder_seed = vehicle_context.reshape(-1, self.encoder_size).unsqueeze(1).repeat(1, self.para_length, 1)
        decoder_query = self.te_emb(decoder_seed) + decoder_seed
        decoder_mask = nn.Transformer.generate_square_subsequent_mask(decoder_query.size(1)).to(decoder_query.device)
        decoded = self.nat(
            decoder_query,
            history_mamba,
            tgt_mask=decoder_mask,
            tgt_is_causal=True,
        )
        decoded = decoded.reshape(-1, self.veh_num, self.para_length, self.encoder_size)
        params = self.softplus(self.decoder(decoded))
        return params, mu, log_var


class Predictor:
    """Analyzable parameters encoded computational graph (APeCG)."""

    def __init__(self, args: dict) -> None:
        self.hist_para: torch.Tensor | None = None
        self.args = args
        self.epsilon = 1e-10
        self.device = args["device"]
        self.train_flag = args["train_flag"]
        self.out_dim = args["out_dim"]
        self.dropout = args["dropout"]
        self.batch_size = args["batch_size"]
        self.num_mc = args["num_mc"]
        self.out_length = args["out_length"]
        self.in_length = args["in_length"]
        self.dt = args["time_step"]
        self.para_length = args["para_length"]

    def veh_dynamic(
        self,
        vehicle_state: torch.Tensor,
        equilibrium_history: torch.Tensor,
        speed_difference: torch.Tensor,
    ) -> torch.Tensor:
        equilibrium_state = torch.mean(equilibrium_history[:, :, -self.in_length :, :], dim=2)
        state_error = torch.cat([vehicle_state - equilibrium_state, speed_difference], dim=-1)
        acceleration = torch.sum(torch.mul(self.hist_para, state_error), dim=-1, keepdim=True)
        acceleration = torch.clamp(acceleration, min=-5, max=5)
        next_velocity = vehicle_state[:, :, 1:2] + acceleration * self.dt
        next_gap = vehicle_state[:, :, 0:1] + speed_difference * self.dt
        return torch.cat([next_gap, next_velocity], dim=-1)

    def forward(
        self,
        encoded_params: torch.Tensor,
        leader_future_velocity: torch.Tensor,
        vehicle_state: torch.Tensor,
        equilibrium_history: torch.Tensor,
    ) -> torch.Tensor:
        dynamic_params = encoded_params.clone()
        dynamic_params[:, :, :, 1] = -dynamic_params[:, :, :, 1]

        predictions = []
        param_idx = -1
        if self.num_mc:
            leader_future_velocity = leader_future_velocity.repeat([self.num_mc, 1, 1, 1])
            vehicle_state = vehicle_state.repeat([self.num_mc, 1, 1])
            equilibrium_history = equilibrium_history.repeat([self.num_mc, 1, 1, 1])

        for step in range(self.out_length):
            preceding_velocity = torch.cat(
                [leader_future_velocity[:, :, step, :], vehicle_state[:, :-1, 1:2]],
                dim=1,
            )
            speed_difference = preceding_velocity - vehicle_state[:, :, 1:2]
            if step % 5 == 0:
                param_idx += 1
            self.hist_para = dynamic_params[:, :, param_idx, :]
            vehicle_state = self.veh_dynamic(vehicle_state, equilibrium_history, speed_difference)
            equilibrium_history = torch.cat([equilibrium_history, vehicle_state.unsqueeze(2)], dim=2)
            predictions.append(vehicle_state.unsqueeze(2))

        predicted_state = torch.cat(predictions, dim=2)
        pet = predicted_state[:, :, :, 0:1] / (predicted_state[:, :, :, 1:2] + self.epsilon)
        pet = torch.clamp(pet, min=0.1, max=5)
        return torch.cat([predicted_state, pet], dim=-1)


predictor = Predictor
