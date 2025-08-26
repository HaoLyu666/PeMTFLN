import torch
from torch import nn
from module.mamba import Mamba, MambaConfig


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.device = args['device']
        self.encoder_size = args['encoder_size']
        self.n_head = args['n_head']
        self.in_length = args['in_length']
        self.veh_num = args['veh_num']
        self.para_length = args['para_length']
        self.f_length = args['f_length']
        self.transformer_layer = args['transformer_layer']
        self.dropout = args['dropout']

        self.input_projection = nn.Linear(self.f_length, self.encoder_size)
        self.relu = nn.ReLU()

        mamba_config = MambaConfig(d_model=self.encoder_size, n_layers=1, d_state=8)
        self.mamba = Mamba(mamba_config)

        self.variational_encoder = nn.Sequential(
            nn.Linear(self.encoder_size, self.encoder_size * 4),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.encoder_size * 4, self.encoder_size * 2)
        )

        self.positional_embedding = nn.Embedding(self.veh_num, self.encoder_size)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_size, nhead=self.n_head,
            dim_feedforward=self.encoder_size * 4, batch_first=True, dropout=self.dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=self.transformer_layer)

        self.temporal_embedding = nn.Embedding(self.para_length, self.encoder_size)

        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.encoder_size, nhead=self.n_head,
            dim_feedforward=self.encoder_size * 4, batch_first=True, dropout=self.dropout
        )
        self.nat_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=self.transformer_layer)

        self.output_layer = nn.Linear(self.encoder_size, 3)
        self.softplus = nn.Softplus()

    def reparameterize(self, mu, log_var):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, history_features):
        # Input shape: (batch_size, num_vehicles, in_length, feature_dim)
        batch_size, num_vehicles, in_length, _ = history_features.shape

        # Reshape for Mamba: (batch_size * num_vehicles, in_length, feature_dim)
        history_features = history_features.reshape(-1, in_length, self.f_length)

        # 1. Input Projection
        hist_encoded = self.relu(self.input_projection(history_features))

        # 2. Mamba for sequential feature extraction
        hist_mamba_out = self.mamba(hist_encoded)

        # 3. Extract last time step feature for each vehicle for the VAE
        # Shape: (B*N, T, D) -> (B, N, D)
        last_step_features = hist_mamba_out.reshape(batch_size, num_vehicles, in_length, self.encoder_size)[:, :, -1, :]

        # 4. Variational encoding to get mean and log_variance
        var_out = self.variational_encoder(last_step_features)
        mu, log_var = var_out.chunk(2, dim=-1)

        # 5. Reparameterization trick to sample features
        sampled_features = self.reparameterize(mu, log_var)

        # 6. Add positional embedding for vehicle differentiation
        pos_ids = torch.arange(num_vehicles, device=self.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.positional_embedding(pos_ids)
        token_features = sampled_features + pos_emb

        # 7. Transformer Encoder for inter-vehicle interaction modeling
        causal_mask = nn.Transformer.generate_square_subsequent_mask(num_vehicles, device=self.device)
        vehicle_interactions = self.transformer_encoder(token_features, mask=causal_mask, is_causal=True)

        # 8. Prepare query for the decoder
        # Shape: (B, N, D) -> (B*N, 1, D) -> (B*N, para_len, D)
        decoder_query = vehicle_interactions.reshape(-1, self.encoder_size).unsqueeze(1).repeat(1, self.para_length, 1)

        # 9. Add temporal embedding for the decoder steps
        time_ids = torch.arange(self.para_length, device=self.device).unsqueeze(0).expand(batch_size * num_vehicles, -1)
        time_emb = self.temporal_embedding(time_ids)
        query_with_time = decoder_query + time_emb

        # 10. NAT Decoder to generate dynamic parameters over time
        tgt_causal_mask = nn.Transformer.generate_square_subsequent_mask(self.para_length, device=self.device)
        decoded_params = self.nat_decoder(query_with_time, hist_mamba_out, tgt_mask=tgt_causal_mask, tgt_is_causal=True)

        # 11. Project to final 3 physical parameters and ensure positivity
        # Shape: (B*N, para_len, D) -> (B, N, para_len, 3)
        output_params = self.softplus(self.output_layer(decoded_params))
        output_params = output_params.reshape(batch_size, num_vehicles, self.para_length, -1)

        return output_params, mu, log_var


class Predictor:
    """
    Uses the physical parameters from the Encoder to perform a forward simulation.
    This class does not have trainable weights and is not part of the gradient computation graph.
    """

    def __init__(self, args):
        self.out_length = args['out_length']
        self.in_length = args['in_length']
        self.dt = args['time_step']
        self.dynamic_params = None

    def _update_state(self, current_state, equilibrium_history, speed_difference):
        """Logic for a single simulation time step."""
        # Calculate the dynamic equilibrium state (mean of the history)
        eq_state = torch.mean(equilibrium_history, dim=2)

        # State error: [gap_error, velocity_error, speed_difference]
        state_error = torch.cat([current_state - eq_state, speed_difference], dim=-1)

        # Calculate acceleration: a_t = p1*e_gap + p2*e_vel + p3*e_d_vel
        acceleration = torch.sum(torch.mul(self.dynamic_params, state_error), dim=-1, keepdim=True)
        acceleration = torch.clamp(acceleration, min=-5, max=5)

        # Update velocity and gap based on physics
        v_next = current_state[:, :, 1:2] + acceleration * self.dt
        h_next = current_state[:, :, 0:1] + speed_difference * self.dt

        next_state = torch.cat([h_next, v_next], dim=-1)
        return next_state

    def forward(self, decoded_params, leader_future_vel, initial_state, initial_history):
        # Adjust signs of physical parameters for the physics model
        # The velocity gain parameter should be negative for stability
        params = decoded_params.clone()
        params[:, :, :, 1] = -params[:, :, :, 1]

        prediction_steps = []
        current_state = initial_state
        equilibrium_history = initial_history
        param_idx = -1

        # Simulate step-by-step for the entire prediction horizon
        for i in range(self.out_length):
            # Update the dynamic physical parameters every 5 steps
            if i % 5 == 0:
                param_idx += 1
            self.dynamic_params = params[:, :, param_idx, :]

            # Calculate speed difference with the preceding vehicle
            preceding_vehicle_vel = torch.cat([leader_future_vel[:, :, i, :], current_state[:, :-1, 1:2]], dim=1)
            speed_difference = preceding_vehicle_vel - current_state[:, :, 1:2]

            # Update vehicle state for the next time step
            current_state = self._update_state(current_state, equilibrium_history, speed_difference)

            # Update the history buffer for the next equilibrium calculation
            equilibrium_history = torch.cat([equilibrium_history, current_state.unsqueeze(2)], dim=2)

            prediction_steps.append(current_state.unsqueeze(2))

        # Concatenate all predicted steps into a single trajectory tensor
        predicted_trajectory = torch.cat(prediction_steps, dim=2)

        # Calculate PET as an additional output feature
        gap = predicted_trajectory[:, :, :, 0:1]
        velocity = predicted_trajectory[:, :, :, 1:2]
        pet = gap / (velocity + 1e-8)  # Add epsilon to avoid division by zero
        pet = torch.clamp(pet, min=0.1, max=5)

        # Final output contains [gap, speed, PET]
        final_prediction = torch.cat([predicted_trajectory, pet], dim=-1)

        return final_prediction
