from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Small feed-forward network used by the variational encoder."""

    def __init__(
        self,
        f_in: int,
        f_out: int,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        dropout: float = 0.05,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if activation == "relu":
            activation_layer: nn.Module = nn.ReLU()
        elif activation == "tanh":
            activation_layer = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: list[nn.Module] = [
            nn.Linear(f_in, hidden_dim),
            activation_layer,
            nn.Dropout(dropout),
        ]
        for _ in range(hidden_layers - 2):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    activation_layer,
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(hidden_dim, f_out))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class BERTTimeEmbedding(nn.Module):
    """Learnable positional embedding with the state-dict names used by the paper code."""

    def __init__(self, max_position_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(max_position_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        position_ids = torch.arange(x.size(1), dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(x.shape[0], -1)
        return self.embeddings(position_ids)
