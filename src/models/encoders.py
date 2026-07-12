"""History trajectory encoders for HiLNN."""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUHistoryEncoder(nn.Module):
    """Encode position history into latent mechanical context z_t."""

    def __init__(
        self,
        q_dim: int,
        hidden_dim: int = 64,
        context_dim: int = 32,
        num_layers: int = 1,
    ):
        super().__init__()
        self.q_dim = q_dim
        self.context_dim = context_dim
        self.gru = nn.GRU(
            input_size=q_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, context_dim),
        )

    def forward(self, hist_q: torch.Tensor) -> torch.Tensor:
        """hist_q: [B, L, q_dim] -> z: [B, context_dim]"""
        _, h = self.gru(hist_q)
        return self.proj(h[-1])
