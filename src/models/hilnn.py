"""HiLNN v1: history-informed context-conditioned Lagrangian rollout."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoders import GRUHistoryEncoder
from src.models.integrators import rollout
from src.models.lagrangian_core import ContextLagrangianCore


class HiLNN(nn.Module):
    def __init__(
        self,
        q_dim: int = 1,
        history_len: int = 8,
        context_dim: int = 32,
        encoder_hidden_dim: int = 64,
        encoder_num_layers: int = 1,
        core_hidden_dim: int = 128,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.q_dim = q_dim
        self.history_len = history_len
        self.context_dim = context_dim

        self.encoder = GRUHistoryEncoder(
            q_dim=q_dim,
            hidden_dim=encoder_hidden_dim,
            context_dim=context_dim,
            num_layers=encoder_num_layers,
        )
        self.core = ContextLagrangianCore(
            q_dim=q_dim,
            context_dim=context_dim,
            hidden_dim=core_hidden_dim,
            eps=eps,
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(q_dim + context_dim, core_hidden_dim),
            nn.Tanh(),
            nn.Linear(core_hidden_dim, core_hidden_dim),
            nn.Tanh(),
            nn.Linear(core_hidden_dim, q_dim),
        )

    def infer_initial_state(
        self, hist_q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(hist_q)
        q0 = hist_q[:, -1, :]
        qdot0 = self.velocity_head(torch.cat([q0, z], dim=-1))
        state0 = torch.cat([q0, qdot0], dim=-1)
        return state0, z, qdot0

    def dynamics(self, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        q, qdot = torch.split(state, self.q_dim, dim=-1)
        qddot = self.core.acceleration(q, qdot, z)
        return torch.cat([qdot, qddot], dim=-1)

    def rollout(
        self,
        hist_q: torch.Tensor,
        dt: float,
        horizon: int,
        method: str = "euler",
        detach_between_steps: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state0, z, qdot0 = self.infer_initial_state(hist_q)

        def dyn(s: torch.Tensor) -> torch.Tensor:
            return self.dynamics(s, z)

        pred_state = rollout(
            dyn, state0, dt=dt, horizon=horizon,
            method=method, detach_between_steps=detach_between_steps,
        )
        return pred_state, z, qdot0

    def forward(
        self,
        hist_q: torch.Tensor,
        dt: float,
        horizon: int,
        method: str = "euler",
        detach_between_steps: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.rollout(hist_q, dt, horizon, method, detach_between_steps)
