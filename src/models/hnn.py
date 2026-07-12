"""Hamiltonian Neural Network baseline (p ≈ qdot for pendulum)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.integrators import rollout


class HNN(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.state_dim = state_dim
        layers: list[nn.Module] = []
        d = state_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.Tanh()]
            d = hidden_dim
        layers.append(nn.Linear(d, 1))
        self.h_net = nn.Sequential(*layers)

    def hamiltonian(self, x: torch.Tensor) -> torch.Tensor:
        return self.h_net(x).squeeze(-1)

    def time_derivative(self, x: torch.Tensor) -> torch.Tensor:
        inference = not torch.is_grad_enabled()

        with torch.enable_grad():
            if inference:
                x = x.detach().requires_grad_(True)
            elif not x.requires_grad:
                x = x.detach().requires_grad_(True)

            h = self.hamiltonian(x).sum()
            grad = torch.autograd.grad(h, x, create_graph=True)[0]
            dH_dq = grad[..., :1]
            dH_dp = grad[..., 1:2]
            dxdt = torch.cat([dH_dp, -dH_dq], dim=-1)

        return dxdt.detach() if inference else dxdt

    def dynamics(self, x: torch.Tensor) -> torch.Tensor:
        return self.time_derivative(x)

    def predict_sequence(
        self,
        x0: torch.Tensor,
        horizon: int,
        dt: float,
        detach_between_steps: bool = False,
        method: str = "rk4",
    ) -> torch.Tensor:
        return rollout(
            self.dynamics,
            x0,
            dt=dt,
            horizon=horizon,
            method=method,
            detach_between_steps=detach_between_steps,
        )
