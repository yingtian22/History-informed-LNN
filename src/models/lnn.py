"""Vanilla Lagrangian Neural Network (1D generalized coordinate)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.integrators import rollout


class LNN(nn.Module):
    """
    Structured Lagrangian: L = 0.5 * M(q) * qdot^2 - V(q)
    M(q) = softplus(m_net(q)) + eps
    """

    def __init__(self, q_dim: int = 1, hidden_dim: int = 64, num_layers: int = 2, eps: float = 1e-4):
        super().__init__()
        self.q_dim = q_dim
        self.eps = eps
        self.m_net = self._scalar_net(q_dim, hidden_dim, num_layers)
        self.v_net = self._scalar_net(q_dim, hidden_dim, num_layers)

    @staticmethod
    def _scalar_net(in_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.Tanh()]
            d = hidden_dim
        layers.append(nn.Linear(d, 1))
        return nn.Sequential(*layers)

    def mass(self, q: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.m_net(q)) + self.eps

    def potential(self, q: torch.Tensor) -> torch.Tensor:
        return self.v_net(q)

    def lagrangian(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        m = self.mass(q)
        kinetic = 0.5 * m * (qdot ** 2)
        return kinetic.squeeze(-1) - self.potential(q).squeeze(-1)

    def acceleration(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        """
        1D Euler--Lagrange (analytic, avoids nested autograd.grad through RK4):
        qddot = (-0.5 * dM/dq * qdot^2 - dV/dq) / M
        """
        inference = not torch.is_grad_enabled()

        with torch.enable_grad():
            if inference:
                q = q.detach().requires_grad_(True)
            elif not q.requires_grad:
                q = q.detach().requires_grad_(True)

            create_graph = not inference
            m = self.mass(q)
            dm_dq = torch.autograd.grad(
                m.sum(), q, create_graph=create_graph, retain_graph=True
            )[0]
            v = self.potential(q)
            dv_dq = torch.autograd.grad(v.sum(), q, create_graph=create_graph)[0]
            qddot = (-0.5 * dm_dq * (qdot ** 2) - dv_dq) / m

        return qddot.detach() if inference else qddot

    def dynamics(self, state: torch.Tensor) -> torch.Tensor:
        q = state[..., : self.q_dim]
        qdot = state[..., self.q_dim :]
        qddot = self.acceleration(q, qdot)
        return torch.cat([qdot, qddot], dim=-1)

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
