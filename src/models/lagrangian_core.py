"""Context-conditioned Lagrangian core for HiLNN."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextLagrangianCore(nn.Module):
    """
    L(q, qdot; z) = T - V,  T = 0.5 * M(q,z) * qdot^2
    """

    def __init__(
        self,
        q_dim: int,
        context_dim: int,
        hidden_dim: int = 128,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.q_dim = q_dim
        self.context_dim = context_dim
        self.eps = eps
        inp = q_dim + context_dim

        def _net(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(inp, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, out_dim),
            )

        self.mass_net = _net(q_dim)
        self.potential_net = _net(1)

    def _qz(self, q: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if z.shape[0] == q.shape[0]:
            return torch.cat([q, z], dim=-1)
        return torch.cat([q, z.expand(q.shape[0], -1)], dim=-1)

    def mass(self, q: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.mass_net(self._qz(q, z))) + self.eps

    def potential(self, q: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.potential_net(self._qz(q, z)).squeeze(-1)

    def lagrangian(self, q: torch.Tensor, qdot: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        m = self.mass(q, z)
        kinetic = 0.5 * torch.sum(m * (qdot ** 2), dim=-1)
        return kinetic - self.potential(q, z)

    def acceleration(self, q: torch.Tensor, qdot: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """1D analytic EL with fixed context z: qddot = (-0.5 dm/dq qdot^2 - dV/dq) / m."""
        inference = not torch.is_grad_enabled()

        with torch.enable_grad():
            if inference:
                q = q.detach().requires_grad_(True)
            elif not q.requires_grad:
                q = q.detach().requires_grad_(True)

            create_graph = not inference
            m = self.mass(q, z)
            dm_dq = torch.autograd.grad(m.sum(), q, create_graph=create_graph, retain_graph=True)[0]
            v = self.potential(q, z)
            dv_dq = torch.autograd.grad(v.sum(), q, create_graph=create_graph)[0]
            qddot = (-0.5 * dm_dq * (qdot ** 2) - dv_dq) / m

        return qddot.detach() if inference else qddot
