"""Neural ODE baseline with fixed RK4 rollout (default) or torchdiffeq."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.integrators import rollout

try:
    from torchdiffeq import odeint as _odeint

    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


class ODEFunc(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        d = state_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.Tanh()]
            d = hidden_dim
        layers.append(nn.Linear(d, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return self.net(x)


class NeuralODE(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        method: str = "rk4",
    ):
        super().__init__()
        self.func = ODEFunc(state_dim, hidden_dim, num_layers)
        self.method = method

    def dynamics(self, x: torch.Tensor) -> torch.Tensor:
        return self.func(torch.zeros((), device=x.device, dtype=x.dtype), x)

    def predict_sequence(
        self,
        x0: torch.Tensor,
        horizon: int,
        dt: float,
        use_torchdiffeq: bool = False,
        detach_between_steps: bool = False,
        integrator_method: str = "rk4",
    ) -> torch.Tensor:
        if use_torchdiffeq and HAS_TORCHDIFFEQ and self.method != "rk4_fixed":
            t_eval = torch.arange(1, horizon + 1, device=x0.device, dtype=x0.dtype) * dt
            out = _odeint(self.func, x0, t_eval, method=self.method)
            return out.transpose(0, 1)
        return rollout(
            self.dynamics,
            x0,
            dt=dt,
            horizon=horizon,
            method=integrator_method,
            detach_between_steps=detach_between_steps,
        )
