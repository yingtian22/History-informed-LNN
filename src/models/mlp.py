"""MLP baselines: one-step residual and direct multi-step."""

from __future__ import annotations

import torch
import torch.nn as nn

def _mlp_layers(in_dim: int, out_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(num_layers - 1):
        layers += [nn.Linear(d, hidden_dim), nn.Tanh()]
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class MlpOneStep(nn.Module):
    """Predict residual dx; x_{t+1} = x_t + f(x_t)."""

    def __init__(self, state_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.net = _mlp_layers(state_dim, state_dim, hidden_dim, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

    def predict_sequence(self, x0: torch.Tensor, horizon: int, dt: float = 0.05) -> torch.Tensor:
        del dt
        states = []
        x = x0
        for _ in range(horizon):
            x = self.forward(x)
            states.append(x)
        return torch.stack(states, dim=1)


class MlpDirectMultiStep(nn.Module):
    """Flatten history states and predict full future sequence."""

    def __init__(
        self,
        history_len: int,
        state_dim: int,
        horizon: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        self.history_len = history_len
        self.state_dim = state_dim
        self.horizon = horizon
        in_dim = history_len * state_dim
        out_dim = horizon * state_dim
        self.net = _mlp_layers(in_dim, out_dim, hidden_dim, num_layers)

    def forward(self, hist_state: torch.Tensor) -> torch.Tensor:
        batch = hist_state.shape[0]
        flat = hist_state.reshape(batch, -1)
        out = self.net(flat)
        return out.reshape(batch, self.horizon, self.state_dim)
