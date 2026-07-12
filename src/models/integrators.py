"""Shared numerical integrators for dynamical baselines."""

from __future__ import annotations

from typing import Callable

import torch


DynamicsFn = Callable[[torch.Tensor], torch.Tensor]


def euler_step(dynamics: DynamicsFn, x: torch.Tensor, dt: float) -> torch.Tensor:
    return x + dt * dynamics(x)


def rk4_step(dynamics: DynamicsFn, x: torch.Tensor, dt: float) -> torch.Tensor:
    k1 = dynamics(x)
    k2 = dynamics(x + 0.5 * dt * k1)
    k3 = dynamics(x + 0.5 * dt * k2)
    k4 = dynamics(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout(
    dynamics: DynamicsFn,
    x0: torch.Tensor,
    dt: float,
    horizon: int,
    method: str = "rk4",
    detach_between_steps: bool = False,
) -> torch.Tensor:
    """
    Roll out state trajectory.

    Args:
        x0: [B, state_dim]
        detach_between_steps: if True, truncate BPTT between steps (stable multistep training)
    Returns:
        states: [B, horizon, state_dim] (steps t+1 ... t+H)
    """
    step_fn = rk4_step if method == "rk4" else euler_step
    states = []
    x = x0
    for _ in range(horizon):
        x = step_fn(dynamics, x, dt)
        states.append(x)  # keep graph link for this step's loss
        if detach_between_steps:
            x = x.detach()  # truncated BPTT: next step does not backprop further back
    return torch.stack(states, dim=1)
