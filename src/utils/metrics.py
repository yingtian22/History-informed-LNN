"""Evaluation metrics for rollout predictions."""

from __future__ import annotations

import numpy as np
import torch


def batch_pendulum_energy(q: torch.Tensor, qdot: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Mechanical energy using per-sample [g, l, m, c] params. q, qdot: [B, H, q_dim]."""
    g, l, m = params[:, 0:1, None], params[:, 1:2, None], params[:, 2:3, None]
    kinetic = 0.5 * m * (l ** 2) * (qdot ** 2)
    potential = m * g * l * (1.0 - torch.cos(q))
    return (kinetic + potential).squeeze(-1)


def sequence_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def final_step_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred[:, -1] - target[:, -1]) ** 2)


def stepwise_mse(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    err = (pred - target) ** 2
    return err.mean(dim=(0, 2)).detach().cpu().numpy()


def energy_mse(pred_energy: torch.Tensor, true_energy: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred_energy - true_energy) ** 2)


def energy_drift(pred_energy: torch.Tensor) -> tuple[float, float]:
    drift = pred_energy[:, 1:] - pred_energy[:, :-1]
    return float(drift.mean().item()), float(drift[:, -1].mean().item())


def failure_rate(pred: torch.Tensor) -> float:
    bad = ~torch.isfinite(pred)
    return float(bad.any(dim=(1, 2)).float().mean().item())


def compute_metrics(
    pred_state: torch.Tensor,
    true_state: torch.Tensor,
    true_energy: torch.Tensor,
    params: torch.Tensor,
) -> dict[str, float]:
    pred_q = pred_state[..., :1]
    pred_qdot = pred_state[..., 1:2]
    pred_energy = batch_pendulum_energy(pred_q, pred_qdot, params)

    mse = float(sequence_mse(pred_state, true_state).item())
    mae = float(torch.mean(torch.abs(pred_state - true_state)).item())
    final_mse = float(final_step_mse(pred_state, true_state).item())
    e_mse = float(energy_mse(pred_energy, true_energy).item())
    drift_mean, drift_final = energy_drift(pred_energy)
    fail = failure_rate(pred_state)

    return {
        "mse": mse,
        "mae": mae,
        "final_mse": final_mse,
        "energy_mse": e_mse,
        "energy_drift_mean": drift_mean,
        "energy_drift_final": drift_final,
        "failure_rate": fail,
    }
