"""Loss functions for HiLNN training."""

from __future__ import annotations

import torch

from src.utils.metrics import batch_pendulum_energy


def hilnn_loss(
    pred_state: torch.Tensor,
    true_state: torch.Tensor,
    qdot0_pred: torch.Tensor,
    hist_state: torch.Tensor,
    params: torch.Tensor,
    q_weight: float = 1.0,
    qdot_weight: float = 0.1,
    init_vel_weight: float = 0.1,
    energy_weight: float = 0.0,
    energy_future: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    q_dim = pred_state.shape[-1] // 2
    pred_q = pred_state[..., :q_dim]
    pred_qdot = pred_state[..., q_dim:]
    true_q = true_state[..., :q_dim]
    true_qdot = true_state[..., q_dim:]

    q_loss = torch.mean((pred_q - true_q) ** 2)
    qdot_loss = torch.mean((pred_qdot - true_qdot) ** 2)
    qdot0_true = hist_state[:, -1, q_dim:]
    init_vel_loss = torch.mean((qdot0_pred - qdot0_true) ** 2)

    total = (
        q_weight * q_loss
        + qdot_weight * qdot_loss
        + init_vel_weight * init_vel_loss
    )

    energy_loss = torch.tensor(0.0, device=pred_state.device)
    if energy_weight > 0.0 and energy_future is not None:
        pred_energy = batch_pendulum_energy(pred_q, pred_qdot, params)
        energy_loss = torch.mean((pred_energy - energy_future) ** 2)
        total = total + energy_weight * energy_loss

    parts = {
        "loss": total,
        "q_loss": q_loss,
        "qdot_loss": qdot_loss,
        "init_vel_loss": init_vel_loss,
        "energy_loss": energy_loss,
    }
    return total, parts
