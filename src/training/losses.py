"""Training losses for baselines."""

from __future__ import annotations

import torch


def sequence_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    err = (pred - target) ** 2
    if weights is None:
        return err.mean()
    w = weights.view(1, -1, 1)
    return (err * w).sum() / w.sum()


def final_step_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred[:, -1] - target[:, -1]) ** 2)


def rollout_loss(pred_state: torch.Tensor, true_state: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    return sequence_mse(pred_state, true_state, weights)


def one_step_loss(pred_next: torch.Tensor, true_next: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred_next - true_next) ** 2)
