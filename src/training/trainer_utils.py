"""Shared helpers for baseline training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from src.models.hnn import HNN
from src.models.lnn import LNN
from src.models.mlp import MlpDirectMultiStep, MlpOneStep
from src.models.neural_ode import NeuralODE
from src.training.losses import one_step_loss, rollout_loss
from src.utils.metrics import batch_pendulum_energy, compute_metrics, final_step_mse, sequence_mse

MODEL_NAMES = (
    "mlp_one_step",
    "mlp_direct",
    "neural_ode",
    "hnn",
    "lnn",
    "lnn_multistep",
)

# Dynamical baselines: training rollout settings (eval always uses rk4, no detach)
_ROLLOUT_MODELS = frozenset({"neural_ode", "hnn", "lnn", "lnn_multistep"})


def _rollout_kwargs(model_name: str, cfg: dict[str, Any], for_training: bool) -> dict[str, Any]:
    if model_name not in _ROLLOUT_MODELS:
        return {}
    if for_training:
        # lnn_multistep: euler rollout in training (stable); rk4 for val/test
        default_detach = model_name != "lnn_multistep"
        default_integrator = "euler" if model_name == "lnn_multistep" else "rk4"
        return {
            "detach_between_steps": bool(cfg.get("train_detach", default_detach)),
            "method": str(cfg.get("train_integrator", default_integrator)),
        }
    return {"detach_between_steps": False, "method": "rk4"}


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_model(
    model_name: str,
    state_dim: int,
    history_len: int,
    horizon: int,
    cfg: dict[str, Any],
) -> torch.nn.Module:
    hidden = int(cfg.get("hidden_dim", 128))
    layers = int(cfg.get("num_layers", 3))

    if model_name == "mlp_one_step":
        return MlpOneStep(state_dim, hidden_dim=hidden, num_layers=layers)
    if model_name == "mlp_direct":
        h = int(cfg.get("hidden_dim", 256))
        l = int(cfg.get("num_layers", 4))
        return MlpDirectMultiStep(history_len, state_dim, horizon, hidden_dim=h, num_layers=l)
    if model_name == "neural_ode":
        return NeuralODE(state_dim, hidden_dim=hidden, num_layers=layers, method=str(cfg.get("ode_method", "rk4")))
    if model_name in ("hnn",):
        return HNN(state_dim, hidden_dim=hidden, num_layers=layers)
    if model_name in ("lnn", "lnn_multistep"):
        return LNN(q_dim=state_dim // 2, hidden_dim=hidden, num_layers=layers)
    raise ValueError(f"Unknown model: {model_name}")


def forward_predictions(
    model: torch.nn.Module,
    model_name: str,
    batch: dict[str, torch.Tensor],
    horizon: int,
    dt: float,
    cfg: dict[str, Any] | None = None,
    for_training: bool = False,
) -> torch.Tensor:
    cfg = cfg or {}
    if model_name == "mlp_direct":
        return model(batch["hist_state"])
    if model_name == "mlp_one_step":
        return model.predict_sequence(batch["x0"], horizon, dt)
    if model_name in _ROLLOUT_MODELS:
        kw = _rollout_kwargs(model_name, cfg, for_training)
        if model_name == "neural_ode":
            kw = {"detach_between_steps": kw["detach_between_steps"], "integrator_method": kw["method"]}
        return model.predict_sequence(batch["x0"], horizon, dt, **kw)
    raise ValueError(model_name)


def training_loss(
    model: torch.nn.Module,
    model_name: str,
    batch: dict[str, torch.Tensor],
    horizon: int,
    dt: float,
    cfg: dict[str, Any] | None = None,
) -> torch.Tensor:
    if model_name == "mlp_one_step":
        pred = model(batch["x0"])
        target = batch["future_state"][:, 0]
        return one_step_loss(pred, target)

    if model_name == "lnn":
        pred = model.predict_sequence(batch["x0"], 1, dt)
        target = batch["future_state"][:, :1]
        return rollout_loss(pred, target)

    pred = forward_predictions(
        model, model_name, batch, horizon, dt, cfg=cfg, for_training=True
    )
    return rollout_loss(pred, batch["future_state"])


@torch.no_grad()
def evaluate_batch(
    model: torch.nn.Module,
    model_name: str,
    batch: dict[str, torch.Tensor],
    horizon: int,
    dt: float,
    cfg: dict[str, Any] | None = None,
) -> dict[str, float]:
    model.eval()
    pred = forward_predictions(
        model, model_name, batch, horizon, dt, cfg=cfg or {}, for_training=False
    )
    target = batch["future_state"]
    pred_q, pred_qdot = pred[..., :1], pred[..., 1:2]
    pred_energy = batch_pendulum_energy(pred_q, pred_qdot, batch["params"])
    metrics = compute_metrics(pred, target, batch["energy_future"], batch["params"])
    metrics["loss"] = float(sequence_mse(pred, target).item())
    metrics["final_mse"] = float(final_step_mse(pred, target).item())
    return metrics


def checkpoint_path(
    project_root: Path,
    dataset: str,
    model_name: str,
    history_len: int,
    horizon: int,
) -> Path:
    return (
        project_root
        / "outputs"
        / "checkpoints"
        / dataset
        / f"{model_name}_L{history_len}_H{horizon}_best.pt"
    )


def log_path(project_root: Path, dataset: str, model_name: str, history_len: int, horizon: int) -> Path:
    return (
        project_root
        / "outputs"
        / "logs"
        / dataset
        / f"{model_name}_L{history_len}_H{horizon}_train_log.csv"
    )


def prediction_path(project_root: Path, dataset: str, model_name: str, history_len: int, horizon: int) -> Path:
    return (
        project_root
        / "outputs"
        / "predictions"
        / dataset
        / f"{model_name}_L{history_len}_H{horizon}_test_predictions.npz"
    )
