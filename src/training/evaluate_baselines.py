"""Evaluate all trained baselines and write metrics CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataset import make_dataloader  # noqa: E402
from src.models.hilnn import HiLNN  # noqa: E402
from src.training.trainer_utils import (  # noqa: E402
    MODEL_NAMES,
    build_model,
    checkpoint_path,
    forward_predictions,
    prediction_path,
)

ALL_MODELS = list(MODEL_NAMES) + ["hilnn"]
from src.utils.metrics import compute_metrics, stepwise_mse  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Stage-2 baselines")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--history_len", type=int, default=8)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=ALL_MODELS,
        choices=ALL_MODELS,
    )
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--hilnn_tag", type=str, default=None, help="HiLNN variant tag (e.g. v1c_energy001)")
    p.add_argument(
        "--output_table",
        type=str,
        default=None,
        help="Override metrics CSV filename under outputs/tables/",
    )
    return p.parse_args()


def hilnn_stem(history_len: int, horizon: int, tag: str | None = None) -> str:
    if tag:
        return f"hilnn_{tag}_L{history_len}_H{horizon}"
    return f"hilnn_L{history_len}_H{horizon}"


def hilnn_ckpt_path(dataset: str, history_len: int, horizon: int, tag: str | None = None) -> Path:
    return PROJECT_ROOT / "outputs" / "checkpoints" / dataset / f"{hilnn_stem(history_len, horizon, tag)}_best.pt"


def build_hilnn_from_ckpt(cfg: dict) -> HiLNN:
    enc = cfg.get("encoder", {})
    lag = cfg.get("lagrangian", {})
    return HiLNN(
        q_dim=int(cfg.get("q_dim", 1)),
        history_len=int(cfg.get("history_len", 8)),
        context_dim=int(enc.get("context_dim", 32)),
        encoder_hidden_dim=int(enc.get("hidden_dim", 64)),
        encoder_num_layers=int(enc.get("num_layers", 1)),
        core_hidden_dim=int(lag.get("hidden_dim", 128)),
        eps=float(lag.get("eps", 1e-3)),
    )


@torch.no_grad()
def evaluate_hilnn(model: HiLNN, loader, horizon: int, dt: float, device: torch.device, cfg: dict):
    from src.utils.metrics import batch_pendulum_energy

    model.eval()
    ro = cfg.get("rollout", {})
    method = str(ro.get("eval_integrator", "rk4"))
    all_pred, all_true, all_energy, all_params = [], [], [], []
    all_z, all_qdot0 = [], []
    step_err_sum = None
    n_samples = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred, z, qdot0 = model(batch["hist_q"], dt, horizon, method=method, detach_between_steps=False)
        all_pred.append(pred.cpu())
        all_true.append(batch["future_state"].cpu())
        all_energy.append(batch["energy_future"].cpu())
        all_params.append(batch["params"].cpu())
        all_z.append(z.cpu())
        all_qdot0.append(qdot0.cpu())
        err = stepwise_mse(pred, batch["future_state"])
        n = batch["future_state"].shape[0]
        step_err_sum = err * n if step_err_sum is None else step_err_sum + err * n
        n_samples += n
    pred_cat = torch.cat(all_pred)
    true_cat = torch.cat(all_true)
    params_cat = torch.cat(all_params)
    metrics = compute_metrics(pred_cat, true_cat, torch.cat(all_energy), params_cat)
    metrics["stepwise_mse"] = (step_err_sum / max(n_samples, 1)).tolist()
    pred_energy = batch_pendulum_energy(pred_cat[..., :1], pred_cat[..., 1:2], params_cat)
    extras = {
        "z": torch.cat(all_z),
        "qdot0_pred": torch.cat(all_qdot0),
        "pred_energy": pred_energy,
    }
    return metrics, pred_cat, true_cat, params_cat, torch.cat(all_energy), extras


@torch.no_grad()
def evaluate_model(model, model_name, loader, horizon, dt, device):
    model.eval()
    all_pred, all_true, all_energy, all_params = [], [], [], []
    step_err_sum = None
    n_samples = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = forward_predictions(model, model_name, batch, horizon, dt)
        all_pred.append(pred.cpu())
        all_true.append(batch["future_state"].cpu())
        all_energy.append(batch["energy_future"].cpu())
        all_params.append(batch["params"].cpu())
        err = stepwise_mse(pred, batch["future_state"])
        if step_err_sum is None:
            step_err_sum = err * batch["future_state"].shape[0]
        else:
            step_err_sum += err * batch["future_state"].shape[0]
        n_samples += batch["future_state"].shape[0]

    pred_cat = torch.cat(all_pred)
    true_cat = torch.cat(all_true)
    energy_cat = torch.cat(all_energy)
    params_cat = torch.cat(all_params)
    metrics = compute_metrics(pred_cat, true_cat, energy_cat, params_cat)
    metrics["stepwise_mse"] = (step_err_sum / max(n_samples, 1)).tolist()
    return metrics, pred_cat, true_cat, params_cat, energy_cat


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    loader = make_dataloader(
        PROJECT_ROOT, args.dataset, "test", args.history_len, args.horizon,
        args.batch_size, shuffle=False,
    )
    sample = next(iter(loader))
    state_dim = int(sample["hist_state"].shape[-1])

    table_name = args.output_table or f"{args.dataset}_baseline_metrics_L{args.history_len}_H{args.horizon}.csv"
    table_path = PROJECT_ROOT / "outputs" / "tables" / table_name
    table_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model", "mse", "mae", "final_mse", "energy_mse",
        "energy_drift_mean", "energy_drift_final", "failure_rate",
    ]
    rows = []

    for model_name in args.models:
        if model_name == "hilnn":
            ckpt_file = hilnn_ckpt_path(args.dataset, args.history_len, args.horizon, args.hilnn_tag)
        else:
            ckpt_file = checkpoint_path(PROJECT_ROOT, args.dataset, model_name, args.history_len, args.horizon)
        if not ckpt_file.exists():
            print(f"Skip {model_name}: checkpoint not found at {ckpt_file}")
            continue

        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})

        if model_name == "hilnn":
            model = build_hilnn_from_ckpt(cfg).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            metrics, pred, true, params, energy, extras = evaluate_hilnn(
                model, loader, args.horizon, args.dt, device, cfg
            )
        else:
            model = build_model(model_name, state_dim, args.history_len, args.horizon, cfg).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            metrics, pred, true, params, energy = evaluate_model(
                model, model_name, loader, args.horizon, args.dt, device
            )
        rows.append({"model": model_name, **{k: metrics[k] for k in fieldnames if k != "model"}})
        print(f"{model_name}: mse={metrics['mse']:.4e} final_mse={metrics['final_mse']:.4e}")

        if args.save_predictions:
            from src.utils.metrics import batch_pendulum_energy

            if model_name == "hilnn":
                stem = hilnn_stem(args.history_len, args.horizon, args.hilnn_tag)
                out = PROJECT_ROOT / "outputs" / "predictions" / args.dataset / f"{stem}_test_predictions.npz"
            else:
                out = prediction_path(PROJECT_ROOT, args.dataset, model_name, args.history_len, args.horizon)
            out.parent.mkdir(parents=True, exist_ok=True)
            pred_energy = batch_pendulum_energy(pred[..., :1], pred[..., 1:2], params)
            save_kw = dict(
                pred_state=pred.numpy(),
                true_state=true.numpy(),
                params=params.numpy(),
                true_energy=energy.numpy(),
                pred_energy=pred_energy.numpy(),
            )
            if model_name == "hilnn":
                save_kw["z"] = extras["z"].numpy()
                save_kw["qdot0_pred"] = extras["qdot0_pred"].numpy()
            np.savez_compressed(out, **save_kw)

    with open(table_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"Saved metrics table: {table_path}")


if __name__ == "__main__":
    main()
