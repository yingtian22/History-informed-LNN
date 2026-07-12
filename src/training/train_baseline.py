"""Train a single baseline model on window data."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.optim import Adam

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataset import make_dataloader  # noqa: E402
from src.training.trainer_utils import (  # noqa: E402
    MODEL_NAMES,
    build_model,
    checkpoint_path,
    evaluate_batch,
    load_yaml_config,
    log_path,
    prediction_path,
    training_loss,
)
from src.utils.logger import CsvLogger  # noqa: E402
from src.utils.metrics import batch_pendulum_energy, stepwise_mse  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HiLNN Stage-2 baselines")
    p.add_argument("--model", type=str, required=True, choices=MODEL_NAMES)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--history_len", type=int, default=8)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", type=str, default=None, help="Optional YAML hyperparameters")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=15, help="Early stopping patience on val loss")
    p.add_argument("--max_train_batches", type=int, default=None, help="Cap train batches per epoch")
    return p.parse_args()


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    model_name: str,
    horizon: int,
    dt: float,
    device: torch.device,
    train: bool,
    cfg: dict,
) -> float:
    total_loss = 0.0
    n_batches = 0
    if train:
        model.train()
    else:
        model.eval()

    max_batches = cfg.get("max_train_batches") if train else None
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        if train:
            optimizer.zero_grad()
            loss = training_loss(model, model_name, batch, horizon, dt, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                loss = training_loss(model, model_name, batch, horizon, dt, cfg)
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, model_name, horizon, dt, device, cfg: dict):
    model.eval()
    agg: dict[str, float] = {}
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        m = evaluate_batch(model, model_name, batch, horizon, dt, cfg)
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}


@torch.no_grad()
def save_test_predictions(model, loader, model_name, horizon, dt, device, save_path: Path):
    model.eval()
    preds, trues, hists, params_list, true_e, pred_e = [], [], [], [], [], []
    from src.training.trainer_utils import forward_predictions

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = forward_predictions(model, model_name, batch, horizon, dt)
        preds.append(out.cpu())
        trues.append(batch["future_state"].cpu())
        hists.append(batch["hist_state"].cpu())
        params_list.append(batch["params"].cpu())
        true_e.append(batch["energy_future"].cpu())
        pe = batch_pendulum_energy(out[..., :1], out[..., 1:2], batch["params"])
        pred_e.append(pe.cpu())

    import numpy as np

    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        pred_state=torch.cat(preds).numpy(),
        true_state=torch.cat(trues).numpy(),
        hist_state=torch.cat(hists).numpy(),
        params=torch.cat(params_list).numpy(),
        true_energy=torch.cat(true_e).numpy(),
        pred_energy=torch.cat(pred_e).numpy(),
    )
    print(f"Saved test predictions: {save_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = load_yaml_config(args.config)
    if args.max_train_batches is not None:
        cfg["max_train_batches"] = args.max_train_batches

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA not available (install GPU PyTorch). Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    train_loader = make_dataloader(
        PROJECT_ROOT, args.dataset, "train", args.history_len, args.horizon,
        args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    val_loader = make_dataloader(
        PROJECT_ROOT, args.dataset, "val", args.history_len, args.horizon,
        args.batch_size, shuffle=False, num_workers=args.num_workers,
    )
    test_loader = make_dataloader(
        PROJECT_ROOT, args.dataset, "test", args.history_len, args.horizon,
        args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    sample = next(iter(train_loader))
    state_dim = int(sample["hist_state"].shape[-1])

    model = build_model(args.model, state_dim, args.history_len, args.horizon, cfg).to(device)
    lr = float(cfg.get("lr", args.lr))
    optimizer = Adam(model.parameters(), lr=lr)
    if args.model == "lnn_multistep":
        print(
            f"lnn_multistep train: integrator={cfg.get('train_integrator', 'euler')}, "
            f"detach={cfg.get('train_detach', False)} | eval: rk4"
        )

    ckpt_path = checkpoint_path(PROJECT_ROOT, args.dataset, args.model, args.history_len, args.horizon)
    log_csv = log_path(PROJECT_ROOT, args.dataset, args.model, args.history_len, args.horizon)
    logger = CsvLogger(
        log_csv,
        ["epoch", "train_loss", "val_loss", "val_mse", "val_final_mse", "val_energy_mse", "lr", "time"],
    )

    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(
            model, train_loader, optimizer, args.model, args.horizon, args.dt, device, True, cfg
        )
        val_metrics = validate(model, val_loader, args.model, args.horizon, args.dt, device, cfg)
        elapsed = time.time() - t0

        logger.log({
            "epoch": epoch,
            "train_loss": f"{train_loss:.6e}",
            "val_loss": f"{val_metrics['loss']:.6e}",
            "val_mse": f"{val_metrics['mse']:.6e}",
            "val_final_mse": f"{val_metrics['final_mse']:.6e}",
            "val_energy_mse": f"{val_metrics['energy_mse']:.6e}",
            "lr": lr,
            "time": f"{elapsed:.2f}",
        })
        print(
            f"Epoch {epoch}/{args.epochs} | train={train_loss:.4e} "
            f"val={val_metrics['loss']:.4e} final_mse={val_metrics['final_mse']:.4e}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            bad_epochs = 0
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": args.model,
                    "dataset": args.dataset,
                    "history_len": args.history_len,
                    "horizon": args.horizon,
                    "state_dim": state_dim,
                    "config": cfg,
                    "model_state_dict": model.state_dict(),
                },
                ckpt_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    pred_path = prediction_path(PROJECT_ROOT, args.dataset, args.model, args.history_len, args.horizon)
    save_test_predictions(model, test_loader, args.model, args.horizon, args.dt, device, pred_path)
    print(f"Best checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
