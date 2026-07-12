"""Train HiLNN v1 on window datasets."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim import Adam

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataset import make_dataloader  # noqa: E402
from src.models.hilnn import HiLNN  # noqa: E402
from src.training.hilnn_losses import hilnn_loss  # noqa: E402
from src.utils.logger import CsvLogger  # noqa: E402
from src.utils.metrics import compute_metrics, final_step_mse, sequence_mse  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "system" in cfg:
        cfg.setdefault("dataset", cfg["system"])
    return cfg


def build_hilnn(cfg: dict[str, Any]) -> HiLNN:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HiLNN v1")
    p.add_argument("--config", type=str, default="configs/hilnn_pendulum.yaml")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--history_len", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--qdot_weight", type=float, default=None)
    p.add_argument("--init_vel_weight", type=float, default=None)
    p.add_argument("--energy_weight", type=float, default=None)
    p.add_argument("--train_integrator", type=str, default=None)
    p.add_argument("--eval_integrator", type=str, default=None)
    p.add_argument("--train_detach", action="store_true", default=None)
    p.add_argument("--no_train_detach", action="store_true")
    p.add_argument("--tag", type=str, default=None, help="Variant tag for checkpoint/log filenames")
    p.add_argument("--max_train_batches", type=int, default=None, help="Cap train batches per epoch (quick runs)")
    return p.parse_args()


def merge_cfg(cfg: dict, args: argparse.Namespace) -> dict:
    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.history_len is not None:
        cfg["history_len"] = args.history_len
    if args.horizon is not None:
        cfg["horizon"] = args.horizon
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device:
        cfg["device"] = args.device
    tr = cfg.setdefault("training", {})
    if args.epochs is not None:
        tr["epochs"] = args.epochs
    if args.batch_size is not None:
        tr["batch_size"] = args.batch_size
    if args.lr is not None:
        tr["lr"] = args.lr
    loss = cfg.setdefault("loss", {})
    if args.qdot_weight is not None:
        loss["qdot_weight"] = args.qdot_weight
    if args.init_vel_weight is not None:
        loss["init_vel_weight"] = args.init_vel_weight
    if args.energy_weight is not None:
        loss["energy_weight"] = args.energy_weight
    ro = cfg.setdefault("rollout", {})
    if args.train_integrator:
        ro["train_integrator"] = args.train_integrator
    if args.eval_integrator:
        ro["eval_integrator"] = args.eval_integrator
    if args.train_detach:
        ro["train_detach"] = True
    if args.no_train_detach:
        ro["train_detach"] = False
    if args.tag:
        cfg["tag"] = args.tag
    if args.max_train_batches is not None:
        cfg.setdefault("training", {})["max_train_batches"] = args.max_train_batches
    cfg.setdefault("dataset", "pendulum")
    return cfg


def model_stem(cfg: dict) -> str:
    l, h = int(cfg["history_len"]), int(cfg["horizon"])
    tag = cfg.get("tag")
    if tag:
        return f"hilnn_{tag}_L{l}_H{h}"
    return f"hilnn_L{l}_H{h}"


def ckpt_path(cfg: dict) -> Path:
    ds = cfg["dataset"]
    return PROJECT_ROOT / "outputs" / "checkpoints" / ds / f"{model_stem(cfg)}_best.pt"


def log_path(cfg: dict) -> Path:
    ds = cfg["dataset"]
    return PROJECT_ROOT / "outputs" / "logs" / ds / f"{model_stem(cfg)}_train_log.csv"


def pred_path(cfg: dict) -> Path:
    ds = cfg["dataset"]
    return PROJECT_ROOT / "outputs" / "predictions" / ds / f"{model_stem(cfg)}_test_predictions.npz"


def forward_batch(model, batch, cfg, for_training: bool):
    ro = cfg.get("rollout", {})
    dt = float(cfg.get("dt", 0.05))
    h = int(cfg["horizon"])
    if for_training:
        method = str(ro.get("train_integrator", "euler"))
        detach = bool(ro.get("train_detach", True))
    else:
        method = str(ro.get("eval_integrator", "rk4"))
        detach = False
    return model(batch["hist_q"], dt, h, method=method, detach_between_steps=detach)


def run_epoch(model, loader, optimizer, cfg, device, train: bool) -> dict[str, float]:
    loss_cfg = cfg.get("loss", {})
    if train:
        model.train()
    else:
        model.eval()

    keys = ["loss", "q_loss", "qdot_loss", "init_vel_loss", "energy_loss"]
    sums = {k: 0.0 for k in keys}
    n = 0
    max_batches = None
    if train:
        max_batches = cfg.get("training", {}).get("max_train_batches")

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        if train:
            optimizer.zero_grad()
            pred, z, qdot0 = forward_batch(model, batch, cfg, True)
            loss, parts = hilnn_loss(
                pred, batch["future_state"], qdot0, batch["hist_state"],
                batch["params"],
                q_weight=float(loss_cfg.get("q_weight", 1.0)),
                qdot_weight=float(loss_cfg.get("qdot_weight", 0.1)),
                init_vel_weight=float(loss_cfg.get("init_vel_weight", 0.1)),
                energy_weight=float(loss_cfg.get("energy_weight", 0.0)),
                energy_future=batch["energy_future"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("grad_clip", 1.0)))
            optimizer.step()
        else:
            with torch.no_grad():
                pred, z, qdot0 = forward_batch(model, batch, cfg, False)
                _, parts = hilnn_loss(
                    pred, batch["future_state"], qdot0, batch["hist_state"],
                    batch["params"],
                    q_weight=float(loss_cfg.get("q_weight", 1.0)),
                    qdot_weight=float(loss_cfg.get("qdot_weight", 0.1)),
                    init_vel_weight=float(loss_cfg.get("init_vel_weight", 0.1)),
                    energy_weight=float(loss_cfg.get("energy_weight", 0.0)),
                    energy_future=batch["energy_future"],
                )

        for k in keys:
            if k in parts:
                sums[k] += float(parts[k].item())
        n += 1

    return {k: sums[k] / max(n, 1) for k in keys}


@torch.no_grad()
def validate_metrics(model, loader, cfg, device) -> dict[str, float]:
    model.eval()
    preds, trues, energies, params_list = [], [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred, _, _ = forward_batch(model, batch, cfg, False)
        preds.append(pred.cpu())
        trues.append(batch["future_state"].cpu())
        energies.append(batch["energy_future"].cpu())
        params_list.append(batch["params"].cpu())
    pred_cat = torch.cat(preds)
    true_cat = torch.cat(trues)
    m = compute_metrics(pred_cat, true_cat, torch.cat(energies), torch.cat(params_list))
    m["loss"] = float(sequence_mse(pred_cat, true_cat).item())
    m["final_mse"] = float(final_step_mse(pred_cat, true_cat).item())
    return m


@torch.no_grad()
def save_test_predictions(model, loader, cfg, device) -> None:
    model.eval()
    preds, trues, hists, hist_qs, zs, qdot0_p, qdot0_t, params_l, e_fut, e_pred = [], [], [], [], [], [], [], [], [], []
    from src.utils.metrics import batch_pendulum_energy

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred, z, qdot0 = forward_batch(model, batch, cfg, False)
        preds.append(pred.cpu())
        trues.append(batch["future_state"].cpu())
        hists.append(batch["hist_state"].cpu())
        hist_qs.append(batch["hist_q"].cpu())
        zs.append(z.cpu())
        qdot0_p.append(qdot0.cpu())
        qdot0_t.append(batch["hist_state"][:, -1, 1:2].cpu())
        params_l.append(batch["params"].cpu())
        e_fut.append(batch["energy_future"].cpu())
        pe = batch_pendulum_energy(pred[..., :1], pred[..., 1:2], batch["params"])
        e_pred.append(pe.cpu())

    out = pred_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    tr = torch.cat(trues)
    pr = torch.cat(preds)
    np.savez_compressed(
        out,
        pred_state=pr.numpy(),
        true_state=tr.numpy(),
        hist_state=torch.cat(hists).numpy(),
        hist_q=torch.cat(hist_qs).numpy(),
        true_q=tr[..., :1].numpy(),
        true_qdot=tr[..., 1:2].numpy(),
        pred_q=pr[..., :1].numpy(),
        pred_qdot=pr[..., 1:2].numpy(),
        z=torch.cat(zs).numpy(),
        qdot0_pred=torch.cat(qdot0_p).numpy(),
        qdot0_true=torch.cat(qdot0_t).numpy(),
        params=torch.cat(params_l).numpy(),
        energy_future=torch.cat(e_fut).numpy(),
        pred_energy=torch.cat(e_pred).numpy(),
    )
    print(f"Saved test predictions: {out}")


def main() -> None:
    args = parse_args()
    cfg = merge_cfg(load_config(args.config), args)
    set_seed(int(cfg.get("seed", 42)))

    if str(cfg.get("device", "cpu")).startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(cfg.get("device", "cpu"))
    print(f"Using device: {device}")

    l, h = int(cfg["history_len"]), int(cfg["horizon"])
    bs = int(cfg["training"]["batch_size"])
    ds = cfg["dataset"]

    train_loader = make_dataloader(PROJECT_ROOT, ds, "train", l, h, bs, shuffle=True)
    val_loader = make_dataloader(PROJECT_ROOT, ds, "val", l, h, bs, shuffle=False)
    test_loader = make_dataloader(PROJECT_ROOT, ds, "test", l, h, bs, shuffle=False)

    model = build_hilnn(cfg).to(device)
    lr = float(cfg["training"]["lr"])
    wd = float(cfg["training"].get("weight_decay", 0.0))
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)

    ro = cfg.get("rollout", {})
    print(
        f"HiLNN train: {ro.get('train_integrator', 'euler')} detach={ro.get('train_detach', True)} | "
        f"eval: {ro.get('eval_integrator', 'rk4')}"
    )

    energy_w = float(cfg.get("loss", {}).get("energy_weight", 0.0))
    log_cols = [
        "epoch", "train_loss", "train_q_loss", "train_qdot_loss", "train_init_vel_loss",
        "val_loss", "val_q_loss", "val_qdot_loss", "val_init_vel_loss",
        "val_mse", "val_final_mse", "lr", "time",
    ]
    if energy_w > 0.0:
        log_cols.insert(6, "train_energy_loss")
        log_cols.insert(12, "val_energy_loss")
    logger = CsvLogger(log_path(cfg), log_cols)

    best_val = float("inf")
    bad = 0
    patience = int(cfg["training"].get("early_stop_patience", 15))
    epochs = int(cfg["training"]["epochs"])

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, optimizer, cfg, device, True)
        va = run_epoch(model, val_loader, optimizer, cfg, device, False)
        vm = validate_metrics(model, val_loader, cfg, device)
        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": f"{tr['loss']:.6e}",
            "train_q_loss": f"{tr['q_loss']:.6e}",
            "train_qdot_loss": f"{tr['qdot_loss']:.6e}",
            "train_init_vel_loss": f"{tr['init_vel_loss']:.6e}",
            "val_loss": f"{va['loss']:.6e}",
            "val_q_loss": f"{va['q_loss']:.6e}",
            "val_qdot_loss": f"{va['qdot_loss']:.6e}",
            "val_init_vel_loss": f"{va['init_vel_loss']:.6e}",
            "val_mse": f"{vm['mse']:.6e}",
            "val_final_mse": f"{vm['final_mse']:.6e}",
            "lr": lr,
            "time": f"{elapsed:.2f}",
        }
        if energy_w > 0.0:
            row["train_energy_loss"] = f"{tr.get('energy_loss', 0.0):.6e}"
            row["val_energy_loss"] = f"{va.get('energy_loss', 0.0):.6e}"
        logger.log(row)
        print(
            f"Epoch {epoch}/{epochs} | train={tr['loss']:.4e} val={va['loss']:.4e} "
            f"val_mse={vm['mse']:.4e} final_mse={vm['final_mse']:.4e}"
        )

        if va["loss"] < best_val:
            best_val = va["loss"]
            bad = 0
            p = ckpt_path(cfg)
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"config": cfg, "model_state_dict": model.state_dict()}, p)
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    ckpt = torch.load(ckpt_path(cfg), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    save_test_predictions(model, test_loader, cfg, device)
    print(f"Best checkpoint: {ckpt_path(cfg)}")


if __name__ == "__main__":
    main()
