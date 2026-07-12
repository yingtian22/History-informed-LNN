"""Quick forward-pass smoke test (no full training). Requires: pip install torch."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataset import WindowDataset
from src.training.trainer_utils import MODEL_NAMES, build_model, forward_predictions, training_loss

DATA = PROJECT_ROOT / "data/processed/pendulum/train_windows_L8_H32.npz"
H, L, DT = 32, 8, 0.05


def main() -> None:
    ds = WindowDataset(DATA)
    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))
    state_dim = batch["hist_state"].shape[-1]

    for name in MODEL_NAMES:
        model = build_model(name, state_dim, L, H, {})
        loss = training_loss(model, name, batch, H, DT)
        pred = forward_predictions(model, name, batch, H, DT)
        assert torch.isfinite(loss).all(), name
        assert torch.isfinite(pred).all(), name
        print(f"OK {name:16s} loss={loss.item():.4e} pred_shape={tuple(pred.shape)}")


if __name__ == "__main__":
    main()
