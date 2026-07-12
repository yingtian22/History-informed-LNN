"""Quick HiLNN forward + backward smoke test (no full training)."""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.dataset import WindowDataset
from src.models.hilnn import HiLNN
from src.training.hilnn_losses import hilnn_loss

DATA = ROOT / "data/processed/pendulum/train_windows_L8_H32.npz"


def main() -> None:
    ds = WindowDataset(DATA)
    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))
    model = HiLNN(q_dim=1, history_len=8, context_dim=32)
    model.train()
    pred, z, qdot0 = model(batch["hist_q"], 0.05, 32, method="euler", detach_between_steps=True)
    loss, parts = hilnn_loss(pred, batch["future_state"], qdot0, batch["hist_state"], batch["params"])
    loss.backward()
    gn = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print("OK HiLNN", f"loss={loss.item():.4e}", f"pred={tuple(pred.shape)}", f"z={tuple(z.shape)}", f"grad_norm={gn:.4f}")
    assert torch.isfinite(loss).all()
    assert gn > 0


if __name__ == "__main__":
    main()
