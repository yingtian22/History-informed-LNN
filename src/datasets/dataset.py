"""PyTorch Dataset for Stage 1 window .npz files."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

Split = Literal["train", "val", "test"]


class WindowDataset(Dataset):
    """Load history/future windows from processed npz files."""

    def __init__(self, npz_path: str | Path):
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        data = np.load(path)
        self.hist_q = torch.from_numpy(data["hist_q"]).float()
        self.hist_qdot = torch.from_numpy(data["hist_qdot"]).float()
        self.hist_state = torch.from_numpy(data["hist_state"]).float()
        self.future_q = torch.from_numpy(data["future_q"]).float()
        self.future_qdot = torch.from_numpy(data["future_qdot"]).float()
        self.future_state = torch.from_numpy(data["future_state"]).float()
        self.params = torch.from_numpy(data["params"]).float()
        self.energy_hist = torch.from_numpy(data["energy_hist"]).float()
        self.energy_future = torch.from_numpy(data["energy_future"]).float()

        self.history_len = int(self.hist_state.shape[1])
        self.horizon = int(self.future_state.shape[1])
        self.state_dim = int(self.hist_state.shape[-1])
        self.q_dim = int(self.hist_q.shape[-1])

    def __len__(self) -> int:
        return int(self.hist_state.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "hist_q": self.hist_q[idx],
            "hist_qdot": self.hist_qdot[idx],
            "hist_state": self.hist_state[idx],
            "future_q": self.future_q[idx],
            "future_qdot": self.future_qdot[idx],
            "future_state": self.future_state[idx],
            "params": self.params[idx],
            "energy_hist": self.energy_hist[idx],
            "energy_future": self.energy_future[idx],
            "x0": self.hist_state[idx, -1],
            "y_future": self.future_state[idx],
        }


def window_npz_path(
    project_root: Path,
    dataset: str,
    split: Split,
    history_len: int,
    horizon: int,
) -> Path:
    return (
        project_root
        / "data"
        / "processed"
        / dataset
        / f"{split}_windows_L{history_len}_H{horizon}.npz"
    )


def make_dataloader(
    project_root: Path,
    dataset: str,
    split: Split,
    history_len: int,
    horizon: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    path = window_npz_path(project_root, dataset, split, history_len, horizon)
    ds = WindowDataset(path)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=split == "train",
    )
