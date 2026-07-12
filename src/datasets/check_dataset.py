"""Validate generated datasets and plot sample trajectories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_TRAJ_KEYS = ("states", "q", "qdot", "times", "params", "energy")
REQUIRED_WINDOW_KEYS = (
    "hist_q",
    "hist_qdot",
    "hist_state",
    "future_q",
    "future_qdot",
    "future_state",
    "params",
    "energy_hist",
    "energy_future",
)


def check_arrays(data: dict, window_mode: bool) -> None:
    keys = REQUIRED_WINDOW_KEYS if window_mode else REQUIRED_TRAJ_KEYS
    missing = [k for k in keys if k not in data]
    if missing:
        raise KeyError(f"Missing keys: {missing}")

    for key in keys:
        arr = data[key]
        if not np.issubdtype(arr.dtype, np.floating) and key not in ("history_len", "horizon"):
            continue
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"Non-finite values found in '{key}'")
        print(f"  {key:16s} shape={arr.shape}, dtype={arr.dtype}")

    if window_mode:
        n = data["hist_q"].shape[0]
        l, h = data["hist_q"].shape[1], data["future_q"].shape[1]
        assert data["future_q"].shape == (n, h, data["hist_q"].shape[2])
        assert data["hist_qdot"].shape == data["hist_q"].shape
        print(f"  OK: {n} windows, L={l}, H={h}")
    else:
        n, t = data["states"].shape[:2]
        assert data["q"].shape[:2] == (n, t)
        assert data["energy"].shape == (n, t)
        print(f"  OK: {n} trajectories, T={t}, dt≈{data['times'][1] - data['times'][0]:.4f}")


def plot_trajectories(data: dict, save_fig: Path, window_mode: bool, num_samples: int = 3) -> None:
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3 * num_samples))
    if num_samples == 1:
        axes = np.array([axes])

    if window_mode:
        n = data["hist_q"].shape[0]
        idx = rng.choice(n, size=min(num_samples, n), replace=False)
        for row, i in enumerate(idx):
            t_hist = np.arange(data["hist_q"].shape[1])
            t_fut = np.arange(data["future_q"].shape[1]) + data["hist_q"].shape[1]
            axes[row, 0].plot(t_hist, data["hist_q"][i, :, 0], label="hist q")
            axes[row, 0].plot(t_fut, data["future_q"][i, :, 0], label="future q")
            axes[row, 0].set_title(f"Window {i}: position")
            axes[row, 0].legend(fontsize=8)
            e = np.concatenate([data["energy_hist"][i], data["energy_future"][i]])
            axes[row, 1].plot(np.arange(len(e)), e)
            axes[row, 1].set_title(f"Window {i}: energy")
    else:
        n = data["states"].shape[0]
        idx = rng.choice(n, size=min(num_samples, n), replace=False)
        times = data["times"]
        for row, i in enumerate(idx):
            axes[row, 0].plot(times, data["q"][i, :, 0], label="q")
            axes[row, 0].plot(times, data["qdot"][i, :, 0], label="qdot", alpha=0.7)
            axes[row, 0].set_title(f"Trajectory {i}")
            axes[row, 0].legend(fontsize=8)
            axes[row, 1].plot(times, data["energy"][i])
            axes[row, 1].set_title(f"Trajectory {i}: energy")

    fig.tight_layout()
    save_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_fig, dpi=120)
    plt.close(fig)
    print(f"  Saved figure: {save_fig}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check dataset integrity and visualize samples")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--save_fig", type=str, default=None)
    parser.add_argument("--window_mode", action="store_true")
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()

    path = Path(args.input)
    print(f"Checking {path} ...")
    data = dict(np.load(path))
    check_arrays(data, args.window_mode)

    if args.save_fig:
        plot_trajectories(data, Path(args.save_fig), args.window_mode, args.num_samples)


if __name__ == "__main__":
    main()
