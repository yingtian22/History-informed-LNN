"""Slice full trajectories into history/future windows for HiLNN training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_windows_from_file(
    input_path: Path,
    save_path: Path,
    history_len: int,
    horizon: int,
    stride: int = 1,
) -> None:
    data = np.load(input_path)
    states = data["states"]
    q = data["q"]
    qdot = data["qdot"]
    energy = data["energy"]
    params = data["params"]

    n_traj, t_len = states.shape[:2]
    min_len = history_len + horizon
    if t_len < min_len:
        raise ValueError(f"Trajectory length {t_len} < history_len + horizon ({min_len})")

    hist_q_list = []
    hist_qdot_list = []
    hist_state_list = []
    future_q_list = []
    future_qdot_list = []
    future_state_list = []
    params_list = []
    energy_hist_list = []
    energy_future_list = []

    for i in range(n_traj):
        for start in range(0, t_len - min_len + 1, stride):
            hist_end = start + history_len
            future_end = hist_end + horizon
            hist_q_list.append(q[i, start:hist_end])
            hist_qdot_list.append(qdot[i, start:hist_end])
            hist_state_list.append(states[i, start:hist_end])
            future_q_list.append(q[i, hist_end:future_end])
            future_qdot_list.append(qdot[i, hist_end:future_end])
            future_state_list.append(states[i, hist_end:future_end])
            params_list.append(params[i])
            energy_hist_list.append(energy[i, start:hist_end])
            energy_future_list.append(energy[i, hist_end:future_end])

    out = {
        "hist_q": np.stack(hist_q_list, axis=0),
        "hist_qdot": np.stack(hist_qdot_list, axis=0),
        "hist_state": np.stack(hist_state_list, axis=0),
        "future_q": np.stack(future_q_list, axis=0),
        "future_qdot": np.stack(future_qdot_list, axis=0),
        "future_state": np.stack(future_state_list, axis=0),
        "params": np.stack(params_list, axis=0),
        "energy_hist": np.stack(energy_hist_list, axis=0),
        "energy_future": np.stack(energy_future_list, axis=0),
        "history_len": np.array(history_len),
        "horizon": np.array(horizon),
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, **out)
    print(
        f"Saved {save_path} | windows={out['hist_q'].shape[0]}, "
        f"L={history_len}, H={horizon}"
    )


def build_all_splits(dataset_dir: Path, history_len: int, horizon: int, stride: int) -> None:
    for split in ("train", "val", "test"):
        inp = dataset_dir / f"{split}.npz"
        out = dataset_dir / f"{split}_windows_L{history_len}_H{horizon}.npz"
        if not inp.exists():
            raise FileNotFoundError(f"Missing input file: {inp}")
        build_windows_from_file(inp, out, history_len, horizon, stride)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build history/future window datasets")
    parser.add_argument("--input", type=str, default=None, help="Input .npz (single file mode)")
    parser.add_argument("--save", type=str, default=None, help="Output .npz (single file mode)")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset folder name under data/processed/, e.g. pendulum",
    )
    parser.add_argument("--history_len", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    if args.input and args.save:
        build_windows_from_file(
            Path(args.input),
            Path(args.save),
            args.history_len,
            args.horizon,
            args.stride,
        )
        return

    if args.dataset:
        dataset_dir = PROJECT_ROOT / "data" / "processed" / args.dataset
        build_all_splits(dataset_dir, args.history_len, args.horizon, args.stride)
        return

    parser.error("Provide either (--input and --save) or --dataset")


if __name__ == "__main__":
    main()
