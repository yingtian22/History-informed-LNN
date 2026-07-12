"""Generate damped pendulum train/val/test datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets._common import add_config_arg, generate_splits, load_config  # noqa: E402


def sample_params(_rng: np.random.Generator, cfg: dict) -> np.ndarray:
    return np.array([cfg["g"], cfg["l"], cfg["m"], cfg["c"]], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate damped pendulum dataset")
    add_config_arg(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    generate_splits(cfg, lambda rng: sample_params(rng, cfg), int(cfg["seed"]))


if __name__ == "__main__":
    main()
