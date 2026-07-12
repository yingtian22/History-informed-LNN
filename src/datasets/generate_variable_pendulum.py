"""Generate variable-parameter pendulum train/val/test datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets._common import add_config_arg, generate_splits, load_config  # noqa: E402


def sample_params(rng: np.random.Generator, cfg: dict) -> np.ndarray:
    l_lo, l_hi = cfg["l_range"]
    m_lo, m_hi = cfg["m_range"]
    c_lo, c_hi = cfg["c_range"]
    l = rng.uniform(l_lo, l_hi)
    m = rng.uniform(m_lo, m_hi)
    c = rng.uniform(c_lo, c_hi)
    return np.array([cfg["g"], l, m, c], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate variable-parameter pendulum dataset")
    add_config_arg(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    generate_splits(cfg, lambda rng: sample_params(rng, cfg), int(cfg["seed"]))


if __name__ == "__main__":
    main()
