"""Shared utilities for pendulum dataset generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import yaml
from scipy.integrate import solve_ivp


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pendulum_energy(theta: np.ndarray, theta_dot: np.ndarray, g: float, l: float, m: float) -> np.ndarray:
    kinetic = 0.5 * m * (l ** 2) * (theta_dot ** 2)
    potential = m * g * l * (1.0 - np.cos(theta))
    return kinetic + potential


def make_pendulum_rhs(g: float, l: float, c: float = 0.0) -> Callable[[float, np.ndarray], np.ndarray]:
    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        theta, theta_dot = y
        theta_ddot = -(g / l) * np.sin(theta) - c * theta_dot
        return np.array([theta_dot, theta_ddot], dtype=np.float64)

    return rhs


def integrate_trajectory(
    y0: np.ndarray,
    t_eval: np.ndarray,
    g: float,
    l: float,
    c: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sol = solve_ivp(
        make_pendulum_rhs(g, l, c),
        (t_eval[0], t_eval[-1]),
        y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-9,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    theta = sol.y[0]
    theta_dot = sol.y[1]
    states = np.stack([theta, theta_dot], axis=-1)
    return states, theta, theta_dot


def sample_initial_conditions(rng: np.random.Generator, cfg: dict) -> np.ndarray:
    theta_lo, theta_hi = cfg["theta_range"]
    dot_lo, dot_hi = cfg["theta_dot_range"]
    theta = rng.uniform(theta_lo, theta_hi)
    theta_dot = rng.uniform(dot_lo, dot_hi)
    return np.array([theta, theta_dot], dtype=np.float64)


def build_dataset_array(
    rng: np.random.Generator,
    num_traj: int,
    cfg: dict,
    sample_params_fn: Callable[[np.random.Generator], np.ndarray],
) -> dict[str, np.ndarray]:
    t_len = int(cfg["trajectory_length"])
    dt = float(cfg["dt"])
    g = float(cfg["g"])
    t_eval = np.arange(t_len, dtype=np.float64) * dt

    states = np.zeros((num_traj, t_len, 2), dtype=np.float64)
    q = np.zeros((num_traj, t_len, 1), dtype=np.float64)
    qdot = np.zeros((num_traj, t_len, 1), dtype=np.float64)
    energy = np.zeros((num_traj, t_len), dtype=np.float64)
    param_dim = sample_params_fn(rng).shape[0]
    params = np.zeros((num_traj, param_dim), dtype=np.float64)

    for i in range(num_traj):
        p = sample_params_fn(rng)
        params[i] = p
        l, m, c = float(p[1]), float(p[2]), float(p[3]) if p.shape[0] > 3 else 0.0
        y0 = sample_initial_conditions(rng, cfg)
        traj_states, theta, theta_dot = integrate_trajectory(y0, t_eval, g=g, l=l, c=c)
        states[i] = traj_states
        q[i, :, 0] = theta
        qdot[i, :, 0] = theta_dot
        energy[i] = pendulum_energy(theta, theta_dot, g=g, l=l, m=m)

    return {
        "states": states,
        "q": q,
        "qdot": qdot,
        "times": t_eval,
        "params": params,
        "energy": energy,
    }


def save_split(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"Saved {path} | trajectories={data['states'].shape[0]}, T={data['states'].shape[1]}")


def generate_splits(cfg: dict, sample_params_fn: Callable[[np.random.Generator], np.ndarray], seed: int) -> None:
    from src.utils.seed import set_seed

    set_seed(seed)
    rng = np.random.default_rng(seed)
    save_dir = Path(cfg["save_dir"])

    splits = [
        ("train", int(cfg["num_train"])),
        ("val", int(cfg["num_val"])),
        ("test", int(cfg["num_test"])),
    ]
    for name, n in splits:
        data = build_dataset_array(rng, n, cfg, sample_params_fn)
        save_split(save_dir / f"{name}.npz", data)


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
