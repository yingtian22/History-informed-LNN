# HiLNN

**History-informed Lagrangian Neural Networks for long-horizon physical forecasting.**

This repository contains the **minimal reproducible pipeline** for the paper main results (Table 1): three pendulum datasets, baseline methods, and HiLNN training/evaluation at `L=8`, `H=32`.

It intentionally excludes ablation studies, paper figure/table export scripts, and visualization utilities from the full research codebase.

## What this repo reproduces

| Dataset | Methods in main table | HiLNN checkpoint tag |
|---------|----------------------|----------------------|
| Pendulum | LNN, LNN-multistep, HNN, Neural ODE, MLP-one-step, HiLNN | `energy001` |
| Damped pendulum | LNN, LNN-multistep, HiLNN | `damped_energy001` |
| Variable pendulum | LNN, LNN-multistep, HiLNN | `variable_energy001` |

Reference numbers: `results/paper_main_results_reference.csv`

## Environment

```bash
conda create -n hilnn python=3.10 -y
conda activate hilnn
pip install -r requirements.txt
```

Install PyTorch for your platform from [pytorch.org](https://pytorch.org) if needed.

## Quick start (full pipeline)

**Windows (PowerShell):**

```powershell
.\scripts\run_paper_main.ps1
```

**Linux / macOS:**

```bash
bash scripts/run_paper_main.sh
```

Or run step by step:

### 1. Generate data

```bash
python src/datasets/generate_pendulum.py --config configs/pendulum.yaml
python src/datasets/generate_damped_pendulum.py --config configs/damped_pendulum.yaml
python src/datasets/generate_variable_pendulum.py --config configs/variable_pendulum.yaml

python src/datasets/build_windows.py --dataset pendulum --history_len 8 --horizon 32
python src/datasets/build_windows.py --dataset damped_pendulum --history_len 8 --horizon 32
python src/datasets/build_windows.py --dataset variable_pendulum --history_len 8 --horizon 32
```

### 2. Train baselines

**Pendulum (all main-table baselines):**

```bash
python src/training/train_baseline.py --model lnn --dataset pendulum --history_len 8 --horizon 32 --epochs 50 --config configs/baseline_lnn.yaml
python src/training/train_baseline.py --model lnn_multistep --dataset pendulum --history_len 8 --horizon 32 --epochs 50 --config configs/baseline_lnn_multistep.yaml
python src/training/train_baseline.py --model mlp_one_step --dataset pendulum --history_len 8 --horizon 32 --epochs 50
python src/training/train_baseline.py --model neural_ode --dataset pendulum --history_len 8 --horizon 32 --epochs 50 --batch_size 128 --config configs/baseline_neural_ode.yaml
python src/training/train_baseline.py --model hnn --dataset pendulum --history_len 8 --horizon 32 --epochs 50 --config configs/baseline_hnn.yaml
```

**Damped & variable (LNN family only):**

```bash
python src/training/train_baseline.py --model lnn --dataset damped_pendulum --history_len 8 --horizon 32 --epochs 50
python src/training/train_baseline.py --model lnn_multistep --dataset damped_pendulum --history_len 8 --horizon 32 --epochs 50
python src/training/train_baseline.py --model lnn --dataset variable_pendulum --history_len 8 --horizon 32 --epochs 50
python src/training/train_baseline.py --model lnn_multistep --dataset variable_pendulum --history_len 8 --horizon 32 --epochs 50
```

### 3. Train HiLNN (paper main configuration)

```bash
python src/training/train_hilnn.py --config configs/hilnn_pendulum.yaml \
  --energy_weight 0.01 --train_integrator rk4 --eval_integrator rk4 --no_train_detach \
  --tag energy001 --device cuda

python src/training/train_hilnn.py --config configs/hilnn_damped_pendulum.yaml \
  --dataset damped_pendulum --energy_weight 0.01 --train_integrator rk4 --eval_integrator rk4 --no_train_detach \
  --tag damped_energy001 --device cuda

python src/training/train_hilnn.py --config configs/hilnn_variable_pendulum.yaml \
  --dataset variable_pendulum --energy_weight 0.01 --train_integrator rk4 --eval_integrator rk4 --no_train_detach \
  --tag variable_energy001 --device cuda
```

### 4. Evaluate

```bash
python src/training/evaluate_baselines.py --dataset pendulum --history_len 8 --horizon 32 \
  --models lnn lnn_multistep hnn neural_ode mlp_one_step hilnn --hilnn_tag energy001 --device cpu

python src/training/evaluate_baselines.py --dataset damped_pendulum --history_len 8 --horizon 32 \
  --models lnn lnn_multistep hilnn --hilnn_tag damped_energy001 --device cpu \
  --output_table damped_pendulum_metrics_L8_H32.csv

python src/training/evaluate_baselines.py --dataset variable_pendulum --history_len 8 --horizon 32 \
  --models lnn lnn_multistep hilnn --hilnn_tag variable_energy001 --device cpu \
  --output_table variable_pendulum_metrics_L8_H32.csv
```

Metrics are written to `outputs/tables/`.

## Smoke tests (optional)

After data generation:

```bash
python scripts/smoke_test_baselines.py
python scripts/smoke_test_hilnn.py
```

## Project layout

```text
HiLNN-release/
├── configs/              # Dataset + model YAML configs
├── data/processed/       # Generated trajectories & windows (not in git)
├── results/              # Paper reference metrics for verification
├── scripts/              # One-click reproduction scripts
├── src/
│   ├── datasets/         # Data generation & loading
│   ├── models/           # HiLNN + baselines
│   ├── training/         # train / evaluate entry points
│   └── utils/            # Metrics, logging, seeding
└── outputs/              # Checkpoints, logs, metrics (generated)
```

## Citation

If you use this code, please cite our paper (bibtex to be added).

## Code Attribution

Parts of the Lagrangian Neural Network baseline implementation in this repository are based on and adapted from [MilesCranmer/lagrangian_nns](https://github.com/MilesCranmer/lagrangian_nns), which is released under the Apache License 2.0. The original implementation has been modified and extended for position-only history encoding, context-conditioned Lagrangian dynamics, dissipative and variable-parameter systems, and differentiable multi-step RK4 forecasting. We thank the original authors for making their code publicly available.

## License

The original HiLNN contributions are released under the license provided in this repository. Portions adapted from `MilesCranmer/lagrangian_nns` remain subject to the Apache License 2.0.
