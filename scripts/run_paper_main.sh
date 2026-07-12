#!/usr/bin/env bash
# Reproduce paper main results (Table 1)
# Usage: bash scripts/run_paper_main.sh [cuda|cpu]
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${1:-cuda}"
COMMON=(--history_len 8 --horizon 32 --epochs 50 --batch_size 256 --device "$DEVICE")

echo "=== Step 1/4: Data generation ==="
bash scripts/run_data.sh

echo "=== Step 2/4: Baseline training ==="
python src/training/train_baseline.py --model lnn --dataset pendulum --config configs/baseline_lnn.yaml "${COMMON[@]}"
python src/training/train_baseline.py --model lnn_multistep --dataset pendulum --config configs/baseline_lnn_multistep.yaml "${COMMON[@]}"
python src/training/train_baseline.py --model mlp_one_step --dataset pendulum "${COMMON[@]}"
python src/training/train_baseline.py --model neural_ode --dataset pendulum --config configs/baseline_neural_ode.yaml --history_len 8 --horizon 32 --epochs 50 --batch_size 128 --device "$DEVICE"
python src/training/train_baseline.py --model hnn --dataset pendulum --config configs/baseline_hnn.yaml "${COMMON[@]}"

python src/training/train_baseline.py --model lnn --dataset damped_pendulum "${COMMON[@]}"
python src/training/train_baseline.py --model lnn_multistep --dataset damped_pendulum "${COMMON[@]}"
python src/training/train_baseline.py --model lnn --dataset variable_pendulum "${COMMON[@]}"
python src/training/train_baseline.py --model lnn_multistep --dataset variable_pendulum "${COMMON[@]}"

echo "=== Step 3/4: HiLNN training (paper main config) ==="
python src/training/train_hilnn.py --config configs/hilnn_pendulum.yaml \
  --energy_weight 0.01 --train_integrator rk4 --eval_integrator rk4 --no_train_detach \
  --tag energy001 --device "$DEVICE"

python src/training/train_hilnn.py --config configs/hilnn_damped_pendulum.yaml \
  --dataset damped_pendulum --energy_weight 0.01 --train_integrator rk4 --eval_integrator rk4 --no_train_detach \
  --tag damped_energy001 --device "$DEVICE"

python src/training/train_hilnn.py --config configs/hilnn_variable_pendulum.yaml \
  --dataset variable_pendulum --energy_weight 0.01 --train_integrator rk4 --eval_integrator rk4 --no_train_detach \
  --tag variable_energy001 --device "$DEVICE"

echo "=== Step 4/4: Evaluation ==="
python src/training/evaluate_baselines.py --dataset pendulum --history_len 8 --horizon 32 \
  --models lnn lnn_multistep hnn neural_ode mlp_one_step hilnn --hilnn_tag energy001 --device cpu

python src/training/evaluate_baselines.py --dataset damped_pendulum --history_len 8 --horizon 32 \
  --models lnn lnn_multistep hilnn --hilnn_tag damped_energy001 --device cpu \
  --output_table damped_pendulum_metrics_L8_H32.csv

python src/training/evaluate_baselines.py --dataset variable_pendulum --history_len 8 --horizon 32 \
  --models lnn lnn_multistep hilnn --hilnn_tag variable_energy001 --device cpu \
  --output_table variable_pendulum_metrics_L8_H32.csv

echo ""
echo "Done. Metrics: outputs/tables/"
echo "Compare with: results/paper_main_results_reference.csv"
