#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python src/datasets/generate_pendulum.py --config configs/pendulum.yaml
python src/datasets/generate_damped_pendulum.py --config configs/damped_pendulum.yaml
python src/datasets/generate_variable_pendulum.py --config configs/variable_pendulum.yaml

python src/datasets/build_windows.py --dataset pendulum --history_len 8 --horizon 32
python src/datasets/build_windows.py --dataset damped_pendulum --history_len 8 --horizon 32
python src/datasets/build_windows.py --dataset variable_pendulum --history_len 8 --horizon 32

echo "Data pipeline finished."
