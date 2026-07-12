# Generate all datasets and L=8, H=32 windows (run from project root)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

python src/datasets/generate_pendulum.py --config configs/pendulum.yaml
python src/datasets/generate_damped_pendulum.py --config configs/damped_pendulum.yaml
python src/datasets/generate_variable_pendulum.py --config configs/variable_pendulum.yaml

python src/datasets/build_windows.py --dataset pendulum --history_len 8 --horizon 32
python src/datasets/build_windows.py --dataset damped_pendulum --history_len 8 --horizon 32
python src/datasets/build_windows.py --dataset variable_pendulum --history_len 8 --horizon 32

Write-Host "Data pipeline finished."
