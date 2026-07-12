# Release scope

This folder (`HiLNN-release`) is a **minimal open-source snapshot** for reproducing the paper **main results (Table 1)**.

## Included

- Data generation for three pendulum datasets
- Window construction (`L=8`, `H=32`)
- Baseline training: LNN, LNN-multistep, HNN, Neural ODE, MLP-one-step
- HiLNN training with paper main config (`energy_weight=0.01`, RK4, no detach)
- Evaluation and metrics export to `outputs/tables/`
- Reference metrics in `results/paper_main_results_reference.csv`

## Excluded (kept in full research repo `HiLNN/`)

- Ablation experiments (energy weight grid, history length, init-velocity, training design)
- Stage 4 orchestration (`run_stage4.ps1`)
- Analysis / visualization (`src/analysis/`, rollout/energy plots, context probes)
- Paper export scripts (LaTeX tables, figure rendering, schematic figures)
- `external/` reference clones
- Internal experiment notes (`PAPER_*.md`, `EXPERIMENT_PROGRESS.md`, `HiLNN_Stage*.md`)
- `mlp_direct` baseline (excluded from paper main table)

## One-command reproduction

```powershell
.\scripts\run_paper_main.ps1
```

```bash
bash scripts/run_paper_main.sh
```
