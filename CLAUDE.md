# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PPO-based reinforcement learning driving agent trained in `highway-env`, with a failure-diagnosis pipeline that probes safety-critical scenarios and classifies failures into a six-category taxonomy (Perception, Prediction, Planning, Alignment, Robustness, Interaction) with ISO 21448 SOTIF-inspired risk scoring.

## Environment Setup

```
pip install -r requirements.txt
```

Python 3.10+ recommended. All source modules live in `src/` and import each other by name (no package install needed — run from the repo root or add `src/` to `PYTHONPATH`).

## Key Commands

**Train the PPO agent:**
```
python src/train.py
python src/train.py --env highway-v0 --timesteps 500000 --output results/checkpoints/ppo_highway
```

**Probe safety-critical scenarios and generate failure/risk reports:**
```
python src/evaluate.py --model results/checkpoints/ppo_highway/ppo_final
python src/evaluate.py --model <path> --episodes 50 --scenarios high_density,cut_in
```

Both scripts accept `--help` for full flag lists.

## Architecture

```
src/
  train.py      – PPO training loop (SB3 + VecNormalize + Eval/Checkpoint callbacks)
  evaluate.py   – Multi-scenario failure probing; outputs CSVs to results/evaluation/
  taxonomy.py   – Six-category failure classifier (rule-based + info-dict heuristics)
  risk_matrix.py– Severity × Likelihood scoring; derives risk bands (Acceptable/ALARP/Unacceptable)
results/
  checkpoints/  – Saved model zips + VecNormalize pkl (gitignored)
  evaluation/   – failure_report.csv, risk_matrix.csv
data/raw/       – Raw scenario data (gitignored)
```

### Data flow

1. `train.py` → `results/checkpoints/ppo_final.zip` + `vec_normalize.pkl`
2. `evaluate.py` loads the model, runs it across `SCENARIOS` dict, calls `taxonomy.classify_from_info()` per step, builds `EpisodeFailureReport` objects
3. `RiskMatrix.add_from_counts()` converts per-scenario failure rates → `Likelihood` levels, combined with default `Severity` per category → risk scores
4. Outputs: `failure_report.csv` (per-episode) and `risk_matrix.csv` (per scenario × category)

### Six failure categories (`taxonomy.FailureCategory`)

| Category    | Typical trigger in highway-env                         |
|-------------|--------------------------------------------------------|
| Perception  | Crash with no vehicles visible in prior observation    |
| Prediction  | Cut-in / sudden brake not anticipated                  |
| Planning    | Traffic violation, off-road, collision at normal speed |
| Alignment   | Crash while actively maximising speed reward           |
| Robustness  | High-density / OOD traffic configuration               |
| Interaction | Deadlock, yield failures in multi-agent scenarios      |

### Probed scenarios (`evaluate.SCENARIOS`)

`baseline`, `high_density`, `highway_merging`, `cut_in`, `low_speed`, `sparse` — each overrides `BASE_CONFIG` keys passed to `gym.make`.

## Outputs

- `results/checkpoints/` – model checkpoints every 50k steps, best model, TensorBoard logs
- `results/evaluation/failure_report.csv` – per-episode: scenario, steps, crashed, primary category, per-category counts
- `results/evaluation/risk_matrix.csv` – per scenario × category: S, L, risk score, band
