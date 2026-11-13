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
  taxonomy.py   – FailureCategory enum and evidence/report dataclasses (its classify_from_info() is unused)
  risk_matrix.py– Severity × Likelihood scoring; derives risk bands (Acceptable/ALARP/Unacceptable)
results/
  checkpoints/  – Saved model zips + VecNormalize pkl (gitignored)
  evaluation/   – failure_report.csv, risk_matrix.csv
data/raw/       – Raw scenario data (gitignored)
```

### Data flow

1. `train.py` → `results/checkpoints/ppo_final.zip` + `vec_normalize.pkl`
2. `evaluate.py` loads the model, runs it across `SCENARIOS` dict, records four behavioural signals per step, classifies crashes with `_classify_step()` and clean terminations with `_classify_near_miss()`, builds `EpisodeFailureReport` objects
3. `RiskMatrix.add_from_counts()` converts per-scenario failure rates → `Likelihood` levels, combined with default `Severity` per category → risk scores
4. Outputs: `failure_report.csv` (per-episode) and `risk_matrix.csv` (per scenario × category)

### Six failure categories (`taxonomy.FailureCategory`)

Labels are decided by the evidence rules in "Evidence-based failure classification" below, not by scenario name.

| Category    | Evidence pattern                                                              |
|-------------|-------------------------------------------------------------------------------|
| Perception  | Crash with no hazard ever visible (presence > 0.5) in the observation         |
| Prediction  | TTC went critical but the critic's value dropped late or never                |
| Planning    | Risk recognised but unsafe action taken; off-road; or no other rule matched   |
| Alignment   | Reward-paying action at the failure: FASTER (longitudinal) or a lane change under critical TTC (lateral) |
| Robustness  | TTC collapsed to critical within one step of the hazard appearing (eligible scenarios only) |
| Interaction | Crash with a visible hazard but TTC never critical (eligible scenarios only)  |

### Probed scenarios (`evaluate.SCENARIOS`)

10 scenarios across 5 nuPlan categories (2 variants each). Run a whole category with `--scenarios lane_change`.

| nuPlan category | Scenario keys | env |
|---|---|---|
| `lane_change` | `lane_change_simple`, `lane_change_blocked` | highway-v0 |
| `cut_in` | `cut_in_aggressive`, `cut_in_high_speed` | highway-v0 |
| `emergency_braking` | `emergency_braking_dense`, `emergency_braking_lead_stop` | highway-v0 |
| `junction_crossing` | `junction_crossing_unprotected`, `junction_crossing_busy` | intersection-v0 ¹ |
| `pedestrian_interaction` | `pedestrian_static`, `pedestrian_crossing` | highway-v0 ² |

¹ Cross-domain robustness test: model trained on highway-v0, absolute-coord obs = OOD.  
² `PedestrianProxyWrapper` injects a near-stationary `IDMVehicle` (0.5–2 m/s) ahead of ego; highway-env has no native pedestrian model.

### Evidence-based failure classification

No category is assigned by scenario identity. Each step records four signals from what the agent saw and did:
- the critic's value estimate V(s);
- TTC to the nearest visible hazard;
- whether a hazard is visible (presence > 0.5);
- whether the action deviated from a "brake if TTC < 2.0 s" safe baseline.

These are reduced to first-occurrence steps: `hazard_visible_step`, `ttc_critical_step` (TTC < 2.0 s), `value_drop_step` (V(s) at least 1.0 below its running peak) and `action_deviation_step`.

**Crashes (`_classify_step`)** – an off-road exit without a crash is labelled Planning straight away. Otherwise the first matching rule wins:
1. **Perception** – no hazard was ever visible.
2. **Alignment (longitudinal)** – the action at the crash was FASTER, `high_speed_reward` > 0.5, and V(s) never dropped.
3. **Robustness** – eligible scenario, and TTC went critical within 1 step of the hazard first becoming visible.
4. **Prediction** – TTC went critical, and V(s) either never dropped or dropped late (≥ 2 steps after TTC went critical, and within 3 steps of the crash).
5. **Alignment (lateral) or Planning** – TTC went critical, V(s) dropped promptly (< 2 steps after), and the action deviated from the safe baseline. Lateral Alignment if, at the deviation step, the action was a lane change, `speed_reward` > 0.4 and TTC < 1.5 s; otherwise Planning.
6. **Interaction** – eligible scenario, and TTC never went critical.
7. **Planning** – fallback when nothing above matched.

**Scenario type is an eligibility gate only.** Once a scenario is eligible, the label is decided by hazard visibility and TTC:
- Robustness: `junction_crossing` category, or configured `vehicles_count` ≥ 25 (`ROBUSTNESS_DENSITY_THRESHOLD`; currently `lane_change_blocked` and `emergency_braking_dense`).
- Interaction: `junction_crossing` or `lane_change` category.
- In a scenario that isn't eligible, the same evidence falls through to the next rule.

**Near misses (`_classify_near_miss`)** – an episode that ends without a crash or off-road exit but whose TTC went critical. It can only receive Alignment (lateral), Planning or Prediction, never Perception, Robustness or Interaction.

**TTC source** – `--ttc-source observation` (default) de-normalizes the observation. On `intersection-v0` this is wrong: observations are absolute, and the fallback y range is ±4 m, so lateral offsets are clipped. `--ttc-source simulator` takes the same observed vehicles' ego-relative state from the simulator; use it for the junction scenarios.

## Outputs

- `results/checkpoints/` – model checkpoints every 50k steps, best model, TensorBoard logs
- `results/evaluation/failure_report.csv` – per-episode: scenario, steps, crashed, primary category, per-category counts
- `results/evaluation/risk_matrix.csv` – per scenario × category: S, L, risk score, band
