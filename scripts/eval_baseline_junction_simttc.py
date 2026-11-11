#!/usr/bin/env python3
"""
eval_baseline_junction_simttc.py — re-run baseline agents on the two junction
scenarios with TTC computed from simulator state (evaluate.TTC_SOURCE).

Mirrors scripts/eval_baseline_all_scenarios.py::evaluate_seed_on_scenario
exactly (same checkpoints, VecNormalize, 30 episodes, EVAL_SEED 42,
deterministic) except that it sets evaluate.TTC_SOURCE and writes to
results/analysis/runs/baseline_seed{N}_{scenario}{suffix}/, never to the
committed baseline_seed{N}_{scenario}/ directories.

Why: intersection-v0 observations are absolute, and highway-env's fallback
y range there is ±4 m, so the observation-based TTC clips lateral offsets.

Run from repo root:
    python scripts/eval_baseline_junction_simttc.py                       # simulator TTC -> *_simttc
    python scripts/eval_baseline_junction_simttc.py --ttc-source observation --suffix _obsttc_check
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import evaluate  # noqa: E402
import eval_baseline_all_scenarios as base  # noqa: E402

SCENARIO_KEYS = ["junction_crossing_unprotected", "junction_crossing_busy"]
SEEDS = [0, 1, 2, 3, 4]


def run(scenario_key: str, seed: int, episodes: int, suffix: str) -> None:
    out_dir = base.RUNS_ROOT / f"baseline_seed{seed}_{scenario_key}{suffix}"
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; refusing to overwrite")
    # evaluate_seed_on_scenario writes evidence to RUNS_ROOT/baseline_seed{N}_{key}/evidence;
    # point RUNS_ROOT at a staging root so the committed directories are never touched.
    staging = base.RUNS_ROOT / f"_staging{suffix}"
    orig_root = base.RUNS_ROOT
    base.RUNS_ROOT = staging
    try:
        rows = base.evaluate_seed_on_scenario(scenario_key, seed, episodes)
    finally:
        base.RUNS_ROOT = orig_root
    (staging / f"baseline_seed{seed}_{scenario_key}").rename(out_dir)
    with open(out_dir / "failure_report.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["scenario", "seed", "episode", "crashed", "failure_category"])
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ttc-source", choices=["observation", "simulator"], default="simulator")
    p.add_argument("--suffix", default="_simttc")
    p.add_argument("--episodes", type=int, default=base.EPISODES_DEFAULT)
    args = p.parse_args()
    evaluate.TTC_SOURCE = args.ttc_source
    for key in SCENARIO_KEYS:
        for seed in SEEDS:
            run(key, seed, args.episodes, args.suffix)
    staging = base.RUNS_ROOT / f"_staging{args.suffix}"
    if staging.exists():
        staging.rmdir()


if __name__ == "__main__":
    main()
