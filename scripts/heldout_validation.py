#!/usr/bin/env python3
"""
heldout_validation.py — apply the v2 Alignment gate to the held-out scenarios.

Runs scripts/reclassify_matrix_v2.py's reclassify_episode, unchanged and at its
defaults (TTC_CRIT 2.0, SPEED_REWARD_MIN 0.4, value decline at or before the
deviation step), over the saved baseline evidence for the nine scenarios other
than lane_change_simple. The junction scenarios use the simulator-TTC evidence
(*_simttc), not the original observation-TTC directories.

This is a validation output only. It does not replace the evaluation-pass
labels or any existing scenario table. Writes results/analysis/heldout_validation.csv.

Run from repo root:
    python scripts/heldout_validation.py
"""
from __future__ import annotations

import csv
import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import reclassify_matrix_v2 as v2  # noqa: E402

OUT_CSV = REPO / "results" / "analysis" / "heldout_validation.csv"
SEEDS = [0, 1, 2, 3, 4]
# scenario -> run-directory suffix holding its evidence
SCENARIOS = {
    "lane_change_blocked": "lane_change_blocked",
    "cut_in_aggressive": "cut_in_aggressive",
    "cut_in_high_speed": "cut_in_high_speed",
    "emergency_braking_dense": "emergency_braking_dense",
    "emergency_braking_lead_stop": "emergency_braking_lead_stop",
    "junction_crossing_unprotected": "junction_crossing_unprotected_simttc",
    "junction_crossing_busy": "junction_crossing_busy_simttc",
    "pedestrian_static": "pedestrian_static",
    "pedestrian_crossing": "pedestrian_crossing",
}


def analyse(run_dir: Path) -> dict:
    """Same loop as reclassify_matrix_v2.analyse_agent, for an arbitrary run dir."""
    counts, reasons = defaultdict(int), defaultdict(int)
    files = sorted(glob.glob(str(run_dir / "evidence" / "*.json")))
    for f in files:
        d = json.load(open(f))
        cat, reason = v2.reclassify_episode(d.get("evidence_trace", []))
        counts[cat] += 1
        reasons[reason] += 1
    return {"n": len(files), "counts": counts, "reasons": reasons}


def main() -> None:
    assert v2.STRICT is False and v2.TTC_CRIT == 2.0 and v2.SPEED_REWARD_MIN == 0.4
    rows = []
    for scenario, run_suffix in SCENARIOS.items():
        per_seed = []
        for seed in SEEDS:
            run_dir = v2.RUNS / f"baseline_seed{seed}_{run_suffix}"
            r = analyse(run_dir)
            row = {
                "scenario": scenario,
                "evidence_dir": run_dir.relative_to(REPO).as_posix(),
                "seed": seed,
                "n_evidence": r["n"],
                "alignment": r["counts"].get("Alignment", 0),
                "planning": r["counts"].get("Planning", 0),
                "lateral": r["reasons"].get("lateral_reward_seeking", 0),
                "longitudinal": r["reasons"].get("longitudinal_reward_seeking", 0),
            }
            rows.append(row)
            per_seed.append(row)
        rows.append({
            "scenario": scenario, "evidence_dir": "", "seed": "mean",
            **{k: f"{statistics.mean(r[k] for r in per_seed):.2f}"
               for k in ("n_evidence", "alignment", "planning", "lateral", "longitudinal")},
        })

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
