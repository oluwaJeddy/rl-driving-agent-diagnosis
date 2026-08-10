#!/usr/bin/env python3
"""
analyze_matrix.py — Aggregate the 16-run reward-ablation experiment matrix.

Evaluates every trained agent (condition x seed) on two scenarios:
  - lane_change_simple  : where the reward-seeking Alignment failure lives
  - pedestrian_static   : CONTROL — should NOT change across conditions

Then aggregates crash rate, Alignment count, Planning count across seeds per
condition, and prints the table that answers: which reward ingredient
eliminated the reward-seeking lateral manoeuvre?

Run from repo root, inside the `carla` conda env:
    python scripts/analyze_matrix.py
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
import statistics

REPO = Path(__file__).resolve().parent.parent
CKPT = REPO / "results" / "checkpoints"
OUT = REPO / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

CONDITIONS = {
    "baseline": [0, 1, 2, 3, 4],
    "full_ablation": [0, 1, 2, 3, 4],
    "on_road_only": [0, 1, 2],
    "collision10_only": [0, 1, 2],
}
SCENARIOS = ["lane_change_simple", "pedestrian_static"]
EPISODES = 30
EVAL_SEED = 42  # fixed eval seed so every agent faces identical scenarios


def model_path(cond: str, seed: int) -> Path:
    # evaluate.py wants the path WITHOUT the .zip extension
    return CKPT / cond / f"seed_{seed}" / f"ppo_{cond}_final"


def run_one(cond: str, seed: int, scenario: str) -> dict | None:
    """Run evaluate.py for one agent on one scenario, return parsed summary."""
    mp = model_path(cond, seed)
    if not mp.with_suffix(".zip").exists():
        print(f"  !! missing model: {mp}.zip", file=sys.stderr)
        return None

    out_dir = OUT / "runs" / f"{cond}_seed{seed}_{scenario}"
    cmd = [
        sys.executable, str(REPO / "src" / "evaluate.py"),
        "--model", str(mp),
        "--scenarios", scenario,
        "--episodes", str(EPISODES),
        "--seed", str(EVAL_SEED),
        "--output", str(out_dir),
    ]
    print(f"  running {cond}/seed_{seed} on {scenario} ...", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  !! FAILED {cond}/seed_{seed}/{scenario}", file=sys.stderr)
        print(res.stderr[-1500:], file=sys.stderr)
        return None

    # Parse the failure_report.csv the evaluator writes
    return parse_output(out_dir, scenario)


def parse_output(out_dir: Path, scenario: str) -> dict:
    """Extract crash rate + per-category counts from evaluator output.

    Tries failure_report.csv first; falls back to counting evidence JSONs.
    """
    result = {"crashes": 0, "episodes": EPISODES, "categories": defaultdict(int)}

    # ---- Primary: failure_report.csv ----
    report = out_dir / "failure_report.csv"
    if report.exists():
        import csv
        with open(report) as f:
            rows = list(csv.DictReader(f))
        # crashes: rows where this scenario crashed. Columns vary; be defensive.
        for r in rows:
            scen = r.get("scenario", "") or r.get("scenario_key", "")
            if scenario not in scen:
                continue
            crashed = str(r.get("crashed", "")).lower() in ("true", "1", "yes")
            if crashed:
                result["crashes"] += 1
            cat = r.get("failure_category") or r.get("category")
            if cat:
                result["categories"][cat] += 1
        return result

    # ---- Fallback: evidence JSON files ----
    ev_dir = out_dir / "evidence"
    if ev_dir.exists():
        for jf in ev_dir.glob(f"{scenario}*.json"):
            try:
                d = json.loads(jf.read_text())
            except Exception:
                continue
            cat = d.get("category")
            if cat:
                result["categories"][cat] += 1
                result["crashes"] += 1
    return result


def agg(values: list[float]) -> str:
    if not values:
        return "  n/a "
    if len(values) == 1:
        return f"{values[0]:.1f}      "
    return f"{statistics.mean(values):.1f} ± {statistics.pstdev(values):.1f}"


def main():
    # condition -> scenario -> list of per-seed dicts
    data = defaultdict(lambda: defaultdict(list))

    for cond, seeds in CONDITIONS.items():
        print(f"\n=== {cond} ({len(seeds)} seeds) ===")
        for seed in seeds:
            for scen in SCENARIOS:
                r = run_one(cond, seed, scen)
                if r is not None:
                    data[cond][scen].append(r)

    # ---------- Build summary ----------
    rows = []
    for cond in CONDITIONS:
        row = {"condition": cond, "n_seeds": len(CONDITIONS[cond])}
        for scen in SCENARIOS:
            runs = data[cond][scen]
            crash_rates = [100.0 * r["crashes"] / r["episodes"] for r in runs]
            align = [r["categories"].get("Alignment", 0) for r in runs]
            plan = [r["categories"].get("Planning", 0) for r in runs]
            percep = [r["categories"].get("Perception", 0) for r in runs]
            row[f"{scen}__crash%"] = agg(crash_rates)
            row[f"{scen}__Alignment"] = agg([float(a) for a in align])
            row[f"{scen}__Planning"] = agg([float(p) for p in plan])
            row[f"{scen}__Perception"] = agg([float(p) for p in percep])
        rows.append(row)

    # ---------- Print headline table ----------
    print("\n" + "=" * 78)
    print("REWARD-ABLATION MATRIX — does the ingredient kill the reward hacking?")
    print("=" * 78)
    print(f"\n{'Condition':<18}{'seeds':<7}{'LC crash%':<14}{'LC Alignment':<16}{'LC Planning':<14}")
    print("-" * 69)
    for r in rows:
        print(f"{r['condition']:<18}{r['n_seeds']:<7}"
              f"{r['lane_change_simple__crash%']:<14}"
              f"{r['lane_change_simple__Alignment']:<16}"
              f"{r['lane_change_simple__Planning']:<14}")

    print(f"\n{'Condition':<18}{'seeds':<7}{'PED crash% (control)':<22}{'PED Perception':<16}")
    print("-" * 63)
    for r in rows:
        print(f"{r['condition']:<18}{r['n_seeds']:<7}"
              f"{r['pedestrian_static__crash%']:<22}"
              f"{r['pedestrian_static__Perception']:<16}")

    # ---------- Save CSV ----------
    import csv
    csv_path = OUT / "matrix_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved -> {csv_path}")

    print("\nREAD IT LIKE THIS:")
    print("  * If on_road_only kills Alignment but collision10_only does NOT,")
    print("    the survival bonus was the true causal lever (consistent with")
    print("    normalize_reward neutering the collision penalty).")
    print("  * pedestrian crash% should stay ~100 in EVERY condition (control).")


if __name__ == "__main__":
    main()
