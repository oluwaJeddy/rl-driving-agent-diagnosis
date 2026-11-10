#!/usr/bin/env python3
"""
eval_baseline_all_scenarios.py — Extend the multi-seed matrix to the eight
probe scenarios it doesn't yet cover.

The existing multi-seed matrix (scripts/analyze_matrix.py +
scripts/reclassify_matrix.py) only evaluates lane_change_simple (the
reward-seeking Alignment finding) and pedestrian_static (control) across the
5 baseline seeds. This script runs the same 5 baseline-condition agents
across the remaining 8 scenarios in evaluate.SCENARIOS:

    lane_change_blocked, cut_in_aggressive, cut_in_high_speed,
    emergency_braking_dense, emergency_braking_lead_stop,
    junction_crossing_unprotected, junction_crossing_busy, pedestrian_crossing

No retraining, no classifier changes. Reuses src/evaluate.py unchanged:
  - evaluate.SCENARIOS supplies the exact scenario configs.
  - evaluate.probe_scenario() / run_episode() do the env-construction,
    VecNormalize-based obs normalization, episode loop, and evidence
    recording — the same machinery every other reported result is built
    from.
  - evaluate._classify_step() / _classify_near_miss() apply the CURRENT
    taxonomy criteria, which already include the reward-seeking-lateral-
    manoeuvre gate added in b318652 ("Add lateral reward-hacking Alignment
    gate to failure classifier"). That gate is what reclassify_matrix.py
    used to have to apply as a separate post-hoc pass over raw evidence
    traces; it now lives directly in evaluate.py, so a single evaluate.py
    pass already produces "reclassified" categories. No second pass needed.
  - Each episode's evidence_dir JSON records the four instrumented
    behavioural signals evaluate.py's docstring defines them by (value
    estimate, TTC to nearest hazard, hazard-visible step, action deviation
    from the safe brake baseline) for every step, plus the resulting
    FailureEvent category/subcategory.

Seeding: fixed EVAL_SEED (42) shared across all 5 baseline seeds, per the
existing multi-seed matrix convention in scripts/analyze_matrix.py ("fixed
eval seed so every agent faces identical scenarios"). Combined with
probe_scenario's existing per-episode offset (ep_seed = base + episode_index),
every seed's evaluation on a given scenario sees identical episode initial
conditions episode-for-episode, isolating differences to the trained
policies.

Usage (from repo root):
    python scripts/eval_baseline_all_scenarios.py --scenario cut_in_aggressive
    python scripts/eval_baseline_all_scenarios.py --scenario cut_in_aggressive --episodes 5   # smoke test
    python scripts/eval_baseline_all_scenarios.py --merge   # after all 8 scenarios have run

Slurm (one array task per scenario, then a merge task):
    sbatch slurm/eval_baseline_all_scenarios.slurm
    # then, once all 8 tasks have finished:
    python scripts/eval_baseline_all_scenarios.py --merge
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import List, Optional, Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import gymnasium as gym  # noqa: E402
import highway_env  # noqa: E402,F401 -- side-effect: registers highway-v0 / intersection-v0
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from evaluate import SCENARIOS, probe_scenario  # noqa: E402
from taxonomy import FailureCategory  # noqa: E402

BASELINE_SEEDS = [0, 1, 2, 3, 4]

# The multi-seed matrix (analyze_matrix.py) already covers these two.
_MATRIX_COVERED = {"lane_change_simple", "pedestrian_static"}
ALL_SCENARIOS: List[str] = [k for k in SCENARIOS if k not in _MATRIX_COVERED]

EPISODES_DEFAULT = 30
EVAL_SEED = 42  # matches scripts/analyze_matrix.py's fixed EVAL_SEED convention

CHECKPOINT_ROOT = REPO / "results" / "checkpoints" / "baseline"
RUNS_ROOT = REPO / "results" / "analysis" / "runs"
OUT_CSV = REPO / "results" / "analysis" / "baseline_all_scenarios.csv"

CATEGORY_ORDER = [
    FailureCategory.PERCEPTION,
    FailureCategory.PREDICTION,
    FailureCategory.PLANNING,
    FailureCategory.ALIGNMENT,
    FailureCategory.ROBUSTNESS,
    FailureCategory.INTERACTION,
]


def _dummy_env_for(scenario_cfg: dict, env_id: str) -> DummyVecEnv:
    """A single-env DummyVecEnv matching the scenario's obs/action space, used
    only to load VecNormalize's running statistics (shape-matching, not
    actually stepped)."""
    return DummyVecEnv([lambda: gym.make(env_id, config=scenario_cfg)])


def evaluate_seed_on_scenario(scenario_key: str, seed: int, episodes: int) -> List[dict]:
    scenario = SCENARIOS[scenario_key]
    ckpt_dir = CHECKPOINT_ROOT / f"seed_{seed}"
    model_path = ckpt_dir / "ppo_baseline_final.zip"
    vecnorm_path = ckpt_dir / "vec_normalize.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing final checkpoint: {model_path}")
    if not vecnorm_path.exists():
        raise FileNotFoundError(f"Missing vec_normalize.pkl: {vecnorm_path}")

    print(f"[{scenario_key} | seed {seed}] loading {model_path.name} + {vecnorm_path.name}")
    model = PPO.load(str(model_path))

    merged_cfg = {**scenario.base_config, **scenario.config}
    dummy = _dummy_env_for(merged_cfg, scenario.env_id)
    vec_norm = VecNormalize.load(str(vecnorm_path), dummy)
    vec_norm.training = False
    vec_norm.norm_reward = False

    evidence_dir = RUNS_ROOT / f"baseline_seed{seed}_{scenario_key}" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    reports, metrics = probe_scenario(
        scenario,
        model,
        episodes,
        vec_norm,
        deterministic=True,
        traj_dir=None,
        evidence_dir=evidence_dir,
        base_seed=EVAL_SEED,
    )
    dummy.close()

    rows = []
    for r, m in zip(reports, metrics):
        rows.append({
            "scenario": scenario_key,
            "seed": seed,
            "episode": r.episode_id,
            "crashed": bool(r.terminated_abnormally),
            "failure_category": m.failure_category.value if m.failure_category else "",
        })
    return rows


def per_scenario_csv_path(scenario_key: str) -> Path:
    return OUT_CSV.with_suffix(f".{scenario_key}.csv")


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scenario", "seed", "episode", "crashed", "failure_category"])
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[dict]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        return [
            {
                "scenario": row["scenario"],
                "seed": int(row["seed"]),
                "episode": int(row["episode"]),
                "crashed": row["crashed"] in ("True", "true", "1"),
                "failure_category": row["failure_category"],
            }
            for row in reader
        ]


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def summarize_scenario(scenario_key: str, rows: List[dict], seeds: Sequence[int]) -> dict:
    """One row: per-seed crash rate + per-category episode counts, aggregated
    to mean/std across seeds."""
    crash_rates: List[float] = []
    cat_counts: dict = {c.value: [] for c in CATEGORY_ORDER}

    for seed in seeds:
        seed_rows = [r for r in rows if r["seed"] == seed]
        if not seed_rows:
            continue
        n = len(seed_rows)
        crash_rates.append(sum(r["crashed"] for r in seed_rows) / n)
        for c in CATEGORY_ORDER:
            cat_counts[c.value].append(
                sum(1 for r in seed_rows if r["failure_category"] == c.value)
            )

    out = {"scenario": scenario_key, "n_seeds": len(crash_rates)}
    out["n_episodes_per_seed"] = len(rows) // len(seeds) if seeds and rows else 0
    m, s = _mean_std(crash_rates)
    out["crash_rate_mean"], out["crash_rate_std"] = round(m, 4), round(s, 4)
    for c in CATEGORY_ORDER:
        m, s = _mean_std([float(v) for v in cat_counts[c.value]])
        key = c.value.lower()
        out[f"{key}_mean"], out[f"{key}_std"] = round(m, 3), round(s, 3)
    return out


SUMMARY_FIELDS = (
    ["scenario", "n_seeds", "n_episodes_per_seed", "crash_rate_mean", "crash_rate_std"]
    + [f"{c.value.lower()}_{stat}" for c in CATEGORY_ORDER for stat in ("mean", "std")]
)


def write_summary_csv(path: Path, summary_rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)


def print_summary(summary_rows: List[dict]) -> None:
    header = (
        f"{'scenario':<28}{'n_seeds':>8}{'crash_rate':>14}"
        + "".join(f"{c.value:>13}" for c in CATEGORY_ORDER)
    )
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        cat_str = "".join(
            f"{row[f'{c.value.lower()}_mean']:>6.1f}±{row[f'{c.value.lower()}_std']:<5.1f}"
            for c in CATEGORY_ORDER
        )
        print(
            f"{row['scenario']:<28}{row['n_seeds']:>8}"
            f"{row['crash_rate_mean']:>7.3f}±{row['crash_rate_std']:<6.3f}"
            f"{cat_str}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--scenario", choices=ALL_SCENARIOS, default=None,
        help="Single scenario to evaluate (all 5 baseline seeds). Slurm array tasks pass one each.",
    )
    p.add_argument(
        "--scenario-index", type=int, default=None,
        help="Index into the fixed 8-scenario list (0-7) — convenience for $SLURM_ARRAY_TASK_ID.",
    )
    p.add_argument("--episodes", type=int, default=EPISODES_DEFAULT,
                    help=f"Episodes per seed per scenario (default: {EPISODES_DEFAULT})")
    p.add_argument("--merge", action="store_true",
                    help="Skip evaluation; merge existing per-scenario partial CSVs into the final summary CSV.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.merge:
        summary_rows = []
        missing = []
        for scenario_key in ALL_SCENARIOS:
            p = per_scenario_csv_path(scenario_key)
            if not p.exists():
                missing.append(scenario_key)
                continue
            rows = read_csv(p)
            summary_rows.append(summarize_scenario(scenario_key, rows, BASELINE_SEEDS))
        if missing:
            print(f"WARNING: missing partial CSVs for scenarios {missing}; summarizing what's present.")
        write_summary_csv(OUT_CSV, summary_rows)
        print(f"\nWrote per-scenario summary ({len(summary_rows)} scenarios) -> {OUT_CSV}")
        print_summary(summary_rows)
        return

    scenario_key: Optional[str] = args.scenario
    if scenario_key is None and args.scenario_index is not None:
        scenario_key = ALL_SCENARIOS[args.scenario_index]
    if scenario_key is None:
        print("Specify --scenario <key>, --scenario-index N, or --merge. Scenarios:")
        for i, k in enumerate(ALL_SCENARIOS):
            print(f"  [{i}] {k}")
        sys.exit(1)

    all_rows: List[dict] = []
    for seed in BASELINE_SEEDS:
        seed_rows = evaluate_seed_on_scenario(scenario_key, seed, args.episodes)
        all_rows.extend(seed_rows)
        crashes = sum(r["crashed"] for r in seed_rows)
        print(f"[{scenario_key} | seed {seed}] done: {len(seed_rows)} episodes, {crashes} collisions")

    p = per_scenario_csv_path(scenario_key)
    write_csv(p, all_rows)
    print(f"\nWrote partial -> {p}")
    print("Run with --merge once all 8 scenario partials exist to produce the final summary CSV.")

    print()
    summary_row = summarize_scenario(scenario_key, all_rows, BASELINE_SEEDS)
    print_summary([summary_row])


if __name__ == "__main__":
    main()
