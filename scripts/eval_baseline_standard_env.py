#!/usr/bin/env python3
"""
eval_baseline_standard_env.py — Baseline agent performance on the standard,
unmodified highway-env configuration (no hazard-probe overrides).

Purpose: establish that the agent is competent under normal driving
conditions before reporting its failure rate on the constructed hazard
probe scenarios in results/evaluation/. This is a separate measurement
from the risk-matrix probes and does not touch that pipeline's outputs.

Reuses existing machinery rather than re-implementing it:
  - src/train_experiment.py's _ENV_BASE + CONDITIONS["baseline"] gives the
    *exact* env config the baseline checkpoints were trained on (no probe
    scenario overrides are applied).
  - src/evaluate.py's ScenarioDefinition/_make_env/probe_scenario/run_episode
    are used unchanged for env construction, VecNormalize-based observation
    normalization, and the episode loop, so this measurement is built the
    same way as every other reported result. traj_dir/evidence_dir are left
    as None, so none of evaluate.py's failure-classification file output is
    written here — this script only records reward/steps/collision.

Seeding: one fixed base seed (BASE_EVAL_SEED) shared across all five model
seeds, with probe_scenario's existing per-episode offset (ep_seed = base +
episode_index). All five seeds' evaluations therefore see identical episode
initial conditions episode-for-episode, isolating differences in the metrics
to the trained policies rather than to the traffic layout they're evaluated
on.

Usage (from repo root):
    python scripts/eval_baseline_standard_env.py
    python scripts/eval_baseline_standard_env.py --seeds 0 1 --episodes 20   # quick smoke test
    python scripts/eval_baseline_standard_env.py --seeds 2                  # single seed (Slurm array task)
    python scripts/eval_baseline_standard_env.py --merge                    # combine per-seed partial CSVs

Slurm (one array task per seed, then a merge task):
    sbatch slurm/eval_baseline_standard.slurm
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import List, Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import gymnasium as gym  # noqa: E402
import highway_env  # noqa: E402,F401 -- side-effect: registers highway-v0
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from evaluate import NuPlanCategory, ScenarioDefinition, probe_scenario  # noqa: E402
from train_experiment import CONDITIONS, _ENV_BASE  # noqa: E402

ALL_SEEDS = [0, 1, 2, 3, 4]
N_EPISODES_DEFAULT = 100
BASE_EVAL_SEED = 1000  # fixed; per-episode seed = BASE_EVAL_SEED + episode_index
CHECKPOINT_ROOT = REPO / "results" / "checkpoints" / "baseline"
OUT_CSV = REPO / "results" / "analysis" / "baseline_standard_env.csv"

# Exactly the config train_experiment.py trained the baseline condition on.
# {**_ENV_BASE, **CONDITIONS["baseline"]} is the same expression train.py
# uses in _make_env_fn(); no scenario-probe keys (lanes_count/vehicles_count/
# duration/etc. from evaluate.py's SCENARIOS) are applied.
STANDARD_ENV_CONFIG: dict = {**_ENV_BASE, **CONDITIONS["baseline"]}
DURATION = STANDARD_ENV_CONFIG["duration"]

# ScenarioDefinition is evaluate.py's unit of "what to run"; _make_env()
# merges scenario.base_config (fixed per env_id) with scenario.config, and
# since STANDARD_ENV_CONFIG already supplies every key HIGHWAY_BASE_CONFIG
# would, the merge reduces to exactly STANDARD_ENV_CONFIG.
STANDARD_SCENARIO = ScenarioDefinition(
    key="standard_highway",
    nuplan_category=NuPlanCategory.LANE_CHANGE,  # unused label; not a hazard probe
    env_id="highway-v0",
    config=STANDARD_ENV_CONFIG,
    description="Standard unmodified highway-env training configuration (no probe overrides).",
    expected_failures=(),
)


def evaluate_seed(seed: int, episodes: int) -> List[dict]:
    ckpt_dir = CHECKPOINT_ROOT / f"seed_{seed}"
    model_path = ckpt_dir / "ppo_baseline_final.zip"
    vecnorm_path = ckpt_dir / "vec_normalize.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing final checkpoint: {model_path}")
    if not vecnorm_path.exists():
        raise FileNotFoundError(f"Missing vec_normalize.pkl: {vecnorm_path}")

    print(f"[seed {seed}] loading {model_path.name} + {vecnorm_path.name}")
    model = PPO.load(str(model_path))

    dummy = DummyVecEnv([lambda: gym.make("highway-v0", config=STANDARD_ENV_CONFIG)])
    vec_norm = VecNormalize.load(str(vecnorm_path), dummy)
    vec_norm.training = False
    vec_norm.norm_reward = False

    reports, metrics = probe_scenario(
        STANDARD_SCENARIO,
        model,
        episodes,
        vec_norm,
        deterministic=True,
        traj_dir=None,
        evidence_dir=None,
        base_seed=BASE_EVAL_SEED,
    )
    dummy.close()

    rows = []
    for r, m in zip(reports, metrics):
        collided = bool(r.terminated_abnormally)
        completed = (not collided) and (r.total_steps >= DURATION)
        rows.append({
            "seed": seed,
            "episode": r.episode_id,
            "total_reward": round(m.total_reward, 4),
            "steps": r.total_steps,
            "collision": collided,
            "completed_full_duration": completed,
        })
    return rows


def per_seed_csv_path(seed: int) -> Path:
    return OUT_CSV.with_suffix(f".seed{seed}.csv")


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["seed", "episode", "total_reward", "steps", "collision", "completed_full_duration"]
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[dict]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for row in reader:
            rows.append({
                "seed": int(row["seed"]),
                "episode": int(row["episode"]),
                "total_reward": float(row["total_reward"]),
                "steps": int(row["steps"]),
                "collision": row["collision"] in ("True", "true", "1"),
                "completed_full_duration": row["completed_full_duration"] in ("True", "true", "1"),
            })
        return rows


def summarize(rows: List[dict], seeds: Sequence[int]) -> List[dict]:
    """Per-seed rows plus one 'ALL' aggregate row, in the shape printed/reported."""
    summary = []
    for seed in seeds:
        seed_rows = [r for r in rows if r["seed"] == seed]
        if not seed_rows:
            continue
        rewards = [r["total_reward"] for r in seed_rows]
        n = len(seed_rows)
        summary.append({
            "seed": str(seed),
            "n_episodes": n,
            "mean_reward": statistics.mean(rewards),
            "std_reward": statistics.stdev(rewards) if n > 1 else 0.0,
            "collision_rate": sum(r["collision"] for r in seed_rows) / n,
            "completion_rate": sum(r["completed_full_duration"] for r in seed_rows) / n,
        })
    if rows:
        rewards = [r["total_reward"] for r in rows]
        n = len(rows)
        summary.append({
            "seed": "ALL",
            "n_episodes": n,
            "mean_reward": statistics.mean(rewards),
            "std_reward": statistics.stdev(rewards) if n > 1 else 0.0,
            "collision_rate": sum(r["collision"] for r in rows) / n,
            "completion_rate": sum(r["completed_full_duration"] for r in rows) / n,
        })
    return summary


def print_summary(summary: List[dict]) -> None:
    header = f"{'seed':>6} {'n_eps':>6} {'mean_reward':>12} {'std_reward':>11} {'collision_rate':>15} {'completion_rate':>16}"
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['seed']:>6} {row['n_episodes']:>6} "
            f"{row['mean_reward']:>12.3f} {row['std_reward']:>11.3f} "
            f"{row['collision_rate']:>15.3f} {row['completion_rate']:>16.3f}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Baseline seeds to evaluate (default: 0 1 2 3 4). "
                         "A single seed writes a per-seed partial CSV instead of the final CSV.")
    p.add_argument("--episodes", type=int, default=N_EPISODES_DEFAULT,
                    help=f"Episodes per seed (default: {N_EPISODES_DEFAULT})")
    p.add_argument("--merge", action="store_true",
                    help="Skip evaluation; merge existing per-seed partial CSVs into the final CSV.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.merge:
        rows: List[dict] = []
        missing = []
        for seed in ALL_SEEDS:
            p = per_seed_csv_path(seed)
            if p.exists():
                rows.extend(read_csv(p))
            else:
                missing.append(seed)
        if missing:
            print(f"WARNING: missing partial CSVs for seeds {missing}; merging what's present.")
        write_csv(OUT_CSV, rows)
        print(f"\nMerged {len(rows)} episode rows -> {OUT_CSV}")
        summary = summarize(rows, sorted({r['seed'] for r in rows}))
        print_summary(summary)
        return

    seeds = args.seeds if args.seeds is not None else ALL_SEEDS
    all_rows: List[dict] = []
    for seed in seeds:
        seed_rows = evaluate_seed(seed, args.episodes)
        all_rows.extend(seed_rows)
        crashes = sum(r["collision"] for r in seed_rows)
        print(f"[seed {seed}] done: {len(seed_rows)} episodes, {crashes} collisions")

    if seeds == ALL_SEEDS:
        write_csv(OUT_CSV, all_rows)
        print(f"\nWrote {len(all_rows)} episode rows -> {OUT_CSV}")
    else:
        for seed in seeds:
            seed_rows = [r for r in all_rows if r["seed"] == seed]
            p = per_seed_csv_path(seed)
            write_csv(p, seed_rows)
            print(f"Wrote partial -> {p}")
        print("Run with --merge once all seed partials exist to produce the final CSV.")

    print()
    summary = summarize(all_rows, seeds)
    print_summary(summary)


if __name__ == "__main__":
    main()
