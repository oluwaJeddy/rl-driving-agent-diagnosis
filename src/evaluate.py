"""
Failure-probing evaluation across safety-critical scenarios.

Usage:
    python src/evaluate.py --model results/checkpoints/ppo_highway/ppo_final
    python src/evaluate.py --model <path> --episodes 50 --scenarios all
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from taxonomy import (
    EpisodeFailureReport,
    FailureCategory,
    classify_from_info,
)
from risk_matrix import Likelihood, RiskMatrix


# ---------------------------------------------------------------------------
# Scenario definitions – each overrides the base highway-env config
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, dict] = {
    "baseline": {
        "vehicles_count": 15,
        "lanes_count": 4,
        "duration": 40,
    },
    "high_density": {
        "vehicles_count": 40,
        "lanes_count": 4,
        "duration": 40,
    },
    "highway_merging": {
        "vehicles_count": 20,
        "lanes_count": 3,
        "duration": 40,
        "initial_lane_id": 0,
    },
    "cut_in": {
        # Aggressive cut-in by surrounding vehicles
        "vehicles_count": 10,
        "lanes_count": 4,
        "duration": 30,
        "other_vehicles_type": "highway_env.vehicle.behavior.AggressiveVehicle",
    },
    "low_speed": {
        # Slow traffic – tests patience / planning
        "vehicles_count": 20,
        "lanes_count": 4,
        "duration": 40,
        "reward_speed_range": [5, 15],
    },
    "sparse": {
        "vehicles_count": 3,
        "lanes_count": 4,
        "duration": 40,
    },
}


BASE_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "normalize": True,
        "absolute": False,
    },
    "action": {"type": "DiscreteMetaAction"},
    "collision_reward": -5.0,
    "high_speed_reward": 0.4,
    "lane_change_reward": 0.1,
    "reward_speed_range": [20, 30],
}


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

def _make_env(env_id: str, scenario_config: dict):
    cfg = {**BASE_CONFIG, **scenario_config}
    env = gym.make(env_id, config=cfg)
    return env


def run_episode(
    env: gym.Env,
    model: PPO,
    episode_id: int,
    scenario_name: str,
    vec_norm: Optional[VecNormalize] = None,
    deterministic: bool = True,
) -> EpisodeFailureReport:
    obs, info = env.reset()
    report = EpisodeFailureReport(
        episode_id=episode_id,
        scenario=scenario_name,
        total_steps=0,
        terminated_abnormally=False,
    )

    step = 0
    prev_obs_dict: Optional[dict] = None

    while True:
        # Normalise obs if VecNormalize was used during training
        obs_input = obs[np.newaxis] if vec_norm is None else (
            vec_norm.normalize_obs(obs[np.newaxis])
        )
        action, _ = model.predict(obs_input, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        step += 1

        event = classify_from_info(info, obs_snapshot=prev_obs_dict, step=step)
        if event is not None:
            report.events.append(event)

        prev_obs_dict = {"vehicles_count": int(np.sum(obs[:, 0]))} if hasattr(obs, '__len__') else None

        if terminated or truncated:
            report.terminated_abnormally = terminated and info.get("crashed", False)
            break

    report.total_steps = step
    return report


def probe_scenario(
    scenario_name: str,
    scenario_config: dict,
    model: PPO,
    n_episodes: int,
    env_id: str,
    vec_norm: Optional[VecNormalize],
) -> List[EpisodeFailureReport]:
    reports: List[EpisodeFailureReport] = []
    for ep in range(n_episodes):
        env = _make_env(env_id, scenario_config)
        report = run_episode(env, model, ep, scenario_name, vec_norm)
        env.close()
        reports.append(report)
        if (ep + 1) % 10 == 0:
            crashes = sum(r.terminated_abnormally for r in reports)
            print(f"  [{scenario_name}] {ep+1}/{n_episodes} eps | crashes: {crashes}")
    return reports


# ---------------------------------------------------------------------------
# Aggregate & export
# ---------------------------------------------------------------------------

def aggregate_reports(all_reports: Dict[str, List[EpisodeFailureReport]]) -> pd.DataFrame:
    rows = []
    for scenario, reports in all_reports.items():
        for r in reports:
            row = {
                "scenario": scenario,
                "episode": r.episode_id,
                "steps": r.total_steps,
                "crashed": r.terminated_abnormally,
                "primary_category": r.primary_category.value,
                "failure_events": len(r.events),
            }
            for cat in FailureCategory:
                row[f"n_{cat.value.lower()}"] = r.category_counts.get(cat.value, 0)
            rows.append(row)
    return pd.DataFrame(rows)


def build_risk_matrix(
    all_reports: Dict[str, List[EpisodeFailureReport]],
) -> RiskMatrix:
    matrix = RiskMatrix()
    for scenario, reports in all_reports.items():
        total = len(reports)
        # Aggregate category counts across all episodes in this scenario
        combined: Dict[str, int] = {c.value: 0 for c in FailureCategory}
        for r in reports:
            for cat, cnt in r.category_counts.items():
                combined[cat] += cnt
        matrix.add_from_counts(scenario, combined, total)
    return matrix


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe PPO agent across safety-critical scenarios")
    p.add_argument("--model", required=True, help="Path to saved PPO model (no .zip)")
    p.add_argument("--vecnorm", default=None, help="Path to VecNormalize pickle")
    p.add_argument("--env", default="highway-v0")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--scenarios", default="all",
                   help="Comma-separated scenario names, or 'all'")
    p.add_argument("--output", default="results/evaluation")
    p.add_argument("--deterministic", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = PPO.load(args.model)

    vec_norm: Optional[VecNormalize] = None
    if args.vecnorm and Path(args.vecnorm).exists():
        dummy_env = DummyVecEnv([lambda: gym.make(args.env, config=BASE_CONFIG)])
        vec_norm = VecNormalize.load(args.vecnorm, dummy_env)
        vec_norm.training = False
        vec_norm.norm_reward = False

    scenario_names = (
        list(SCENARIOS.keys())
        if args.scenarios == "all"
        else [s.strip() for s in args.scenarios.split(",")]
    )

    all_reports: Dict[str, List[EpisodeFailureReport]] = {}
    for name in scenario_names:
        if name not in SCENARIOS:
            print(f"Unknown scenario '{name}', skipping.")
            continue
        print(f"\nProbing scenario: {name}")
        all_reports[name] = probe_scenario(
            name, SCENARIOS[name], model, args.episodes, args.env, vec_norm
        )

    df = aggregate_reports(all_reports)
    csv_path = output_dir / "failure_report.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFailure report → {csv_path}")

    matrix = build_risk_matrix(all_reports)
    risk_df = matrix.to_dataframe()
    risk_csv = output_dir / "risk_matrix.csv"
    risk_df.to_csv(risk_csv, index=False)
    print(f"Risk matrix    → {risk_csv}")

    unacceptable = matrix.unacceptable_entries()
    if unacceptable:
        print(f"\n⚠  UNACCEPTABLE RISKS ({len(unacceptable)}):")
        for e in unacceptable:
            print(f"  {e.scenario} / {e.failure_category}: S={e.severity} × L={e.likelihood} = {e.score}")

    summary = matrix.summary()
    print("\nRisk summary:\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
