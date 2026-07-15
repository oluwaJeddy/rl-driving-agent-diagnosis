"""
Unified multi-seed training script for the four-condition experiment matrix.

Conditions (all 50,000 timesteps, checkpoint every 5,000 total steps):
  baseline         – collision -5.0,  no on_road weight (reproduces legacy run)
  full_ablation    – collision -10.0, on_road 1.0       (mirrors train_fixed_reward.py)
  on_road_only     – collision -5.0,  on_road 1.0       (isolates on_road signal)
  collision10_only – collision -10.0, no on_road        (isolates penalty magnitude; cleanest)

Reward normalisation notes (highway-env v1.11, normalize_reward=True by default):
  The normaliser denominator is always high_speed_reward + right_lane_reward = 0.5,
  regardless of on_road_reward.  In conditions with on_road_reward=1.0, r_raw peaks
  at 1.5 so r_norm can exceed 1.0 during normal driving; VecNormalize compensates.
  collision10_only is the only condition where the normalization bounds are tight.

  Condition          norm range        r_norm(collision@vmax) r_norm(safe@vmax)
  baseline           [-5.0,  0.5]      0.091                  1.000
  full_ablation      [-10.0, 0.5]      0.143                  ~1.095  (>1)
  on_road_only       [-5.0,  0.5]      0.273                  ~1.182  (>1)
  collision10_only   [-10.0, 0.5]      0.048                  1.000

Slurm usage (see slurm/experiments.slurm for the job array):
  python src/train_experiment.py --condition baseline --seed 0
  python src/train_experiment.py --condition full_ablation --seed 3
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401 – side-effect: registers highway-v0
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize


# ---------------------------------------------------------------------------
# Shared observation / action / environment settings (identical across all
# conditions so that the only variables are the reward weights).
# ---------------------------------------------------------------------------
_OBS_CFG = {
    "type": "Kinematics",
    "vehicles_count": 10,
    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
    "normalize": True,
    "absolute": False,
}

_ENV_BASE = {
    "observation": _OBS_CFG,
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 15,
    "duration": 40,
    "reward_speed_range": [20, 30],
    "high_speed_reward": 0.4,
    # right_lane_reward:  0.1  (highway-env default; not overridden)
    # normalize_reward:   True (highway-env default; not overridden)
    # offroad_terminal:   False(highway-env default; not overridden)
}

# ---------------------------------------------------------------------------
# Condition-specific reward weight overrides (merged with _ENV_BASE at run time)
# ---------------------------------------------------------------------------
CONDITIONS: dict[str, dict] = {
    "baseline": {
        "collision_reward": -5.0,
        # lane_change_reward 0.1 kept for exact legacy reproduction even though
        # it is inert in v1.11 (_rewards() never emits that key).
        "lane_change_reward": 0.1,
    },
    "full_ablation": {
        "collision_reward": -10.0,
        "lane_change_reward": 0.0,
        "on_road_reward": 1.0,
    },
    "on_road_only": {
        "collision_reward": -5.0,
        "lane_change_reward": 0.0,
        "on_road_reward": 1.0,
    },
    "collision10_only": {
        "collision_reward": -10.0,
        "lane_change_reward": 0.0,
        # on_road_reward intentionally absent → config.get("on_road_reward", 0) = 0
    },
}

# ---------------------------------------------------------------------------
# Training constants (match both legacy scripts exactly)
# ---------------------------------------------------------------------------
TOTAL_TIMESTEPS  = 50_000
CHECKPOINT_EVERY = 5_000   # in total environment steps
EVAL_EVERY       = 10_000  # in total environment steps
N_ENVS           = 4

_PPO_KWARGS = dict(
    n_steps      = 256,
    batch_size   = 64,
    n_epochs     = 10,
    learning_rate= 3e-4,
    gamma        = 0.99,
    gae_lambda   = 0.95,
    clip_range   = 0.2,
    ent_coef     = 0.01,
    policy_kwargs= dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    verbose      = 1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def _make_env_fn(condition: str):
    """Return a zero-argument callable that constructs the configured env."""
    cfg = {**_ENV_BASE, **CONDITIONS[condition]}
    def _init():
        return gym.make("highway-v0", config=cfg)
    return _init


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    _seed_everything(args.seed)

    output_dir = Path(args.output_root) / args.condition / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_cfg = {**_ENV_BASE, **CONDITIONS[args.condition]}
    print(f"Condition : {args.condition}")
    print(f"Seed      : {args.seed}")
    print(f"Output    : {output_dir}")
    print(f"Reward config overrides: {CONDITIONS[args.condition]}")
    print()

    vec_env = make_vec_env(
        _make_env_fn(args.condition),
        n_envs=N_ENVS,
        seed=args.seed,
    )
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    # Separate eval env; training=False so it does not update running stats.
    eval_env = make_vec_env(
        _make_env_fn(args.condition),
        n_envs=1,
        seed=args.seed + 1000,
    )
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    checkpoint_cb = CheckpointCallback(
        save_freq    = max(CHECKPOINT_EVERY // N_ENVS, 1),
        save_path    = str(output_dir),
        name_prefix  = f"ppo_{args.condition}",
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best"),
        log_path    = str(output_dir / "eval_logs"),
        eval_freq   = max(EVAL_EVERY // N_ENVS, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        seed=args.seed,
        tensorboard_log=str(Path(args.output_root) / "tb_logs" / args.condition),
        **_PPO_KWARGS,
    )
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    final_path   = output_dir / f"ppo_{args.condition}_final"
    vecnorm_path = output_dir / "vec_normalize.pkl"
    model.save(str(final_path))
    vec_env.save(str(vecnorm_path))

    print(f"\nModel   → {final_path}.zip")
    print(f"VecNorm → {vecnorm_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train PPO on highway-env for one (condition, seed) in the experiment matrix.\n"
            f"Conditions: {', '.join(CONDITIONS)}"
        )
    )
    p.add_argument(
        "--condition", required=True, choices=list(CONDITIONS),
        help="Reward condition to train",
    )
    p.add_argument(
        "--seed", type=int, required=True,
        help="Random seed (set for numpy / random / torch and passed to SB3 + envs)",
    )
    p.add_argument(
        "--output_root", default="results/checkpoints",
        help="Root output dir; model saved to <root>/<condition>/seed_<seed>/",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
