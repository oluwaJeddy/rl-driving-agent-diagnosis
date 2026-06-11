"""
PPO agent training on highway-env using Stable-Baselines3.

Usage:
    python src/train.py
    python src/train.py --env highway-v0 --timesteps 500000 --output results/checkpoints/ppo_highway
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401 – registers envs as a side-effect
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize


# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    env_id="highway-v0",
    timesteps=300_000,
    n_envs=4,
    n_steps=256,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    output="results/checkpoints/ppo_highway",
    eval_freq=10_000,
    save_freq=50_000,
)

ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "normalize": True,
        "absolute": False,
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 15,
    "duration": 40,
    "reward_speed_range": [20, 30],
    "collision_reward": -5.0,
    "high_speed_reward": 0.4,
    "lane_change_reward": 0.1,
}


def make_env(env_id: str, config: dict | None = None):
    def _init():
        env = gym.make(env_id, config=config or ENV_CONFIG)
        return env
    return _init


def build_model(vec_env, args: argparse.Namespace) -> PPO:
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    return PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(Path(args.output).parent / "tb_logs"),
        verbose=1,
    )


def train(args: argparse.Namespace) -> PPO:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    vec_env = make_vec_env(
        make_env(args.env_id),
        n_envs=args.n_envs,
    )
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    eval_env = make_vec_env(make_env(args.env_id), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),
        save_path=str(output_dir),
        name_prefix="ppo",
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    model = build_model(vec_env, args)
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    final_path = str(output_dir / "ppo_final")
    model.save(final_path)
    vec_env.save(str(output_dir / "vec_normalize.pkl"))
    print(f"Model saved → {final_path}.zip")
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on highway-env")
    for key, val in DEFAULTS.items():
        t = type(val)
        p.add_argument(f"--{key}", type=t, default=val)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
