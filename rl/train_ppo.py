"""
PPO training script for air-jet sorting.

Usage:
    python -m rl.train_ppo                        # default 200k steps
    python -m rl.train_ppo --total-timesteps 500000
    python -m rl.train_ppo --action-mode baseline
    python -m rl.train_ppo --action-mode elevation
    python -m rl.train_ppo --total-timesteps 1000  # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from rl.config import RLConfig, DEFAULT_CONFIG
from rl.env import AirJetSortingEnv


# ---------------------------------------------------------------------------
# Callback: track success rate during training
# ---------------------------------------------------------------------------

class SuccessRateCallback(BaseCallback):
    """Log episode success rate to TensorBoard every N rollouts."""

    def __init__(self, log_freq: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self._log_freq = log_freq
        self._ep_successes: list[bool] = []
        self._ep_rewards: list[float] = []
        self._rollout_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is not None:
                self._ep_rewards.append(float(ep_info["r"]))
            success = info.get("success")
            if success is not None:
                self._ep_successes.append(bool(success))
        return True

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self._rollout_count % self._log_freq == 0 and self._ep_successes:
            rate = float(np.mean(self._ep_successes[-200:]))
            mean_r = float(np.mean(self._ep_rewards[-200:])) if self._ep_rewards else 0.0
            self.logger.record("custom/success_rate", rate)
            self.logger.record("custom/mean_episode_reward", mean_r)
            if self.verbose:
                step = self.num_timesteps
                print(f"  [step {step:>8d}] success_rate={rate:.2%}  mean_reward={mean_r:.3f}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    cfg: RLConfig = DEFAULT_CONFIG,
    total_timesteps: int | None = None,
    action_mode: str | None = None,
) -> str:
    """
    Train PPO on the air-jet sorting environment.

    Returns the path to the saved model zip.
    """
    overrides = {}
    if total_timesteps is not None:
        overrides["total_timesteps"] = total_timesteps
    if action_mode is not None:
        overrides["action_mode"] = action_mode
    if overrides:
        cfg = RLConfig(**{**cfg.__dict__, **overrides})

    action_dim = 3 if cfg.action_mode == "baseline" else 4

    os.makedirs(os.path.dirname(cfg.model_path), exist_ok=True)
    os.makedirs(cfg.tensorboard_log, exist_ok=True)

    print("=" * 60)
    print("  PPO Air-Jet Sorting — Training")
    print(f"  total_timesteps : {cfg.total_timesteps:,}")
    print(f"  n_envs          : {cfg.n_envs}")
    print(f"  action_mode     : {cfg.action_mode} ({action_dim}D)")
    if cfg.action_mode == "elevation":
        print(f"  elevation range : [{cfg.elevation_min_deg:.1f}, {cfg.elevation_max_deg:.1f}] deg")
    print(f"  train seeds     : {cfg.train_seed_min}–{cfg.train_seed_max}")
    print(f"  model path      : {cfg.model_path}.zip")
    print("=" * 60)

    # Vectorised training environment
    def _make_env():
        return AirJetSortingEnv(
            config=cfg,
            seed_range=(cfg.train_seed_min, cfg.train_seed_max),
        )

    vec_env = make_vec_env(_make_env, n_envs=cfg.n_envs)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # PPO model
    policy_kwargs = dict(net_arch=list(cfg.policy_arch))
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        ent_coef=cfg.ent_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log=cfg.tensorboard_log,
        verbose=1,
    )

    callback = SuccessRateCallback(log_freq=5, verbose=1)

    t_start = time.time()
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callback,
        tb_log_name="ppo_airjet",
        reset_num_timesteps=True,
    )
    elapsed = time.time() - t_start

    # Save model + VecNormalize statistics
    model.save(cfg.model_path)
    vec_env.save(cfg.vecnorm_path)

    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Model saved to : {cfg.model_path}.zip")
    print(f"VecNorm saved to: {cfg.vecnorm_path}")

    # Quick training summary
    summary = {
        "total_timesteps": cfg.total_timesteps,
        "n_envs": cfg.n_envs,
        "action_mode": cfg.action_mode,
        "action_dim": action_dim,
        "elevation_min_deg": cfg.elevation_min_deg,
        "elevation_max_deg": cfg.elevation_max_deg,
        "center_bonus_weight": cfg.center_bonus_weight,
        "elapsed_seconds": round(elapsed, 1),
        "train_seed_range": [cfg.train_seed_min, cfg.train_seed_max],
        "eval_seed_range": [cfg.eval_seed_min, cfg.eval_seed_max],
        "policy_arch": list(cfg.policy_arch),
        "learning_rate": cfg.learning_rate,
        "target_x_min": cfg.target_x_min,
        "target_x_max": cfg.target_x_max,
    }
    summary_path = cfg.model_path + "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    return cfg.model_path + ".zip"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO for air-jet sorting")
    parser.add_argument(
        "--total-timesteps", type=int, default=None,
        help="Override total training timesteps (default: from config)"
    )
    parser.add_argument(
        "--action-mode",
        type=str,
        default=None,
        choices=("baseline", "elevation"),
        help="Override action mode (default: from config)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(total_timesteps=args.total_timesteps, action_mode=args.action_mode)
