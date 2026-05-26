"""
Evaluate a trained PPO model on unseen test seeds (1000–1099).

Each seed in [eval_seed_min, eval_seed_max] is evaluated EXACTLY ONCE in a
deterministic order, so the success rate is reproducible across runs.

Usage:
    python -m rl.evaluate_ppo
    python -m rl.evaluate_ppo --model outputs/rl_models/ppo_airjet
    python -m rl.evaluate_ppo --action-mode baseline
    python -m rl.evaluate_ppo --action-mode elevation
    python -m rl.evaluate_ppo --stochastic                                # stochastic policy
    python -m rl.evaluate_ppo --output outputs/rl_results/eval_50k.csv    # custom CSV name
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.config import RLConfig, DEFAULT_CONFIG
from rl.env import AirJetSortingEnv


def _copy_config(cfg: RLConfig, **overrides) -> RLConfig:
    return RLConfig(**{**cfg.__dict__, **overrides})


def _action_mode_from_model(model: PPO) -> str:
    shape = tuple(model.action_space.shape)
    if shape == (3,):
        return "baseline"
    if shape == (4,):
        return "elevation"
    raise ValueError(f"Unsupported model action space shape: {shape}")


def evaluate(
    cfg: RLConfig = DEFAULT_CONFIG,
    model_path: str | None = None,
    vecnorm_path: str | None = None,
    deterministic: bool = True,
    output_csv: str | None = None,
    action_mode: str | None = None,
) -> pd.DataFrame:
    """
    Run evaluation on all unseen seeds, each visited exactly once.
    Returns a DataFrame with one row per seed.
    """
    model_path   = model_path   or cfg.model_path
    vecnorm_path = vecnorm_path or cfg.vecnorm_path
    output_csv   = output_csv   or cfg.results_csv

    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(
            f"Model not found at {model_path}.zip — run train_ppo.py first."
        )
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(
            f"VecNormalize stats not found at {vecnorm_path} — run train_ppo.py first."
        )

    model_probe = PPO.load(model_path)
    model_action_mode = _action_mode_from_model(model_probe)
    if action_mode is None:
        cfg = _copy_config(cfg, action_mode=model_action_mode)
    else:
        cfg = _copy_config(cfg, action_mode=action_mode)
        if action_mode != model_action_mode:
            raise ValueError(
                f"Requested action_mode={action_mode!r}, but model action "
                f"space is {model_probe.action_space.shape} ({model_action_mode!r})."
            )

    eval_seeds = list(range(cfg.eval_seed_min, cfg.eval_seed_max + 1))
    action_dim = 3 if cfg.action_mode == "baseline" else 4

    print("=" * 60)
    print("  PPO Air-Jet Sorting — Evaluation")
    print(f"  model         : {model_path}.zip")
    print(f"  eval seeds    : {cfg.eval_seed_min}–{cfg.eval_seed_max}  "
          f"({len(eval_seeds)} episodes)")
    print(f"  action_mode   : {cfg.action_mode} ({action_dim}D)")
    print(f"  target x      : [{cfg.target_x_min:.3f}, {cfg.target_x_max:.3f}] m")
    print(f"  deterministic : {deterministic}")
    print("=" * 60)

    # Build a single eval env that iterates through the fixed seed list.
    def _make_eval_env():
        return AirJetSortingEnv(config=cfg, fixed_seeds=eval_seeds)

    eval_vec = DummyVecEnv([_make_eval_env])
    eval_vec = VecNormalize.load(vecnorm_path, eval_vec)
    eval_vec.training = False
    eval_vec.norm_reward = False

    model = PPO.load(model_path, env=eval_vec)

    records = []
    obs = eval_vec.reset()
    for ep in range(len(eval_seeds)):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info_list = eval_vec.step(action)

        info = info_list[0]
        records.append(
            {
                "episode": ep,
                "seed": info.get("shape_seed"),
                "object_type": info.get("object_type"),
                "success": info.get("success", False),
                "has_landed": info.get("has_landed", False),
                "landing_x": info.get("landing_x"),
                "landing_y": info.get("landing_y"),
                "landing_z": info.get("landing_z"),
                "reward": float(reward[0]),
                "umax": info.get("umax"),
                "t_on": info.get("t_on"),
                "duration": info.get("duration"),
                "elevation_deg": info.get("elevation_deg"),
                "reward_success": info.get("reward_success"),
                "reward_center": info.get("reward_center"),
                "reward_distance": info.get("reward_distance"),
                "reward_energy": info.get("reward_energy"),
            }
        )
        # DummyVecEnv auto-resets on done; env.fixed_seeds advances the index.

    df = pd.DataFrame(records)

    # Verify coverage
    visited = df["seed"].dropna().astype(int).tolist()
    unique  = sorted(set(visited))
    assert sorted(visited) == eval_seeds, (
        f"Seed coverage mismatch: {len(unique)} unique seeds out of "
        f"{len(eval_seeds)} expected"
    )

    # Summary
    success_rate = df["success"].mean()
    landed_rate  = df["has_landed"].mean()
    mean_reward  = df["reward"].mean()
    landed_mask  = df["has_landed"].astype(bool)
    mean_lx      = df.loc[landed_mask, "landing_x"].mean() if landed_mask.any() else float("nan")

    print(f"\n{'─'*48}")
    print(f"  Episodes evaluated   : {len(df)}")
    print(f"  Unique seeds covered : {len(unique)} / {len(eval_seeds)}")
    print(f"  Landing rate         : {landed_rate:.2%}")
    print(f"  Success rate         : {success_rate:.2%}  "
          f"(landing_x in [{cfg.target_x_min:.3f}, {cfg.target_x_max:.3f}])")
    print(f"  Mean reward          : {mean_reward:.4f}")
    print(f"  Mean landing x       : {mean_lx:.4f} m  (landed only)")
    print(f"{'─'*48}")

    if "object_type" in df.columns:
        print("\n  Success rate by object type:")
        for otype, grp in df.groupby("object_type"):
            print(f"    {otype:12s}: {grp['success'].mean():.2%}  (n={len(grp)})")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n  Results saved to: {output_csv}")

    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO for air-jet sorting")
    parser.add_argument("--model",     type=str, default=None, help="Path to model (without .zip)")
    parser.add_argument("--vecnorm",   type=str, default=None, help="Path to VecNormalize pkl")
    parser.add_argument("--output",    type=str, default=None,
                        help="Output CSV path (default: cfg.results_csv)")
    parser.add_argument("--action-mode", type=str, default=None,
                        choices=("baseline", "elevation"),
                        help="Override action mode; default auto-detects from the model")
    parser.add_argument("--stochastic", action="store_true",   help="Use stochastic policy")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        deterministic=not args.stochastic,
        output_csv=args.output,
        action_mode=args.action_mode,
    )
