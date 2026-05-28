"""
Deterministic baselines on the unseen test seeds (1000–1099).

Three non-learned policies are evaluated on EXACTLY the same seed set as
rl.evaluate_ppo, each seed visited once, in deterministic order:

  * random      — uniformly sampled normalized action, fixed RNG seed
  * fixed_best  — normalized action [+1, 0, +1]   (Umax=max, t_on=nominal, D=max)
  * weak        — normalized action [-1, 0, -1]   (Umax=min, t_on=nominal, D=min)

Results land in:
    outputs/rl_results/baseline_random.csv
    outputs/rl_results/baseline_fixed_best.csv
    outputs/rl_results/baseline_weak.csv

Usage:
    python -m rl.baselines
    python -m rl.baselines --only random
    python -m rl.baselines --action-mode baseline
    python -m rl.baselines --action-mode elevation
    python -m rl.baselines --random-seed 123
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from rl.config import RLConfig, DEFAULT_CONFIG
from rl.env import AirJetSortingEnv


# Baseline normalized actions in [-1, 1]^3:
# (umax_norm, t_on_offset_norm, duration_norm)
FIXED_BEST_ACTION = np.array([1.0, 0.0, 1.0], dtype=np.float32)
WEAK_ACTION       = np.array([-1.0, 0.0, -1.0], dtype=np.float32)


def _copy_config(cfg: RLConfig, **overrides) -> RLConfig:
    return RLConfig(**{**cfg.__dict__, **overrides})


def _run_baseline(
    name: str,
    action_fn: Callable[[np.random.Generator, int, RLConfig], np.ndarray],
    cfg: RLConfig,
    out_csv: str,
    random_seed: int,
) -> pd.DataFrame:
    """
    Evaluate a policy that picks an action from action_fn(rng, episode_idx).
    Visits seeds [eval_seed_min .. eval_seed_max] each exactly once.
    """
    eval_seeds = list(range(cfg.eval_seed_min, cfg.eval_seed_max + 1))
    env = AirJetSortingEnv(config=cfg, fixed_seeds=eval_seeds)
    rng = np.random.default_rng(random_seed)

    print("=" * 60)
    print(f"  Baseline: {name}")
    print(f"  eval seeds : {cfg.eval_seed_min}–{cfg.eval_seed_max} "
          f"({len(eval_seeds)} episodes)")
    print(f"  action mode: {cfg.action_mode} ({env.action_space.shape[0]}D)")
    print(f"  target x   : [{cfg.target_x_min:.3f}, {cfg.target_x_max:.3f}] m")
    print(f"  rng seed   : {random_seed}")
    print("=" * 60)

    records: List[Dict] = []
    for ep in range(len(eval_seeds)):
        obs, reset_info = env.reset()
        action = action_fn(rng, ep, cfg)
        obs, reward, terminated, truncated, info = env.step(action)

        records.append(
            {
                "episode":         ep,
                "seed":            info.get("shape_seed"),
                "object_type":     info.get("object_type"),
                "success":         info.get("success", False),
                "has_landed":      info.get("has_landed", False),
                "landing_x":       info.get("landing_x"),
                "landing_y":       info.get("landing_y"),
                "landing_z":       info.get("landing_z"),
                "reward":          float(reward),
                "umax":            info.get("umax"),
                "t_on":            info.get("t_on"),
                "duration":        info.get("duration"),
                "elevation_deg":   info.get("elevation_deg"),
                "reward_success":   info.get("reward_success"),
                "reward_center":    info.get("reward_center"),
                "reward_distance":  info.get("reward_distance"),
                "reward_overshoot": info.get("reward_overshoot"),
                "reward_energy":    info.get("reward_energy"),
                "umax_norm":        info.get("umax_norm"),
                "duration_norm":    info.get("duration_norm"),
            }
        )

    df = pd.DataFrame(records)

    # Coverage check: every requested seed visited exactly once.
    visited = sorted(int(s) for s in df["seed"].dropna().tolist())
    assert visited == eval_seeds, (
        f"[{name}] seed coverage mismatch: got {len(set(visited))} unique "
        f"seeds out of {len(eval_seeds)} expected"
    )

    # Summary
    success_rate = df["success"].mean()
    landed_rate  = df["has_landed"].mean()
    mean_reward  = df["reward"].mean()
    landed_mask  = df["has_landed"].astype(bool)
    mean_lx      = df.loc[landed_mask, "landing_x"].mean() if landed_mask.any() else float("nan")

    print(f"  Episodes evaluated   : {len(df)}")
    print(f"  Unique seeds covered : {len(set(visited))} / {len(eval_seeds)}")
    print(f"  Landing rate         : {landed_rate:.2%}")
    print(f"  Success rate         : {success_rate:.2%}  "
          f"(landing_x in [{cfg.target_x_min:.3f}, {cfg.target_x_max:.3f}])")
    print(f"  Mean reward          : {mean_reward:.4f}")
    print(f"  Mean landing x       : {mean_lx:.4f} m  (landed only)")

    if "object_type" in df.columns:
        print("  Success rate by object type:")
        for otype, grp in df.groupby("object_type"):
            print(f"    {otype:12s}: {grp['success'].mean():.2%}  (n={len(grp)})")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}\n")
    return df


# -- Action functions --------------------------------------------------------

def _zero_elevation_norm(cfg: RLConfig) -> float:
    span = max(cfg.elevation_max_deg - cfg.elevation_min_deg, 1e-12)
    value = 2.0 * (cfg.jet_angle_deg - cfg.elevation_min_deg) / span - 1.0
    return float(np.clip(value, -1.0, 1.0))


def _with_mode(base_action: np.ndarray, cfg: RLConfig) -> np.ndarray:
    if cfg.action_mode == "baseline":
        return base_action.copy()
    return np.concatenate(
        [base_action, np.array([_zero_elevation_norm(cfg)], dtype=np.float32)]
    ).astype(np.float32)


def _random_action(rng: np.random.Generator, _ep: int, cfg: RLConfig) -> np.ndarray:
    # Uniform over the normalized action space; the env maps it to physical
    # jet parameters the same way it would for the policy.
    action_dim = 3 if cfg.action_mode == "baseline" else 4
    return rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)


def _fixed_best_action(
    _rng: np.random.Generator, _ep: int, cfg: RLConfig
) -> np.ndarray:
    return _with_mode(FIXED_BEST_ACTION, cfg)


def _weak_action(_rng: np.random.Generator, _ep: int, cfg: RLConfig) -> np.ndarray:
    return _with_mode(WEAK_ACTION, cfg)


# -- Entry point -------------------------------------------------------------

BASELINES = {
    "random":     ("outputs/rl_results/baseline_random.csv",     _random_action),
    "fixed_best": ("outputs/rl_results/baseline_fixed_best.csv", _fixed_best_action),
    "weak":       ("outputs/rl_results/baseline_weak.csv",       _weak_action),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic baselines for air-jet sorting")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        choices=list(BASELINES.keys()),
        help="Run a single baseline only (default: run all).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="RNG seed for the random-action baseline (reproducibility).",
    )
    parser.add_argument(
        "--action-mode",
        type=str,
        default=None,
        choices=("baseline", "elevation"),
        help="Override action mode (default: from config).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = DEFAULT_CONFIG
    if args.action_mode is not None:
        cfg = _copy_config(cfg, action_mode=args.action_mode)

    if args.only is not None:
        out_csv, action_fn = BASELINES[args.only]
        _run_baseline(args.only, action_fn, cfg, out_csv, args.random_seed)
        return

    for name, (out_csv, action_fn) in BASELINES.items():
        _run_baseline(name, action_fn, cfg, out_csv, args.random_seed)


if __name__ == "__main__":
    main()
