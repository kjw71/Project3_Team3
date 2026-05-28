"""
Evaluate a trained PPO policy on one manually specified 3D object case.

Usage examples:
    python -m rl.test_custom_case --object-type plate --mass 0.02 \
        --size-x 0.10 --size-y 0.10 --size-z 0.01

    python -m rl.test_custom_case --object-type rod --mass 0.015 \
        --size-x 0.16 --size-y 0.02 --size-z 0.02 --yaw-deg 20
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.config import DEFAULT_CONFIG, RLConfig
from rl.env import AirJetSortingEnv
from source.core_3d import (
    InitialCondition3D,
    Jet3D,
    Simulation3D,
    create_object_3d,
    euler_degrees_to_quaternion,
    simulate_rigid_body_3d,
)


DEFAULT_MODEL_PATH = "outputs/rl_models/ppo_airjet_200k_elevation.zip"
DEFAULT_VECNORM_PATH = "outputs/rl_models/ppo_airjet_vecnormalize_200k_elevation.pkl"
ACTION_MODE = "elevation"


def _copy_config(cfg: RLConfig, **overrides: Any) -> RLConfig:
    return RLConfig(**{**cfg.__dict__, **overrides})


def _resolve_model_path(model_path: str) -> str:
    path = Path(model_path)
    if path.exists():
        return str(path)

    if path.suffix != ".zip":
        zipped = Path(str(path) + ".zip")
        if zipped.exists():
            return str(zipped)

    raise FileNotFoundError(f"Model not found: {model_path}")


def _require_file(path: str, label: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return str(file_path)


def _build_object(args: argparse.Namespace, cfg: RLConfig):
    return create_object_3d(
        object_type=args.object_type,
        mass=args.mass,
        size_x=args.size_x,
        size_y=args.size_y,
        size_z=args.size_z,
        drag_coefficient=cfg.drag_coefficient,
        seed=args.seed,
    )


def _build_initial_condition(args: argparse.Namespace) -> InitialCondition3D:
    quat = euler_degrees_to_quaternion(
        args.roll_deg,
        args.pitch_deg,
        args.yaw_deg,
    )
    return InitialCondition3D(
        position=(args.x0, args.y0, args.z0),
        velocity=(args.vx, args.vy, args.vz),
        angular_velocity=(args.wx, args.wy, args.wz),
        quaternion=quat,
    )


def _set_custom_case(
    env: AirJetSortingEnv,
    obj: Any,
    initial: InitialCondition3D,
    seed: int,
) -> None:
    env._obj = obj
    env._initial = initial
    env._shape_seed = seed
    env._ref_area = float(np.sum(obj.area_weights) / 2.0)

    x0 = float(initial.position[0])
    vx = float(initial.velocity[0])
    release_x = x0 + env.cfg.sim_conveyor_length + env.cfg.sim_free_fall_offset
    dist_conveyor = max(release_x - x0, 0.0)
    dist_to_jet = max(env.cfg.jet_x - release_x, 0.0)
    env._t_nominal = (dist_conveyor + dist_to_jet) / max(vx, 1e-6)


def _make_jet(cfg: RLConfig, umax: float, t_on: float, duration: float, elevation_deg: float) -> Jet3D:
    return Jet3D(
        umax=umax,
        t_on=t_on,
        duration=duration,
        x_center=cfg.jet_x,
        y_center=cfg.jet_y,
        z_center=cfg.jet_z,
        azimuth_deg=cfg.jet_azimuth_deg,
        angle_deg=elevation_deg,
        sigma=cfg.jet_sigma,
        axial_decay=cfg.jet_axial_decay,
        noise_std=cfg.jet_noise_std,
    )


def _make_simulation(cfg: RLConfig) -> Simulation3D:
    return Simulation3D(
        dt=cfg.sim_dt,
        t_max=cfg.sim_t_max,
        gravity=cfg.sim_gravity,
        air_density=cfg.sim_air_density,
        landing_z=cfg.sim_landing_z,
        conveyor_length=cfg.sim_conveyor_length,
        free_fall_start_offset=cfg.sim_free_fall_offset,
    )


def _evaluate_case(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _copy_config(DEFAULT_CONFIG, action_mode=ACTION_MODE)
    model_path = _resolve_model_path(args.model_path)
    vecnorm_path = _require_file(args.vecnorm_path, "VecNormalize stats")

    obj = _build_object(args, cfg)
    initial = _build_initial_condition(args)

    env = AirJetSortingEnv(config=cfg)
    _set_custom_case(env, obj, initial, args.seed)

    raw_obs = env._build_obs()

    vec_env = DummyVecEnv([lambda: AirJetSortingEnv(config=cfg)])
    vec_norm = VecNormalize.load(vecnorm_path, vec_env)
    vec_norm.training = False
    vec_norm.norm_reward = False

    model = PPO.load(model_path, env=vec_norm)
    if tuple(model.action_space.shape) != tuple(env.action_space.shape):
        raise ValueError(
            "Model action space does not match elevation env: "
            f"model={model.action_space.shape}, env={env.action_space.shape}"
        )

    norm_obs = vec_norm.normalize_obs(raw_obs[np.newaxis, :])
    action, _ = model.predict(norm_obs, deterministic=True)
    action = np.asarray(action, dtype=np.float32).reshape(-1)

    umax, t_on, duration, elevation_deg = env._decode_action(action)
    jet = _make_jet(cfg, umax, t_on, duration, elevation_deg)
    sim = _make_simulation(cfg)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = simulate_rigid_body_3d(
            obj=obj,
            jet=jet,
            sim=sim,
            initial=initial,
            target=None,
            reference_area=env._ref_area,
            seed=args.seed,
        )

    reward, info = env._compute_reward(result, umax, duration)

    return {
        "cfg": cfg,
        "model_path": model_path,
        "vecnorm_path": vecnorm_path,
        "object": obj,
        "initial": initial,
        "raw_obs": raw_obs,
        "normalized_action": action,
        "umax": umax,
        "t_on": t_on,
        "duration": duration,
        "elevation_deg": elevation_deg,
        "result": result,
        "reward": reward,
        "info": info,
    }


def _fmt_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "None"
    return f"{float(value):.{digits}f}"


def _fmt_action(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):+.6f}" for v in values) + "]"


def _print_summary(args: argparse.Namespace, data: Dict[str, Any]) -> None:
    cfg: RLConfig = data["cfg"]
    info = data["info"]
    result = data["result"]
    action = data["normalized_action"]

    print("=" * 64)
    print("PPO Air-Jet Sorting - Custom Case")
    print(f"model           : {data['model_path']}")
    print(f"vecnormalize    : {data['vecnorm_path']}")
    print(f"action_mode     : {cfg.action_mode}")
    print(f"target interval : [{cfg.target_x_min:.2f}, {cfg.target_x_max:.2f}]")
    print("=" * 64)

    print("\nInput:")
    print(f"  object_type      : {args.object_type}")
    print(f"  seed             : {args.seed}")
    print(f"  mass             : {args.mass:.6f} kg")
    print(f"  size             : ({args.size_x:.6f}, {args.size_y:.6f}, {args.size_z:.6f}) m")
    print(f"  initial position : ({args.x0:.6f}, {args.y0:.6f}, {args.z0:.6f}) m")
    print(f"  velocity         : ({args.vx:.6f}, {args.vy:.6f}, {args.vz:.6f}) m/s")
    print(f"  angular velocity : ({args.wx:.6f}, {args.wy:.6f}, {args.wz:.6f}) rad/s")
    print(
        "  Euler angles     : "
        f"roll={args.roll_deg:.3f} deg, "
        f"pitch={args.pitch_deg:.3f} deg, "
        f"yaw={args.yaw_deg:.3f} deg"
    )

    print("\nPredicted normalized action:")
    print(f"  raw PPO action   : {_fmt_action(action)}")

    print("\nDecoded physical action:")
    print(f"  Umax             : {data['umax']:.6f} m/s")
    print(f"  t_on             : {data['t_on']:.6f} s")
    print(f"  duration         : {data['duration']:.6f} s")
    print(f"  elevation_deg    : {data['elevation_deg']:.6f} deg")

    print("\nResult:")
    print(f"  has_landed       : {info.get('has_landed')}")
    print(f"  landing_time     : {_fmt_float(result.get('landing_time'))} s")
    print(f"  landing_x        : {_fmt_float(info.get('landing_x'))} m")
    print(f"  landing_y        : {_fmt_float(info.get('landing_y'))} m")
    print(f"  landing_z        : {_fmt_float(info.get('landing_z'))} m")
    print(f"  success          : {info.get('success')}")
    print(f"  reward           : {data['reward']:.6f}")
    print(f"  reward_success   : {info.get('reward_success'):.6f}")
    print(f"  reward_center    : {info.get('reward_center'):.6f}")
    print(f"  reward_distance  : {info.get('reward_distance'):.6f}")
    print(f"  reward_overshoot : {info.get('reward_overshoot'):.6f}")
    print(f"  reward_energy    : {info.get('reward_energy'):.6f}")
    print(f"  umax_norm        : {info.get('umax_norm'):.6f}")
    print(f"  duration_norm    : {info.get('duration_norm'):.6f}")


def _build_record(args: argparse.Namespace, data: Dict[str, Any]) -> Dict[str, Any]:
    cfg: RLConfig = data["cfg"]
    info = data["info"]
    result = data["result"]
    action = data["normalized_action"]

    return {
        "object_type": args.object_type,
        "seed": args.seed,
        "mass": args.mass,
        "size_x": args.size_x,
        "size_y": args.size_y,
        "size_z": args.size_z,
        "x0": args.x0,
        "y0": args.y0,
        "z0": args.z0,
        "vx": args.vx,
        "vy": args.vy,
        "vz": args.vz,
        "wx": args.wx,
        "wy": args.wy,
        "wz": args.wz,
        "roll_deg": args.roll_deg,
        "pitch_deg": args.pitch_deg,
        "yaw_deg": args.yaw_deg,
        "action_umax_norm": float(action[0]),
        "action_t_on_norm": float(action[1]),
        "action_duration_norm": float(action[2]),
        "action_elevation_norm": float(action[3]),
        "umax": data["umax"],
        "t_on": data["t_on"],
        "duration": data["duration"],
        "elevation_deg": data["elevation_deg"],
        "has_landed": info.get("has_landed"),
        "landing_time": result.get("landing_time"),
        "landing_x": info.get("landing_x"),
        "landing_y": info.get("landing_y"),
        "landing_z": info.get("landing_z"),
        "success": info.get("success"),
        "reward": data["reward"],
        "reward_success": info.get("reward_success"),
        "reward_center": info.get("reward_center"),
        "reward_distance": info.get("reward_distance"),
        "reward_overshoot": info.get("reward_overshoot"),
        "reward_energy": info.get("reward_energy"),
        "umax_norm": info.get("umax_norm"),
        "duration_norm": info.get("duration_norm"),
        "target_x_min": cfg.target_x_min,
        "target_x_max": cfg.target_x_max,
    }


def _write_csv(output: str, record: Dict[str, Any]) -> None:
    output_path = Path(output)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        writer.writeheader()
        writer.writerow(record)

    print(f"\nCSV saved to: {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PPO on one manually specified air-jet sorting case"
    )
    parser.add_argument("--object-type", choices=("plate", "rod", "irregular"), default="plate")
    parser.add_argument("--mass", type=float, default=0.02)
    parser.add_argument("--size-x", type=float, default=0.10)
    parser.add_argument("--size-y", type=float, default=0.10)
    parser.add_argument("--size-z", type=float, default=0.01)
    parser.add_argument("--x0", type=float, default=0.0)
    parser.add_argument("--y0", type=float, default=0.0)
    parser.add_argument("--z0", type=float, default=0.20)
    parser.add_argument("--vx", type=float, default=1.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--vz", type=float, default=0.0)
    parser.add_argument("--wx", type=float, default=0.0)
    parser.add_argument("--wy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--roll-deg", type=float, default=0.0)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1, help="Irregular geometry seed")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vecnorm-path", type=str, default=DEFAULT_VECNORM_PATH)
    parser.add_argument("--output", type=str, default=None, help="Optional one-row CSV output path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data = _evaluate_case(args)
    _print_summary(args, data)

    if args.output is not None:
        _write_csv(args.output, _build_record(args, data))


if __name__ == "__main__":
    main()
