"""
Gymnasium environment for PPO-based air-jet sorting (one-step pre-fire).

Coordinate convention (matches the simulator):
    x  conveyor + baseline jet direction (objects move in +x, jet blows in +x)
    y  belt-width direction
    z  vertical

Success criterion:
    target_x_min <= landing_x <= target_x_max

Landing convention:
    COM position at the first timestep when the lowest surface point reaches
    landing_z.  Applied consistently for reward and evaluation.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Make the parent project importable when running `python -m rl.env`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from source.core_3d import (
    Object3D,
    Jet3D,
    Simulation3D,
    InitialCondition3D,
    create_object_3d,
    simulate_rigid_body_3d,
    euler_degrees_to_quaternion,
)
from rl.config import RLConfig, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Observation dimension breakdown (keep in sync with _build_obs)
# ---------------------------------------------------------------------------
# object type one-hot     : 3
# mass                    : 1
# drag_coefficient        : 1
# size_x, y, z            : 3
# inertia Ixx, Iyy, Izz   : 3
# reference_area          : 1
# init pos x, y, z        : 3
# init vel vx, vy, vz     : 3
# init omega x, y, z      : 3
# init quaternion w,x,y,z : 4
# jet centre x, y, z      : 3
# target_x_min, target_x_max : 2
# --------------------------------
# TOTAL                   : 30
OBS_DIM = 30


class AirJetSortingEnv(gym.Env):
    """
    One-step Gymnasium environment for +x air-jet sorting with PPO.

    The agent observes the object's initial state, picks jet parameters once,
    and the simulator runs the full episode.  Reward is computed from the final
    landing_x against [target_x_min, target_x_max].

    Args:
        config: hyperparameters / physics ranges.
        seed_range: (min, max) inclusive when sampling random shape seeds.
        fixed_seeds: optional explicit sequence of shape seeds; if provided,
            reset() iterates through them deterministically (used for eval).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: RLConfig = DEFAULT_CONFIG,
        seed_range: Optional[Tuple[int, int]] = None,
        fixed_seeds: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.cfg = config
        if self.cfg.action_mode not in ("baseline", "elevation"):
            raise ValueError(
                "action_mode must be 'baseline' or 'elevation', "
                f"got {self.cfg.action_mode!r}"
            )

        if seed_range is not None:
            self._seed_min, self._seed_max = seed_range
        else:
            self._seed_min = config.train_seed_min
            self._seed_max = config.train_seed_max

        self._fixed_seeds: Optional[List[int]] = (
            [int(s) for s in fixed_seeds] if fixed_seeds is not None else None
        )
        self._fixed_idx: int = 0

        # Action in [-1, 1]:
        #   baseline  -> [umax_norm, t_on_norm, duration_norm]
        #   elevation -> [umax_norm, t_on_norm, duration_norm, elevation_norm]
        action_dim = 3 if self.cfg.action_mode == "baseline" else 4
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        self._obj: Optional[Object3D] = None
        self._initial: Optional[InitialCondition3D] = None
        self._ref_area: float = 0.0
        self._t_nominal: float = 0.0
        self._shape_seed: int = 0
        self._episode_rng: Optional[np.random.Generator] = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # --- Pick the episode's shape seed ---------------------------------
        if self._fixed_seeds is not None:
            self._shape_seed = self._fixed_seeds[
                self._fixed_idx % len(self._fixed_seeds)
            ]
            self._fixed_idx += 1
        else:
            meta_rng = self.np_random
            self._shape_seed = int(
                meta_rng.integers(self._seed_min, self._seed_max + 1)
            )

        self._episode_rng = np.random.default_rng(self._shape_seed)
        rng = self._episode_rng

        # --- Object --------------------------------------------------------
        obj_type = str(rng.choice(list(self.cfg.object_types)))
        obj = self._sample_object(obj_type, rng)
        self._obj = obj

        # --- Initial conditions -------------------------------------------
        x0 = float(rng.uniform(*self.cfg.init_x_range))
        y0 = float(rng.uniform(*self.cfg.init_y_range))
        z0 = self.cfg.init_z
        vx = float(rng.uniform(*self.cfg.init_vx_range))

        roll  = float(rng.uniform(*self.cfg.init_roll_range))
        pitch = float(rng.uniform(*self.cfg.init_pitch_range))
        yaw   = float(rng.uniform(*self.cfg.init_yaw_range))
        quat  = euler_degrees_to_quaternion(roll, pitch, yaw)

        omega = tuple(float(rng.uniform(*self.cfg.init_omega_range)) for _ in range(3))

        self._initial = InitialCondition3D(
            position=(x0, y0, z0),
            velocity=(vx, 0.0, 0.0),
            quaternion=quat,
            angular_velocity=omega,
        )

        self._ref_area = float(np.sum(obj.area_weights) / 2.0)

        # Nominal jet arrival time: time for object COM to reach jet_x
        # (constant-vx estimate; the agent can offset via action[1])
        release_x = x0 + self.cfg.sim_conveyor_length + self.cfg.sim_free_fall_offset
        dist_conveyor = max(release_x - x0, 0.0)
        dist_to_jet   = max(self.cfg.jet_x - release_x, 0.0)
        self._t_nominal = (dist_conveyor + dist_to_jet) / max(vx, 1e-6)

        obs = self._build_obs()
        return obs, {"shape_seed": self._shape_seed, "object_type": obj_type}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        umax, t_on, duration, elevation_deg = self._decode_action(action)

        jet = Jet3D(
            umax=umax,
            t_on=t_on,
            duration=duration,
            x_center=self.cfg.jet_x,
            y_center=self.cfg.jet_y,
            z_center=self.cfg.jet_z,
            azimuth_deg=self.cfg.jet_azimuth_deg,
            angle_deg=elevation_deg,
            sigma=self.cfg.jet_sigma,
            axial_decay=self.cfg.jet_axial_decay,
            noise_std=self.cfg.jet_noise_std,
        )

        sim = Simulation3D(
            dt=self.cfg.sim_dt,
            t_max=self.cfg.sim_t_max,
            gravity=self.cfg.sim_gravity,
            air_density=self.cfg.sim_air_density,
            landing_z=self.cfg.sim_landing_z,
            conveyor_length=self.cfg.sim_conveyor_length,
            free_fall_start_offset=self.cfg.sim_free_fall_offset,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = simulate_rigid_body_3d(
                obj=self._obj,
                jet=jet,
                sim=sim,
                initial=self._initial,
                target=None,           # we recompute success here
                reference_area=self._ref_area,
                seed=self._shape_seed,
            )

        reward, info = self._compute_reward(result, umax, duration)
        info.update(
            {
                "shape_seed": self._shape_seed,
                "object_type": self._obj.object_type,
                "umax": umax,
                "t_on": t_on,
                "duration": duration,
                "elevation_deg": elevation_deg,
                "t_nominal": self._t_nominal,
            }
        )

        obs = self._build_obs()
        return obs, reward, True, False, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_object(self, obj_type: str, rng: np.random.Generator) -> Object3D:
        cfg = self.cfg
        if obj_type == "plate":
            return create_object_3d(
                object_type="plate",
                mass=float(rng.uniform(*cfg.plate_mass_range)),
                size_x=float(rng.uniform(*cfg.plate_size_x_range)),
                size_y=float(rng.uniform(*cfg.plate_size_y_range)),
                size_z=float(rng.uniform(*cfg.plate_size_z_range)),
                drag_coefficient=cfg.drag_coefficient,
                seed=self._shape_seed,
            )
        if obj_type == "rod":
            rl = float(rng.uniform(*cfg.rod_length_range))
            rr = float(rng.uniform(*cfg.rod_radius_range))
            return create_object_3d(
                object_type="rod",
                mass=float(rng.uniform(*cfg.rod_mass_range)),
                size_x=rl, size_y=2*rr, size_z=2*rr,
                rod_length=rl, rod_radius=rr,
                drag_coefficient=cfg.drag_coefficient,
                seed=self._shape_seed,
            )
        if obj_type == "irregular":
            return create_object_3d(
                object_type="irregular",
                mass=float(rng.uniform(*cfg.irreg_mass_range)),
                size_x=float(rng.uniform(*cfg.irreg_size_x_range)),
                size_y=float(rng.uniform(*cfg.irreg_size_y_range)),
                size_z=float(rng.uniform(*cfg.irreg_size_z_range)),
                drag_coefficient=cfg.drag_coefficient,
                seed=self._shape_seed,
            )
        raise ValueError(f"Unknown object type: {obj_type}")

    def _decode_action(self, action: np.ndarray) -> Tuple[float, float, float, float]:
        """Map normalized action to physical jet parameters."""
        cfg = self.cfg
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < self.action_space.shape[0]:
            raise ValueError(
                f"Expected action with {self.action_space.shape[0]} values, "
                f"got {action.size}"
            )

        a0 = float(np.clip(action[0], -1.0, 1.0))
        umax = cfg.umax_min * (cfg.umax_max / cfg.umax_min) ** ((a0 + 1.0) / 2.0)

        a1 = float(np.clip(action[1], -1.0, 1.0))
        t_on = max(0.0, self._t_nominal + a1 * cfg.t_on_offset)

        a2 = float(np.clip(action[2], -1.0, 1.0))
        duration = cfg.duration_min * (cfg.duration_max / cfg.duration_min) ** ((a2 + 1.0) / 2.0)

        if cfg.action_mode == "elevation":
            a3 = float(np.clip(action[3], -1.0, 1.0))
            elevation = cfg.elevation_min_deg + 0.5 * (a3 + 1.0) * (
                cfg.elevation_max_deg - cfg.elevation_min_deg
            )
        else:
            elevation = cfg.jet_angle_deg

        return float(umax), float(t_on), float(duration), float(elevation)

    def _build_obs(self) -> np.ndarray:
        obj = self._obj
        ic  = self._initial
        cfg = self.cfg

        type_map = {"plate": 0, "rod": 1, "irregular": 2}
        type_onehot = np.zeros(3, dtype=np.float32)
        type_onehot[type_map.get(obj.object_type, 2)] = 1.0

        inertia_diag = np.diag(obj.inertia_body).astype(np.float32)

        obs = np.concatenate([
            type_onehot,                                          # 3
            [obj.mass * 100.0],                                   # 1
            [obj.drag_coefficient],                               # 1
            [obj.size_x * 10.0, obj.size_y * 10.0, obj.size_z * 100.0],  # 3
            inertia_diag * 1e4,                                   # 3
            [self._ref_area * 1e3],                               # 1
            [ic.position[0] * 5.0, ic.position[1] * 20.0, ic.position[2] * 5.0],
            [ic.velocity[0], ic.velocity[1], ic.velocity[2]],
            list(ic.angular_velocity),                            # 3
            list(ic.quaternion),                                  # 4
            [cfg.jet_x * 3.0, cfg.jet_y * 5.0, cfg.jet_z * 5.0],  # 3
            [cfg.target_x_min * 2.0, cfg.target_x_max * 2.0],     # 2
        ], dtype=np.float32)

        assert len(obs) == OBS_DIM, f"Obs dim mismatch: {len(obs)} != {OBS_DIM}"
        return obs

    def _compute_reward(
        self, result: Dict[str, Any], umax: float, duration: float
    ) -> Tuple[float, Dict[str, Any]]:
        cfg = self.cfg
        has_landed = bool(result.get("has_landed", False))
        landing_pos = result.get("landing_position", None)

        # Treat non-landing OR non-finite trajectories as failed episodes.
        if (
            not has_landed
            or landing_pos is None
            or not np.all(np.isfinite(landing_pos))
        ):
            info = {
                "success": False,
                "has_landed": False,
                "landing_x": None,
                "landing_y": None,
                "landing_z": None,
                "reward_success": 0.0,
                "reward_center": 0.0,
                "reward_distance": 0.0,
                "reward_energy": 0.0,
            }
            return cfg.no_landing_penalty, info

        lx, ly, lz = float(landing_pos[0]), float(landing_pos[1]), float(landing_pos[2])

        # Success: landing_x inside the target x-interval
        success = (cfg.target_x_min <= lx <= cfg.target_x_max)
        r_success = cfg.success_bonus if success else 0.0

        # Center-seeking bonus only within the fixed success interval
        if success:
            target_center = 0.5 * (cfg.target_x_min + cfg.target_x_max)
            half_width = 0.5 * (cfg.target_x_max - cfg.target_x_min)
            center_error = abs(lx - target_center) / max(half_width, 1e-12)
            r_center = cfg.center_bonus_weight * max(0.0, 1.0 - center_error)
        else:
            r_center = 0.0

        # Distance to the target interval (0 if inside)
        if lx < cfg.target_x_min:
            distance = cfg.target_x_min - lx
        elif lx > cfg.target_x_max:
            distance = lx - cfg.target_x_max
        else:
            distance = 0.0
        r_distance = -(distance / cfg.distance_scale)

        # Energy penalty: discourage wasteful high-power long bursts
        umax_norm     = (umax - cfg.umax_min) / (cfg.umax_max - cfg.umax_min)
        duration_norm = (duration - cfg.duration_min) / (cfg.duration_max - cfg.duration_min)
        r_energy = -cfg.energy_penalty_w * (umax_norm + duration_norm)

        reward = r_success + r_center + r_distance + r_energy

        info = {
            "success": success,
            "has_landed": True,
            "landing_x": lx,
            "landing_y": ly,
            "landing_z": lz,
            "reward_success": r_success,
            "reward_center": r_center,
            "reward_distance": r_distance,
            "reward_energy": r_energy,
        }
        return float(reward), info


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    env = AirJetSortingEnv()
    obs, info = env.reset(seed=0)
    print(f"obs shape : {obs.shape}")
    print(f"action mode: {env.cfg.action_mode}")
    print(f"action shape: {env.action_space.shape}")
    print(f"reset info: {info}")
    print(f"target    : [{env.cfg.target_x_min}, {env.cfg.target_x_max}] m\n")

    successes = []
    t0 = time.time()
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        successes.append(info.get("success", False))
        lx = info.get("landing_x")
        lx_str = f"{lx:.3f}" if lx is not None else "N/A  "
        print(
            f"ep {i:2d} | obj={str(info['object_type']):<9s} | "
            f"landed={str(info['has_landed']):<5s} | lx={lx_str} | "
            f"elev={info['elevation_deg']:+.1f} deg | "
            f"success={str(info['success']):<5s} | reward={reward:+.3f}"
        )
        obs, info = env.reset()

    print(f"\nSuccess rate (random actions): {sum(successes)/len(successes):.1%}")
    print(f"Wall time for 20 episodes: {time.time()-t0:.2f}s")
