"""
Central configuration for the PPO air-jet sorting RL module.

Coordinate convention (matches the simulator):
    x : conveyor / sorting baseline direction (objects move in +x, jet blows in +x)
    y : belt-width / lateral direction
    z : vertical direction

Baseline jet direction: azimuth_deg=0, angle_deg=0  =>  e_jet = [1, 0, 0]  (+x).
Success criterion (boundary mode): landing_x >= target_x_min.
  target_x_min = 0.42 m is just beyond the no-jet max landing range (~0.411 m).
"""

from dataclasses import dataclass


@dataclass
class RLConfig:
    # -----------------------------------------------------------------------
    # RL action mode
    # -----------------------------------------------------------------------
    # "baseline": 3-action policy [Umax, t_on_offset, duration]
    # "elevation": 4-action policy [Umax, t_on_offset, duration, elevation]
    action_mode: str = "elevation"

    # -----------------------------------------------------------------------
    # Reward mode
    # -----------------------------------------------------------------------
    # "boundary": success = landing_x >= target_x_min  (default)
    # "interval": success = target_x_min <= landing_x <= target_x_max (legacy)
    reward_mode: str = "boundary"

    # -----------------------------------------------------------------------
    # Seed ranges
    # -----------------------------------------------------------------------
    train_seed_min: int = 0
    train_seed_max: int = 999
    eval_seed_min: int = 1000
    eval_seed_max: int = 1099

    # -----------------------------------------------------------------------
    # Object randomisation ranges
    # -----------------------------------------------------------------------
    object_types: tuple = ("plate", "rod", "irregular")

    # Plate
    plate_size_x_range: tuple = (0.06, 0.14)   # m
    plate_size_y_range: tuple = (0.06, 0.14)   # m
    plate_size_z_range: tuple = (0.005, 0.020) # m  (thin plate)
    plate_mass_range:   tuple = (0.005, 0.050) # kg

    # Rod
    rod_length_range:  tuple = (0.06, 0.20)    # m
    rod_radius_range:  tuple = (0.004, 0.020)  # m
    rod_mass_range:    tuple = (0.003, 0.030)  # kg

    # Irregular flake
    irreg_size_x_range: tuple = (0.06, 0.14)
    irreg_size_y_range: tuple = (0.06, 0.14)
    irreg_size_z_range: tuple = (0.005, 0.020)
    irreg_mass_range:   tuple = (0.005, 0.050)

    drag_coefficient: float = 1.0   # fixed for all shapes

    # -----------------------------------------------------------------------
    # Initial condition randomisation
    # -----------------------------------------------------------------------
    init_x_range: tuple = (-0.02, 0.02)   # m
    init_y_range: tuple = (-0.01, 0.01)   # m
    init_z:       float = 0.20            # m  (fixed drop height)

    # Conveyor speed (x-direction)
    init_vx_range: tuple = (0.80, 1.20)   # m/s

    # Initial orientation: random roll/pitch/yaw (degrees)
    init_roll_range:  tuple = (-30.0, 30.0)
    init_pitch_range: tuple = (-30.0, 30.0)
    init_yaw_range:   tuple = (0.0, 360.0)

    # Initial angular velocity (rad/s) — small random tumble
    init_omega_range: tuple = (-2.0, 2.0)

    # -----------------------------------------------------------------------
    # Fixed jet geometry  (+x baseline jet)
    # -----------------------------------------------------------------------
    # The nozzle sits at (jet_x, 0, jet_z) and blows in +x.
    # Objects enter free fall around x = 0.18 m and would land at
    # x ≈ 0.34–0.42 m without the jet (depends on vx).
    # Placing the nozzle at jet_x = 0.20 m means objects are downstream
    # of the nozzle for most of free fall, so their -x-facing surfaces
    # (n_x < 0) receive incoming jet flow and are pushed in +x.
    jet_x:             float = 0.20  # m
    jet_y:             float = 0.00  # m  (centred on belt)
    jet_z:             float = 0.18  # m  (near object mid-fall height)
    jet_azimuth_deg:   float = 0.0   # baseline +x in x-y plane
    jet_angle_deg:     float = 0.0   # baseline +x in x-z plane (elevation)
    elevation_min_deg: float = -10.0
    elevation_max_deg: float = 20.0
    jet_sigma:         float = 0.05  # m  Gaussian radius
    jet_axial_decay:   float = 0.35  # m  downstream decay length
    jet_noise_std:     float = 0.0   # fractional noise; >0 for robustness

    # -----------------------------------------------------------------------
    # Action space: physical ranges
    # -----------------------------------------------------------------------
    umax_min:     float = 10.0   # m/s
    umax_max:     float = 30.0   # m/s
    duration_min: float = 0.01   # s
    duration_max: float = 0.10   # s
    t_on_offset:  float = 0.10   # ±s around nominal arrival time

    # -----------------------------------------------------------------------
    # Sorting success criterion
    # -----------------------------------------------------------------------
    # Boundary mode (default): success = landing_x >= target_x_min
    #   target_x_min = 0.42 m is just beyond the no-jet max landing range:
    #     mean no-jet landing_x ≈ 0.355 m, max ≈ 0.411 m
    #   so any jet effect that pushes landing_x past 0.42 counts as sorted.
    #
    # target_x_max = 0.65 m kept for legacy interval mode and documentation.
    # In boundary mode it is NOT a hard failure boundary.
    target_x_min: float = 0.42   # m  (boundary mode threshold)
    target_x_max: float = 0.65   # m  (legacy interval mode upper bound; reference only)

    # -----------------------------------------------------------------------
    # Reward shaping
    # -----------------------------------------------------------------------
    success_bonus:           float = 1.0
    no_landing_penalty:      float = -1.0
    distance_scale:          float = 0.20   # m — divisor for shortfall/overshoot → penalty
    center_bonus_weight:     float = 0.2    # used only in legacy interval mode
    overshoot_soft_start:    float = 0.75   # m — soft overshoot penalty starts here (boundary mode)
    umax_penalty_weight:     float = 0.03   # penalise high jet strength
    duration_penalty_weight: float = 0.02   # penalise long burst
    overshoot_penalty_weight: float = 0.50  # weight for soft overshoot penalty

    # -----------------------------------------------------------------------
    # Simulation settings
    # -----------------------------------------------------------------------
    sim_dt:                   float = 0.001
    sim_t_max:                float = 3.0
    sim_gravity:              float = 9.81
    sim_air_density:          float = 1.225
    sim_landing_z:            float = 0.0
    sim_conveyor_length:      float = 0.15
    sim_free_fall_offset:     float = 0.03

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    n_envs:           int   = 4
    total_timesteps:  int   = 200_000
    learning_rate:    float = 3e-4
    n_steps:          int   = 512
    batch_size:       int   = 64
    n_epochs:         int   = 10
    gamma:            float = 0.99
    ent_coef:         float = 0.01
    policy_arch:      tuple = (128, 128)

    # -----------------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------------
    model_path:      str = "outputs/rl_models/ppo_airjet"
    vecnorm_path:    str = "outputs/rl_models/ppo_airjet_vecnormalize.pkl"
    results_csv:     str = "outputs/rl_results/evaluation_results.csv"
    tensorboard_log: str = "outputs/tb_logs/"


DEFAULT_CONFIG = RLConfig()
