# RL Module — PPO Air-Jet Sorting

PPO agent that learns to fire an air jet at the right time, strength, and duration to sort random 3D rigid particles along the **+x** conveyor direction.

---

## Installation

```bash
pip install -r requirements.txt
# Requires: gymnasium, stable-baselines3, torch, tensorboard
```

---

## How to train

```bash
# Default (200 000 steps)
python -m rl.train_ppo

# Train the old 3-action baseline policy
python -m rl.train_ppo --action-mode baseline

# Train the new 4-action elevation policy
python -m rl.train_ppo --action-mode elevation

# Full training run
python -m rl.train_ppo --total-timesteps 500000
```

Saved files:
- `outputs/rl_models/ppo_airjet.zip` — PPO weights
- `outputs/rl_models/ppo_airjet_vecnormalize.pkl` — observation normalisation stats
- `outputs/rl_models/ppo_airjet_summary.json` — training metadata
- `outputs/tb_logs/` — TensorBoard logs

---

## How to evaluate

```bash
python -m rl.evaluate_ppo
python -m rl.evaluate_ppo --action-mode baseline
python -m rl.evaluate_ppo --action-mode elevation
```

Each unseen seed in `[1000, 1099]` is evaluated **exactly once** in deterministic order. If `--action-mode` is omitted, evaluation auto-detects baseline vs elevation mode from the saved PPO model action space.

Output: `outputs/rl_results/evaluation_results.csv`
Columns: `episode, seed, object_type, success, has_landed, landing_x, landing_y, landing_z, reward, umax, t_on, duration, elevation_deg, reward_success, reward_undershoot, reward_overshoot, reward_energy, reward_center, reward_distance, umax_norm, duration_norm, target_x_min, overshoot_soft_start, reward_mode`

---

## Smoke test

```bash
python -m rl.env       # 20 random-action episodes, no training required
```

---

## Design overview

### Formulation: pre-fire parameter selection

The agent observes the object's initial state and selects jet parameters **once** before the simulation runs. This matches real industrial air-jet sorters, where a sensor triggers a timed pulse based on object detection ahead of time. Mid-flight online control is a future extension.

### Coordinate convention (matches the simulator)

| Axis | Role |
|------|------|
| x | Conveyor direction **and** baseline jet direction |
| y | Belt-width direction |
| z | Vertical |

- Baseline jet direction is **+x**: `azimuth_deg = 0, angle_deg = 0` → `e_jet = [1, 0, 0]`.
- `azimuth_deg` steers left/right in the x-y plane; `angle_deg` steers up/down (elevation) in the x-z plane.
- The jet is fired from a fixed nozzle at `(jet_x = 0.20, jet_y = 0.0, jet_z = 0.18)` m.

**Note on the project PDF convention.** The final project PDF mentions `y_land > y_c` as the sorting boundary, but this conflicts with the simulator's actual coordinate convention (which sorts in +x). This implementation follows the simulator: success is defined by `landing_x` against an x-interval. The mismatch with the PDF is a notation ambiguity, not a behaviour difference — we just rename "the lateral sorting coordinate" from `y` (PDF) to `x` (simulator).

### Sorting success criterion (boundary mode)

```
success = landing_x >= target_x_min   (target_x_min = 0.42 m)
```

**Why 0.42 m?** The no-jet evaluation shows:
- mean no-jet `landing_x` ≈ 0.355 m
- max  no-jet `landing_x` ≈ 0.411 m

So `target_x_min = 0.42 m` is just beyond the natural ballistic landing range. Any jet effect that pushes the object past this boundary counts as a successful sort.

`target_x_max = 0.65 m` is retained in `config.py` for reference and legacy `interval` mode, but it is **not** a hard failure boundary in boundary mode.

Calibrated from a fixed-action sweep over 200 random shapes so that the task is non-trivial:

| Action | Boundary success rate (`landing_x >= 0.42`) |
|--------|--------------------------------------------|
| no jet (U_min, D_min) | 0% |
| weak (U=14, D=0.03) | 0% |
| medium (U=17, D=0.03) | 0% |
| U_max, D=0.03 | ~8% |
| U_max, D_max (best fixed action) | ~36% |
| uniform random action | ~2% |

PPO has clear room to improve over the best fixed action.

### Landing position convention

`landing_position = COM position at the first timestep when the lowest surface point reaches landing_z`. Applied consistently for reward and evaluation. This is a simplification relative to the actual contact point.

### Observation space (30-dim float32)

| Feature | Dim |
|---------|-----|
| Object type one-hot (plate/rod/irregular) | 3 |
| Mass, drag coefficient | 2 |
| Size (x, y, z) | 3 |
| Inertia diagonal (Ixx, Iyy, Izz) | 3 |
| Reference area | 1 |
| Initial COM position (x, y, z) | 3 |
| Initial COM velocity (vx, vy, vz) | 3 |
| Initial angular velocity (ωx, ωy, ωz) | 3 |
| Initial quaternion (w, x, y, z) | 4 |
| Jet centre (xj, yj, zj) | 3 |
| target_x_min, target_x_max | 2 |

`VecNormalize` (running mean/std) is applied on top during training.

`target_x_max` remains in the observation for structural compatibility. In boundary mode, `vy = vz = 0` always (see initial velocity section below), so those observation slots are always zero.

### Initial velocity

```
vx ~ Uniform(0.80, 1.20) m/s    (conveyor speed, randomised)
vy = 0.0                          (no lateral belt velocity)
vz = 0.0                          (no vertical initial velocity)
```

Objects travel only in the **+x** (conveyor/sorting) direction at the start of each episode. The conveyor and the sorting direction are both x, so randomising vy or vz would add physically meaningless noise. Angular velocity remains randomised: `(ωx, ωy, ωz) ~ Uniform(-2, 2) rad/s`.

### Action modes

`RLConfig.action_mode` selects the PPO action space:

| Mode | Action shape | Normalized action |
|------|--------------|-------------------|
| `baseline` | `Box(-1, 1, shape=(3,))` | `[Umax, t_on_offset, duration]` |
| `elevation` | `Box(-1, 1, shape=(4,))` | `[Umax, t_on_offset, duration, elevation_angle]` |

The default mode is `elevation`. Use `--action-mode baseline` to train or evaluate the old 3-action policy. In baseline mode, `Jet3D.angle_deg` stays fixed at `jet_angle_deg = 0.0`, so the jet direction remains +x. In elevation mode, `Jet3D.azimuth_deg` remains fixed at `0.0`, and the fourth action controls elevation from `0.0` to `60.0` degrees. The lower bound is 0° (horizontal +x) since negative elevation is counterproductive for the boundary-crossing task; the upper bound is 60° to give the policy headroom for increasing flight time and crossing `target_x_min = 0.42 m`.

### Shared action mapping

| Action | Physical range | Mapping |
|--------|----------------|---------|
| `action[0]` → Umax | 10–30 m/s | log-scale |
| `action[1]` → t_on | nominal ± 0.1 s | linear |
| `action[2]` → duration | 0.01–0.10 s | log-scale |
| `action[3]` → elevation | 0–60 deg | linear, elevation mode only (`action[3]=-1`→0°, `0`→30°, `+1`→60°) |

`t_nominal` is the constant-vx estimate of when the object COM reaches `jet_x`. The agent's `action[1]` is an offset around it.

### Reward function (boundary mode)

```
if not has_landed (or non-finite trajectory):
    reward = -1.0

else:
    # Success: landing_x past the boundary
    success = landing_x >= target_x_min
    reward_success = 1.0 if success else 0.0

    # Undershoot penalty: proportional distance below target_x_min
    if landing_x < target_x_min:
        reward_undershoot = -(target_x_min - landing_x) / distance_scale
    else:
        reward_undershoot = 0.0

    # Soft overshoot penalty: only applies beyond overshoot_soft_start = 0.75 m
    # (no penalty between target_x_min and overshoot_soft_start)
    if landing_x > overshoot_soft_start:
        reward_overshoot = -overshoot_penalty_weight * (landing_x - overshoot_soft_start) / distance_scale
    else:
        reward_overshoot = 0.0

    umax_norm     = (umax - umax_min) / (umax_max - umax_min)
    duration_norm = (duration - duration_min) / (duration_max - duration_min)
    reward_energy = -umax_penalty_weight * umax_norm - duration_penalty_weight * duration_norm

    reward = reward_success + reward_undershoot + reward_overshoot + reward_energy
```

**Reward weights (defaults):**

| Weight | Value | Purpose |
|--------|-------|---------|
| `distance_scale` | 0.20 m | Divisor for undershoot/overshoot distance penalties |
| `overshoot_soft_start` | 0.75 m | Soft overshoot penalty only applies beyond this |
| `umax_penalty_weight` | 0.03 | Discourage unnecessarily high jet strength |
| `duration_penalty_weight` | 0.02 | Discourage unnecessarily long jet bursts |
| `overshoot_penalty_weight` | 0.50 | Weight for the soft overshoot penalty |

**No center bonus in boundary mode.** The task is no longer a bounded target-centering problem — any landing past `target_x_min` is a success. The center bonus is disabled (`reward_center = 0.0`) in boundary mode.

**Soft overshoot penalty.** The penalty only kicks in when `landing_x > 0.75 m`. The window `[0.42, 0.75]` is penalty-free, so the policy is free to overshoot within reason without extra cost. Extreme overshoots (> 0.75 m) incur a gentle penalty to discourage wasting energy on unnecessarily powerful jets.

**Backward-compatible keys.** `info["reward_distance"]` is an alias for `reward_undershoot`. `info["reward_center"]` is always `0.0` in boundary mode. Existing scripts reading these keys will still work.

### Seed splits

| Split | Seeds | Purpose |
|-------|-------|---------|
| Training | 0–999 | PPO update (random sampling) |
| Evaluation | 1000–1099 | Held-out generalisation test (each seed visited once, deterministic) |

---

## Configuration

All hyperparameters live in `rl/config.py` (`RLConfig` dataclass).
