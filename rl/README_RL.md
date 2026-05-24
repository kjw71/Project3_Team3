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
```

Each unseen seed in `[1000, 1099]` is evaluated **exactly once** in deterministic order.

Output: `outputs/rl_results/evaluation_results.csv`
Columns: `episode, seed, object_type, success, has_landed, landing_x, landing_y, landing_z, reward, umax, t_on, duration, reward_success, reward_distance, reward_energy`

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

### Sorting success criterion

```
success = target_x_min <= landing_x <= target_x_max
```

Default: `target_x_min = 0.42 m`, `target_x_max = 0.65 m`.

Calibrated from a fixed-action sweep over 200 random shapes so that the task is non-trivial:

| Action | Hit rate on `[0.42, 0.65]` |
|--------|--------------------------|
| no jet (U_min, D_min) | 0% |
| weak (U=14, D=0.03) | 0% |
| medium (U=17, D=0.03) | 0% |
| U_max, D=0.03 | 8% |
| U_max, D_max (best fixed action) | **36%** |
| uniform random action | 2% |

So PPO has clear room to improve over the best fixed action.

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

### Action space (3-dim Box in [-1, 1])

| Action | Physical range | Mapping |
|--------|----------------|---------|
| `action[0]` → Umax | 10–30 m/s | log-scale |
| `action[1]` → t_on | nominal ± 0.1 s | linear |
| `action[2]` → duration | 0.01–0.10 s | log-scale |

`t_nominal` is the constant-vx estimate of when the object COM reaches `jet_x`. The agent's `action[1]` is an offset around it.

### Reward function

```
if not has_landed (or non-finite trajectory):
    reward = -1.0

else:
    success_bonus    = +1.0  if target_x_min <= landing_x <= target_x_max else 0.0

    if landing_x < target_x_min:   distance = target_x_min - landing_x
    elif landing_x > target_x_max: distance = landing_x - target_x_max
    else:                          distance = 0.0
    distance_penalty = -distance / 0.20

    energy_penalty   = -0.02 * (umax_norm + duration_norm)
    reward = success_bonus + distance_penalty + energy_penalty
```

### Seed splits

| Split | Seeds | Purpose |
|-------|-------|---------|
| Training | 0–999 | PPO update (random sampling) |
| Evaluation | 1000–1099 | Held-out generalisation test (each seed visited once, deterministic) |

---

## Configuration

All hyperparameters live in `rl/config.py` (`RLConfig` dataclass).
