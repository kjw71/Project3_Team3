"""
Week 2: 3D rigid-body air-jet sorting simulator.

Coordinate convention:
    x: sorting / air-jet baseline direction
    y: belt-width / lateral direction
    z: vertical direction

Main modeling assumptions:
    - The object is a rigid body represented by discrete surface points.
    - The air jet is a PDF-inspired directional Gaussian velocity field.
    - The air jet is modeled as a directional Gaussian plume.
    - The jet centerline passes through (x_center, y_center, z_center) and
      points in a 3D direction defined by elevation and azimuth angles.
    - The jet intensity decays in two ways:
        1) Gaussian decay with perpendicular distance from the jet centerline
        2) forward-only exponential decay along the jet centerline from the jet center
    - Local surface normals are used to weight the force received by each surface point.
    - Landing is detected at the first contact between the lowest surface point and
      the landing plane.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull


@dataclass
class Object3D:
    """3D rigid-body object represented by surface points."""
    name: str
    object_type: str
    mass: float
    drag_coefficient: float
    surface_points_body: np.ndarray
    surface_normals_body: np.ndarray
    area_weights: np.ndarray
    inertia_body: np.ndarray
    size_x: float
    size_y: float
    size_z: float
    rod_length: Optional[float] = None
    rod_radius: Optional[float] = None


@dataclass
class Jet3D:
    """
    Directional Gaussian air-jet plume model.

    Velocity field:
        u_jet(r, t) = Umax * (1 + epsilon(t))
                      * exp(-d_perp(r)^2 / (2*sigma^2))
                      * exp(-s(r) / axial_decay)
                      * e_jet

    where:
        e_jet = [
            cos(elevation) cos(azimuth),
            cos(elevation) sin(azimuth),
            sin(elevation)
        ]

        r0 = [x_center, y_center, z_center]
        d = r - r0
        s = d dot e_jet
        r_perp = d - s e_jet
        d_perp = ||r_perp||

    Interpretation:
        - s is the downstream distance from the nozzle along the jet direction.
        - d_perp is the perpendicular distance from the jet centerline.
        - upstream points (s < 0) receive no jet velocity.
        - sigma controls the radial Gaussian width.
        - axial_decay controls how quickly the jet weakens downstream.

    Angle convention:
        angle_deg = 0   -> +x sorting direction
        angle_deg = 45  -> +x/+z diagonal upward direction
        angle_deg = 90  -> +z vertical upward direction
    """
    umax: float = 25.0
    x_center: float = 0.00
    y_center: float = 0.00
    z_center: float = 0.20
    sigma: float = 0.08
    axial_decay: float = 0.35
    angle_deg: float = 45.0
    azimuth_deg: float = 0.0
    t_on: float = 0.0
    duration: float = 0.15
    noise_std: float = 0.0


@dataclass
class Simulation3D:
    dt: float = 0.001
    t_max: float = 3.0
    gravity: float = 9.81
    air_density: float = 1.225
    landing_z: float = 0.0
    conveyor_length: float = 0.15
    free_fall_start_offset: float = 0.03


@dataclass
class InitialCondition3D:
    position: Tuple[float, float, float] = (0.0, 0.0, 0.2)
    velocity: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    quaternion: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class TargetRegion3D:
    x_min: float = 0.30
    x_max: float = 0.80


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm_q = np.linalg.norm(q)
    if norm_q < 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / norm_q


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 x q2. Quaternion format: [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(q)
    w, x, y, z = q

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def euler_degrees_to_quaternion(
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> Tuple[float, float, float, float]:
    """
    Convert roll-pitch-yaw angles in degrees to quaternion [w, x, y, z].

    Roll  : rotation around x-axis
    Pitch : rotation around y-axis
    Yaw   : rotation around z-axis
    """
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)

    cr = np.cos(roll / 2.0)
    sr = np.sin(roll / 2.0)
    cp = np.cos(pitch / 2.0)
    sp = np.sin(pitch / 2.0)
    cy = np.cos(yaw / 2.0)
    sy = np.sin(yaw / 2.0)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    q = normalize_quaternion(np.array([w, x, y, z], dtype=float))
    return tuple(float(v) for v in q)


def update_quaternion(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    # Here omega is integrated in the world frame (see simulate_rigid_body_3d),
    # so use q_dot = 0.5 * [0, omega_world] ⊗ q.
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]], dtype=float)
    dqdt = 0.5 * quaternion_multiply(omega_quat, q)
    return normalize_quaternion(q + dqdt * dt)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    safe_norms = np.where(norms < 1.0e-12, 1.0, norms)
    return vectors / safe_norms[:, None]


def compute_box_inertia(mass: float, lx: float, ly: float, lz: float) -> np.ndarray:
    ixx = (1.0 / 12.0) * mass * (ly ** 2 + lz ** 2)
    iyy = (1.0 / 12.0) * mass * (lx ** 2 + lz ** 2)
    izz = (1.0 / 12.0) * mass * (lx ** 2 + ly ** 2)
    return np.diag([ixx, iyy, izz])


def compute_cylinder_inertia_x_axis(mass: float, length: float, radius: float) -> np.ndarray:
    """
    Solid cylinder inertia tensor aligned with body x-axis.

    Ixx = 1/2 m r^2
    Iyy = Izz = 1/12 m (3r^2 + L^2)
    """
    ixx = 0.5 * mass * radius ** 2
    iyy = (1.0 / 12.0) * mass * (3.0 * radius ** 2 + length ** 2)
    izz = iyy
    return np.diag([ixx, iyy, izz])


def create_surface_grid_plate(
    lx: float,
    ly: float,
    lz: float,
    nx: int = 7,
    ny: int = 7,
    nz: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(-lx / 2.0, lx / 2.0, nx)
    ys = np.linspace(-ly / 2.0, ly / 2.0, ny)
    zs = np.linspace(-lz / 2.0, lz / 2.0, nz)

    points = []
    normals = []
    area_weights = []

    for z, normal_z in [(-lz / 2.0, -1.0), (lz / 2.0, 1.0)]:
        for x in xs:
            for y in ys:
                points.append([x, y, z])
                normals.append([0.0, 0.0, normal_z])
                area_weights.append((lx * ly) / (nx * ny))

    for x, normal_x in [(-lx / 2.0, -1.0), (lx / 2.0, 1.0)]:
        for y in ys:
            for z in zs:
                points.append([x, y, z])
                normals.append([normal_x, 0.0, 0.0])
                area_weights.append((ly * lz) / (ny * nz))

    for y, normal_y in [(-ly / 2.0, -1.0), (ly / 2.0, 1.0)]:
        for x in xs:
            for z in zs:
                points.append([x, y, z])
                normals.append([0.0, normal_y, 0.0])
                area_weights.append((lx * lz) / (nx * nz))

    points = np.array(points, dtype=float)
    normals = np.array(normals, dtype=float)
    area_weights = np.array(area_weights, dtype=float)

    return points, normals, area_weights


def create_surface_grid_rod(
    length: float,
    radius: float,
    n_length: int = 17,
    n_theta: int = 16,
    n_cap_rings: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create representative surface points and normals for a cylindrical rod
    aligned with body x-axis.
    """
    xs = np.linspace(-length / 2.0, length / 2.0, n_length)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    points = []
    normals = []

    # Side surface
    for x in xs:
        for theta in thetas:
            y = radius * np.cos(theta)
            z = radius * np.sin(theta)
            points.append([x, y, z])
            normals.append([0.0, np.cos(theta), np.sin(theta)])

    # End caps
    cap_radii = np.linspace(0.0, radius, n_cap_rings)
    for x, normal_x in [(-length / 2.0, -1.0), (length / 2.0, 1.0)]:
        for rr in cap_radii:
            for theta in thetas:
                y = rr * np.cos(theta)
                z = rr * np.sin(theta)
                points.append([x, y, z])
                normals.append([normal_x, 0.0, 0.0])

    points = np.array(points, dtype=float)
    normals = np.array(normals, dtype=float)

    side_area = 2.0 * np.pi * radius * length
    cap_area = 2.0 * np.pi * radius ** 2
    total_area = side_area + cap_area
    area_weights = np.ones(len(points), dtype=float) * (total_area / len(points))

    return points, normals, area_weights


def create_irregular_flake_points(
    lx: float,
    ly: float,
    lz: float,
    n_points: int = 560,
    seed: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create representative surface samples, outward normals, and area weights
    for an irregular flake-like body.

    Workflow:
    1) Generate rough surface vertices on a star-shaped flattened envelope.
    2) Build a triangular convex hull from those vertices.
    3) Area-weight sample a fixed number of points on hull triangles.
    4) Use each sampled triangle normal as the sample normal.

    This keeps the normal-vector-based force model while making the number of
    surface samples deterministic for stable comparisons across runs.
    """
    rng = np.random.default_rng(seed)
    target_points = max(int(n_points), 24)
    n_vertices = max(180, target_points // 2)

    a = max(float(lx) / 2.0, 1.0e-5)
    b = max(float(ly) / 2.0, 1.0e-5)
    c = max(float(lz) / 2.0, 1.0e-5)

    # Sample directions on the unit sphere.
    directions = rng.normal(size=(n_vertices, 3))
    directions = normalize_vectors(directions)
    tiny = np.linalg.norm(directions, axis=1) < 1.0e-12
    if np.any(tiny):
        directions[tiny] = np.array([0.0, 0.0, 1.0], dtype=float)

    dx = directions[:, 0]
    dy = directions[:, 1]
    dz = directions[:, 2]

    # Intersect each direction ray with base ellipsoid:
    # (x/a)^2 + (y/b)^2 + (z/c)^2 = 1.
    denom = (dx / a) ** 2 + (dy / b) ** 2 + (dz / c) ** 2
    base_radius = 1.0 / np.sqrt(np.maximum(denom, 1.0e-12))

    # Add roughness so the surface is visibly irregular.
    low_freq = 0.18 * (0.7 * dx * dy - 0.5 * dy * dz + 0.4 * dx * dz)
    random_rough = 0.10 * rng.normal(size=n_vertices)
    rough_scale = np.clip(1.0 + low_freq + random_rough, 0.60, 1.45)
    radius = base_radius * rough_scale

    vertices = directions * radius[:, None]

    try:
        hull = ConvexHull(vertices, qhull_options="QJ")
        simplices = np.asarray(hull.simplices, dtype=int)

        tri_v0 = []
        tri_v1 = []
        tri_v2 = []
        tri_normals = []
        tri_areas = []

        for tri in simplices:
            v0 = vertices[tri[0]]
            v1 = vertices[tri[1]]
            v2 = vertices[tri[2]]

            edge1 = v1 - v0
            edge2 = v2 - v0
            normal_raw = np.cross(edge1, edge2)
            raw_norm = np.linalg.norm(normal_raw)
            if raw_norm < 1.0e-14:
                continue

            centroid = (v0 + v1 + v2) / 3.0
            normal = normal_raw / raw_norm
            if float(np.dot(normal, centroid)) < 0.0:
                normal = -normal

            area = 0.5 * raw_norm

            tri_v0.append(v0)
            tri_v1.append(v1)
            tri_v2.append(v2)
            tri_normals.append(normal)
            tri_areas.append(area)

        if len(tri_areas) >= 4:
            v0_arr = np.asarray(tri_v0, dtype=float)
            v1_arr = np.asarray(tri_v1, dtype=float)
            v2_arr = np.asarray(tri_v2, dtype=float)
            normals_tri = normalize_vectors(np.asarray(tri_normals, dtype=float))
            areas = np.maximum(np.asarray(tri_areas, dtype=float), 1.0e-12)
            total_area = float(np.sum(areas))
            probs = areas / total_area

            picked = rng.choice(len(areas), size=target_points, replace=True, p=probs)

            r1 = rng.random(target_points)
            r2 = rng.random(target_points)
            sqrt_r1 = np.sqrt(r1)
            w0 = 1.0 - sqrt_r1
            w1 = sqrt_r1 * (1.0 - r2)
            w2 = sqrt_r1 * r2

            points = (
                v0_arr[picked] * w0[:, None]
                + v1_arr[picked] * w1[:, None]
                + v2_arr[picked] * w2[:, None]
            )
            normals = normals_tri[picked]
            area_weights = np.ones(target_points, dtype=float) * (total_area / target_points)
            return points, normals, area_weights
    except Exception:
        pass

    # Fallback: rough surface points with radial normals and uniform weights.
    fallback_idx = rng.integers(0, vertices.shape[0], size=target_points)
    points = vertices[fallback_idx]
    normals = normalize_vectors(points.copy())
    near_center = np.linalg.norm(points, axis=1) < 1.0e-12
    if np.any(near_center):
        normals[near_center] = np.array([0.0, 0.0, 1.0], dtype=float)
    p = 1.6075
    ellipsoid_area = 4.0 * np.pi * (
        ((a ** p) * (b ** p) + (a ** p) * (c ** p) + (b ** p) * (c ** p)) / 3.0
    ) ** (1.0 / p)
    approx_area = float(ellipsoid_area * 1.08)
    area_weights = np.ones(target_points, dtype=float) * (approx_area / target_points)
    return points, normals, area_weights


def create_object_3d(
    object_type: str = "plate",
    mass: float = 0.05,
    size_x: float = 0.10,
    size_y: float = 0.10,
    size_z: float = 0.01,
    drag_coefficient: float = 1.0,
    rod_length: Optional[float] = None,
    rod_radius: Optional[float] = None,
    seed: int = 1,
) -> Object3D:
    object_type = object_type.lower()

    if object_type == "plate":
        points, normals, areas = create_surface_grid_plate(
            lx=size_x,
            ly=size_y,
            lz=size_z,
            nx=10,
            ny=10,
            nz=9,
        )
        inertia = compute_box_inertia(mass, size_x, size_y, size_z)

        return Object3D(
            name="thin plate",
            object_type="plate",
            mass=mass,
            drag_coefficient=drag_coefficient,
            surface_points_body=points,
            surface_normals_body=normals,
            area_weights=areas,
            inertia_body=inertia,
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
        )

    if object_type == "rod":
        if rod_length is None:
            rod_length = size_x
        if rod_radius is None:
            rod_radius = max(size_y, size_z) / 2.0

        size_x = rod_length
        size_y = 2.0 * rod_radius
        size_z = 2.0 * rod_radius

        points, normals, areas = create_surface_grid_rod(
            length=rod_length,
            radius=rod_radius,
            n_length=17,
            n_theta=16,
            n_cap_rings=9,
        )
        inertia = compute_cylinder_inertia_x_axis(mass, rod_length, rod_radius)

        return Object3D(
            name="cylindrical rod",
            object_type="rod",
            mass=mass,
            drag_coefficient=drag_coefficient,
            surface_points_body=points,
            surface_normals_body=normals,
            area_weights=areas,
            inertia_body=inertia,
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
            rod_length=rod_length,
            rod_radius=rod_radius,
        )

    if object_type == "irregular":
        points, normals, areas = create_irregular_flake_points(
            lx=size_x,
            ly=size_y,
            lz=size_z,
            n_points=560,
            seed=seed,
        )
        inertia = compute_box_inertia(mass, size_x, size_y, size_z)

        return Object3D(
            name="irregular flake",
            object_type="irregular",
            mass=mass,
            drag_coefficient=drag_coefficient,
            surface_points_body=points,
            surface_normals_body=normals,
            area_weights=areas,
            inertia_body=inertia,
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
        )

    raise ValueError(f"Unknown object_type: {object_type}")


def transform_surface_points(
    position: np.ndarray,
    quaternion: np.ndarray,
    points_body: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    r_vectors = points_body @ rotation_matrix.T
    points_world = position[None, :] + r_vectors
    return points_world, r_vectors, rotation_matrix


def transform_normals(
    quaternion: np.ndarray,
    normals_body: np.ndarray,
) -> np.ndarray:
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    normals_world = normals_body @ rotation_matrix.T
    return normalize_vectors(normals_world)


def get_jet_direction(jet: Jet3D) -> np.ndarray:
    """Return 3D angle-controlled jet direction.

    angle_deg is elevation measured upward from the x-y plane:
        0 deg  -> +x sorting direction
        45 deg -> +x/+z diagonal upward direction
        90 deg -> +z vertical direction

    azimuth_deg rotates the horizontal projection around z:
        0 deg  -> +x direction
        90 deg -> +y direction
    """
    elevation = np.deg2rad(jet.angle_deg)
    azimuth = np.deg2rad(getattr(jet, "azimuth_deg", 0.0))
    direction = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ],
        dtype=float,
    )
    norm_direction = np.linalg.norm(direction)

    if norm_direction < 1.0e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)

    return direction / norm_direction


def gaussian_jet_velocity(
    points_world: np.ndarray,
    t: float,
    jet: Jet3D,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Compute directional Gaussian air-jet velocity at surface points.

    The jet is modeled as a directional plume.

    For each point r:
        r0 = [x_center, y_center, z_center]
        e = jet direction
        d = r - r0
        s = d dot e
        r_perp = d - s e

    The velocity magnitude is:
        Umax * noise_factor
        * exp(-||r_perp||^2 / (2*sigma^2))
        * exp(-s / axial_decay), only for s >= 0
    """
    n_points = points_world.shape[0]
    velocities = np.zeros((n_points, 3), dtype=float)

    # Time window.
    if not (jet.t_on <= t <= jet.t_on + jet.duration):
        return velocities

    if jet.umax <= 0.0:
        return velocities

    sigma_eff = max(float(jet.sigma), 1.0e-6)
    axial_decay_eff = max(float(getattr(jet, "axial_decay", 0.35)), 1.0e-6)

    jet_direction = get_jet_direction(jet)
    centerline_point = np.array(
        [jet.x_center, jet.y_center, jet.z_center],
        dtype=float,
    )

    # Geometry relative to nozzle / jet center.
    rel = points_world - centerline_point[None, :]

    # Axial coordinate along jet direction.
    # Forward-only nozzle model: no jet contribution for s < 0.
    axial_distance = rel @ jet_direction

    # Perpendicular distance to the jet centerline.
    perpendicular = rel - axial_distance[:, None] * jet_direction[None, :]
    d_perp2 = np.sum(perpendicular * perpendicular, axis=1)

    radial_profile = np.exp(-d_perp2 / (2.0 * sigma_eff ** 2))
    forward_mask = axial_distance >= 0.0
    axial_profile = np.zeros_like(axial_distance, dtype=float)
    axial_profile[forward_mask] = np.exp(
        -axial_distance[forward_mask] / axial_decay_eff
    )

    profile = radial_profile * axial_profile

    noise_factor = 1.0
    if jet.noise_std > 0.0 and rng is not None:
        noise_factor = 1.0 + rng.normal(0.0, jet.noise_std)
        noise_factor = max(0.0, noise_factor)

    u_mag = jet.umax * noise_factor * profile
    velocities = u_mag[:, None] * jet_direction[None, :]

    return velocities


def compute_local_surface_velocity(
    velocity_com: np.ndarray,
    angular_velocity: np.ndarray,
    r_vectors: np.ndarray,
) -> np.ndarray:
    rotational_velocity = np.cross(angular_velocity[None, :], r_vectors)
    return velocity_com[None, :] + rotational_velocity


def compute_jet_forces_and_torque(
    obj: Object3D,
    position: np.ndarray,
    velocity: np.ndarray,
    quaternion: np.ndarray,
    angular_velocity: np.ndarray,
    t: float,
    jet: Jet3D,
    sim: Simulation3D,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    points_world, r_vectors, rotation_matrix = transform_surface_points(
        position=position,
        quaternion=quaternion,
        points_body=obj.surface_points_body,
    )

    normals_world = transform_normals(
        quaternion=quaternion,
        normals_body=obj.surface_normals_body,
    )

    u_jet = gaussian_jet_velocity(
        points_world=points_world,
        t=t,
        jet=jet,
        rng=rng,
    )

    v_surface = compute_local_surface_velocity(
        velocity_com=velocity,
        angular_velocity=angular_velocity,
        r_vectors=r_vectors,
    )

    u_rel = u_jet - v_surface
    speed_rel = np.linalg.norm(u_rel, axis=1)

    # IMPORTANT:
    # This function computes only air-jet-induced surface forces.
    # When the local jet velocity is zero, the local jet force must also be zero.
    # Otherwise, u_rel = -v_surface creates artificial surface drag and
    # incorrectly damps angular velocity after the jet turns off.
    jet_speed = np.linalg.norm(u_jet, axis=1)
    jet_active = jet_speed > 1.0e-12

    active = (speed_rel > 1.0e-12) & jet_active

    local_forces = np.zeros_like(u_rel)

    # Form-drag pressure acts along the exposed surface normal, so yawed
    # side faces can generate lateral force instead of only changing magnitude.
    normal_speed = np.sum(u_rel * normals_world, axis=1)
    incoming = active & (normal_speed < 0.0)

    local_forces[incoming] = (
        0.5
        * sim.air_density
        * obj.drag_coefficient
        * obj.area_weights[incoming, None]
        * normal_speed[incoming, None] ** 2
        * (-normals_world[incoming])
    )

    total_force = np.sum(local_forces, axis=0)
    local_torques = np.cross(r_vectors, local_forces)
    total_torque = np.sum(local_torques, axis=0)

    return {
        "points_world": points_world,
        "r_vectors": r_vectors,
        "normals_world": normals_world,
        "rotation_matrix": rotation_matrix,
        "u_jet": u_jet,
        "v_surface": v_surface,
        "u_rel": u_rel,
        "local_forces": local_forces,
        "total_force": total_force,
        "local_torques": local_torques,
        "total_torque": total_torque,
    }


def compute_body_drag_force(
    velocity: np.ndarray,
    obj: Object3D,
    sim: Simulation3D,
    reference_area: float,
) -> np.ndarray:
    """
    Simplified whole-body drag.

    This does not yet include orientation-dependent projected area.
    """
    speed = np.linalg.norm(velocity)

    if speed < 1.0e-12:
        return np.zeros(3, dtype=float)

    return (
        -0.5
        * sim.air_density
        * obj.drag_coefficient
        * reference_area
        * speed
        * velocity
    )


def simulate_rigid_body_3d(
    obj: Object3D,
    jet: Jet3D,
    sim: Simulation3D,
    initial: InitialCondition3D,
    target: Optional[TargetRegion3D] = None,
    reference_area: Optional[float] = None,
    seed: int = 1,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    position = np.array(initial.position, dtype=float)
    velocity = np.array(initial.velocity, dtype=float)
    quaternion = normalize_quaternion(np.array(initial.quaternion, dtype=float))
    angular_velocity = np.array(initial.angular_velocity, dtype=float)

    if reference_area is None:
        reference_area = float(np.sum(obj.area_weights) / 2.0)

    time_history = []
    position_history = []
    velocity_history = []
    quaternion_history = []
    angular_velocity_history = []

    force_gravity_history = []
    force_drag_history = []
    force_jet_history = []
    force_total_history = []
    torque_jet_history = []

    jet_impulse = np.zeros(3, dtype=float)
    angular_impulse = np.zeros(3, dtype=float)

    max_angular_speed = 0.0
    landing_position = None
    landing_time = None
    support_z = float(initial.position[2])
    release_x = float(
        initial.position[0]
        + max(float(sim.conveyor_length), 0.0)
        + max(float(sim.free_fall_start_offset), 0.0)
    )
    phase_history = []

    n_steps = int(sim.t_max / sim.dt) + 1

    for step in range(n_steps):
        t = step * sim.dt
        on_conveyor = position[0] < release_x
        phase = "conveyor" if on_conveyor else "free_fall"

        force_gravity = np.array([0.0, 0.0, -obj.mass * sim.gravity], dtype=float)

        jet_data = compute_jet_forces_and_torque(
            obj=obj,
            position=position,
            velocity=velocity,
            quaternion=quaternion,
            angular_velocity=angular_velocity,
            t=t,
            jet=jet,
            sim=sim,
            rng=rng,
        )

        force_jet = jet_data["total_force"]
        torque_jet = jet_data["total_torque"]

        force_drag = compute_body_drag_force(
            velocity=velocity,
            obj=obj,
            sim=sim,
            reference_area=reference_area,
        )

        force_total = force_gravity + force_drag + force_jet
        if on_conveyor:
            # Conveyor support cancels vertical acceleration before release.
            force_total[2] = 0.0

        time_history.append(t)
        position_history.append(position.copy())
        velocity_history.append(velocity.copy())
        quaternion_history.append(quaternion.copy())
        angular_velocity_history.append(angular_velocity.copy())
        phase_history.append(phase)

        force_gravity_history.append(force_gravity.copy())
        force_drag_history.append(force_drag.copy())
        force_jet_history.append(force_jet.copy())
        force_total_history.append(force_total.copy())
        torque_jet_history.append(torque_jet.copy())

        jet_impulse += force_jet * sim.dt
        angular_impulse += torque_jet * sim.dt

        angular_speed = np.linalg.norm(angular_velocity)
        max_angular_speed = max(max_angular_speed, angular_speed)

        points_world_for_landing, _, _ = transform_surface_points(
            position=position,
            quaternion=quaternion,
            points_body=obj.surface_points_body,
        )

        lowest_surface_z = np.min(points_world_for_landing[:, 2])

        if (not on_conveyor) and lowest_surface_z <= sim.landing_z and step > 0:
            landing_position = position.copy()
            landing_time = t
            break

        acceleration = force_total / obj.mass

        velocity = velocity + acceleration * sim.dt
        position = position + velocity * sim.dt
        if on_conveyor:
            velocity[2] = 0.0
            position[2] = support_z

        rotation_matrix = quaternion_to_rotation_matrix(quaternion)
        inertia_world = rotation_matrix @ obj.inertia_body @ rotation_matrix.T
        inertia_world_inv = np.linalg.pinv(inertia_world)

        gyroscopic_term = np.cross(
            angular_velocity,
            inertia_world @ angular_velocity,
        )

        angular_acceleration = inertia_world_inv @ (torque_jet - gyroscopic_term)

        angular_velocity = angular_velocity + angular_acceleration * sim.dt
        quaternion = update_quaternion(quaternion, angular_velocity, sim.dt)

    time_array = np.array(time_history)
    position_array = np.array(position_history)
    velocity_array = np.array(velocity_history)
    quaternion_array = np.array(quaternion_history)
    angular_velocity_array = np.array(angular_velocity_history)

    has_landed = landing_position is not None

    success = None
    if target is not None and has_landed:
        success = target.x_min <= landing_position[0] <= target.x_max

    final_points_world, _, _ = transform_surface_points(
        position=position_array[-1],
        quaternion=quaternion_array[-1],
        points_body=obj.surface_points_body,
    )

    return {
        "time": time_array,
        "position": position_array,
        "velocity": velocity_array,
        "quaternion": quaternion_array,
        "angular_velocity": angular_velocity_array,
        "force_gravity": np.array(force_gravity_history),
        "force_drag": np.array(force_drag_history),
        "force_jet": np.array(force_jet_history),
        "force_total": np.array(force_total_history),
        "torque_jet": np.array(torque_jet_history),
        "phase": np.array(phase_history, dtype=object),
        "landing_position": landing_position,
        "landing_time": landing_time,
        "has_landed": has_landed,
        "final_time": time_array[-1],
        "final_position": position_array[-1],
        "success": success,
        "jet_impulse": jet_impulse,
        "angular_impulse": angular_impulse,
        "max_angular_speed": max_angular_speed,
        "final_points_world": final_points_world,
        "object": obj,
        "jet": jet,
        "simulation": sim,
        "support_release_x": release_x,
        "conveyor_surface_z": support_z,
        "initial": initial,
        "target": target,
        "reference_area": reference_area,
    }


def compute_hit_offset(initial_position: Tuple[float, float, float], jet: Jet3D) -> float:
    """Perpendicular distance from initial COM to the jet centerline.

    This is the radial hit offset only. It does not include the downstream
    axial decay factor. Use compute_jet_relative_coordinates() if axial
    distance and combined jet influence are also needed.
    """
    point = np.asarray(initial_position, dtype=float)
    centerline_point = np.array([jet.x_center, jet.y_center, jet.z_center], dtype=float)
    direction = get_jet_direction(jet)

    rel = point - centerline_point
    axial_distance = float(np.dot(rel, direction))
    perpendicular = rel - axial_distance * direction
    return float(np.linalg.norm(perpendicular))


def compute_jet_relative_coordinates(
    point: Tuple[float, float, float],
    jet: Jet3D,
) -> Dict[str, float]:
    """Compute axial and perpendicular coordinates relative to the jet plume.

    Returns:
        axial_distance_m:
            s = (point - jet_center) dot e_jet.

        perpendicular_distance_m:
            distance from the point to the jet centerline.

        radial_profile:
            exp(-d_perp^2 / (2*sigma^2))

        axial_profile:
            exp(-s / axial_decay) for s >= 0, and 0 for s < 0.

        combined_profile:
            radial_profile * axial_profile.
    """
    point_array = np.asarray(point, dtype=float)
    centerline_point = np.array(
        [jet.x_center, jet.y_center, jet.z_center],
        dtype=float,
    )
    direction = get_jet_direction(jet)

    rel = point_array - centerline_point
    axial_distance = float(np.dot(rel, direction))
    perpendicular = rel - axial_distance * direction
    perpendicular_distance = float(np.linalg.norm(perpendicular))

    sigma_eff = max(float(jet.sigma), 1.0e-6)
    axial_decay_eff = max(float(getattr(jet, "axial_decay", 0.35)), 1.0e-6)

    radial_profile = float(
        np.exp(-(perpendicular_distance ** 2) / (2.0 * sigma_eff ** 2))
    )

    axial_profile = (
        float(np.exp(-axial_distance / axial_decay_eff))
        if axial_distance >= 0.0
        else 0.0
    )

    combined_profile = radial_profile * axial_profile

    return {
        "axial_distance_m": axial_distance,
        "perpendicular_distance_m": perpendicular_distance,
        "radial_profile": radial_profile,
        "axial_profile": axial_profile,
        "combined_profile": combined_profile,
    }
