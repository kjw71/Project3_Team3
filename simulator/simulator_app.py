"""
Interactive Week 2 3D Rigid-Body Air-Jet Simulator using Streamlit.

Run from the project root:

    streamlit run simulator/simulator_app.py
"""

from pathlib import Path
import sys
import json
import tempfile
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch, Rectangle, Circle, Polygon
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from source.core_3d import (  # noqa: E402
    Jet3D,
    Simulation3D,
    InitialCondition3D,
    TargetRegion3D,
    create_object_3d,
    simulate_rigid_body_3d,
    compute_hit_offset,
    transform_surface_points,
    euler_degrees_to_quaternion,
)


# ---------------------------------------------------------------------
# Default values and reset utilities
# ---------------------------------------------------------------------

DEFAULTS = {
    # Object
    "object_type": "plate",
    "mass": 0.050,
    "drag_coefficient": 1.0,
    "size_x": 0.10,
    "size_y": 0.10,
    "size_z": 0.01,
    "rod_length": 0.15,
    "rod_radius": 0.025,

    # Initial motion
    "x0": 0.0,
    "y0": 0.0,
    "z0": 0.20,
    "vc": 1.0,
    "vy_initial": 0.0,
    "vz_initial": 0.0,

    # Initial orientation
    "roll0": 0.0,
    "pitch0": 0.0,
    "yaw0": 0.0,

    # Initial angular velocity
    "omega_x0": 0.0,
    "omega_y0": 0.0,
    "omega_z0": 0.0,

    # Air jet
    "umax": 25.0,
    "jet_x_center": 0.10,
    "jet_y_center": 0.00,
    "jet_z_center": 0.12,
    "sigma": 0.08,
    "axial_decay": 0.35,
    "jet_angle_deg": 45.0,
    "jet_azimuth_deg": 0.0,
    "jet_t_on": 0.12,
    "jet_duration": 0.15,
    "noise_std": 0.00,

    # Simulation
    "dt": 0.0005,
    "t_max": 2.0,
    "gravity": 9.81,
    "air_density": 1.225,
    "landing_z": 0.0,
    "conveyor_length": 0.15,
    "free_fall_start_offset": 0.03,

    # Target
    "target_x_min": 0.30,
    "target_x_max": 0.80,

    # Plot axes
    "use_fixed_axes": False,
    "x_plot_min": -0.10,
    "x_plot_max": 1.50,
    "y_plot_min": -0.50,
    "y_plot_max": 0.50,
    "z_plot_min": 0.00,
    "z_plot_max": 0.60,

    # Options
    "seed": 1,
    "auto_run_simulation": False,

    # Animation
    "animation_max_frames": 80,
    "animation_fps": 12,
    "animation_dpi": 100,
}

# Keep analysis plots compact and readable in narrow panels.
ANALYSIS_FIGSIZE = (5.4, 2.8)
ANALYSIS_LEGEND_FONTSIZE = 7

OBJECT_TYPE_ICON_PATHS = {
    "plate": PROJECT_ROOT / "assets" / "icons" / "plate.png",
    "rod": PROJECT_ROOT / "assets" / "icons" / "rod.png",
    "irregular": PROJECT_ROOT / "assets" / "icons" / "irregular.png",
}


def render_icon_image(container, image_bytes):
    """Render icon image with Streamlit version compatibility."""
    try:
        container.image(image_bytes, use_container_width=True)
    except TypeError:
        try:
            container.image(image_bytes, use_column_width=True)
        except TypeError:
            container.image(image_bytes, width=96)


@st.cache_data(show_spinner=False)
def make_object_type_icon_png(object_type, selected=False):
    """Compose object-type icon card from provided PNGs."""
    icon_path = OBJECT_TYPE_ICON_PATHS.get(str(object_type))
    if icon_path is not None and icon_path.exists():
        card_size = 220
        card_margin = 10
        icon_padding = 16
        border_width = 6 if selected else 3
        border_color = (37, 99, 235, 255) if selected else (148, 163, 184, 255)
        bg_color = (248, 250, 252, 255) if selected else (255, 255, 255, 255)

        card = Image.new("RGBA", (card_size, card_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(card)
        draw.rounded_rectangle(
            [card_margin, card_margin, card_size - card_margin, card_size - card_margin],
            radius=28,
            fill=bg_color,
            outline=border_color,
            width=border_width,
        )

        icon = Image.open(icon_path).convert("RGBA")
        inner_size = card_size - 2 * (card_margin + icon_padding)
        inner_size = max(inner_size, 24)
        ratio = min(inner_size / max(icon.width, 1), inner_size / max(icon.height, 1))
        resized = icon.resize(
            (max(1, int(icon.width * ratio)), max(1, int(icon.height * ratio))),
            Image.Resampling.LANCZOS,
        )
        x0 = (card_size - resized.width) // 2
        y0 = (card_size - resized.height) // 2
        card.alpha_composite(resized, (x0, y0))

        buff = io.BytesIO()
        card.save(buff, format="PNG")
        return buff.getvalue()

    # Fallback icon
    fig, ax = plt.subplots(figsize=(1.8, 1.8), dpi=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    face = "#e2e8f0" if selected else "#f8fafc"
    edge = "#2563eb" if selected else "#94a3b8"
    lw = 2.6 if selected else 1.6
    ax.add_patch(Rectangle((0.05, 0.05), 0.90, 0.90, facecolor=face, edgecolor=edge, linewidth=lw))
    if object_type == "plate":
        ax.add_patch(Rectangle((0.20, 0.36), 0.60, 0.28, facecolor="#0ea5e9", edgecolor="#0369a1", linewidth=1.8))
    elif object_type == "rod":
        ax.add_patch(Circle((0.5, 0.5), 0.25, facecolor="#fde68a", edgecolor="#059669", linewidth=2.2))
    else:
        poly = np.array([[0.18, 0.32], [0.42, 0.76], [0.72, 0.64], [0.82, 0.30], [0.48, 0.22]])
        ax.add_patch(Polygon(poly, closed=True, facecolor="#facc15", edgecolor="#84cc16", linewidth=1.8))
    out = io.BytesIO()
    fig.savefig(out, format="png", transparent=True, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    return out.getvalue()


SECTION_KEYS = {
    "object": [
        "object_type",
        "mass",
        "drag_coefficient",
        "size_x",
        "size_y",
        "size_z",
        "rod_length",
        "rod_radius",
    ],
    "initial_motion": [
        "x0",
        "y0",
        "z0",
        "vc",
        "vy_initial",
        "vz_initial",
    ],
    "initial_orientation": [
        "roll0",
        "pitch0",
        "yaw0",
    ],
    "initial_angular_velocity": [
        "omega_x0",
        "omega_y0",
        "omega_z0",
    ],
    "air_jet": [
        "umax",
        "jet_x_center",
        "jet_y_center",
        "jet_z_center",
        "sigma",
        "axial_decay",
        "jet_angle_deg",
        "jet_azimuth_deg",
        "jet_t_on",
        "jet_duration",
        "noise_std",
    ],
    "simulation": [
        "dt",
        "t_max",
        "gravity",
        "air_density",
        "landing_z",
        "conveyor_length",
        "free_fall_start_offset",
    ],
    "target": [
        "target_x_min",
        "target_x_max",
    ],
    "plot_axes": [
        "use_fixed_axes",
        "x_plot_min",
        "x_plot_max",
        "y_plot_min",
        "y_plot_max",
        "z_plot_min",
        "z_plot_max",
    ],
    "options": [
        "seed",
        "auto_run_simulation",
        "animation_max_frames",
        "animation_fps",
        "animation_dpi",
    ],
}


def initialize_session_defaults():
    """Initialize Streamlit session state with default values."""
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_keys(keys):
    """Reset selected session-state keys to their default values."""
    for key in keys:
        st.session_state[key] = DEFAULTS[key]


def reset_all():
    """Reset all user-controlled parameters and clear stored results."""
    for key, value in DEFAULTS.items():
        st.session_state[key] = value

    for key in ["last_result", "last_parameter_json", "last_csv_data", "last_gif_bytes"]:
        if key in st.session_state:
            del st.session_state[key]


initialize_session_defaults()


# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Week 2 3D Rigid-Body Air-Jet Simulator",
    layout="wide",
)

st.title("Week 2: Interactive 3D Rigid-Body Air-Jet Simulator")

st.write(
    """
    This app simulates a 3D rigid object moving in the conveyor direction and being hit by
    a finite-duration directional Gaussian air jet. The object is represented by surface
    points, so the jet can generate both total force and torque.
    """
)

st.info(
    "Coordinate convention: x = conveyor / main jet direction at 0 deg, "
    "y = lateral belt-width direction, z = vertical direction. "
    "The target is defined by landing x-position. "
    "The air jet uses a directional Gaussian profile with radial Gaussian decay "
    "around the centerline and forward-only axial decay from the jet centerline."
)

main_run_button = st.button(
    "Run 3D Simulation",
    type="primary",
    key="main_run_button",
)


# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------

st.sidebar.header("Run")

sidebar_run_button = st.sidebar.button(
    "Run 3D Simulation",
    type="primary",
    key="sidebar_run_button",
)

st.sidebar.button(
    "Reset All Parameters",
    key="reset_all_parameters_button",
    on_click=reset_all,
)

st.sidebar.caption(
    "Adjust the parameters below, then click Run. "
    "Section reset buttons return only that section to its default values."
)

st.sidebar.divider()


# ---------------------------------------------------------------------
# Object parameters
# ---------------------------------------------------------------------

st.sidebar.header("Object Parameters")

st.sidebar.button(
    "Reset Object Parameters",
    key="reset_object_button",
    on_click=reset_keys,
    args=(SECTION_KEYS["object"],),
)

st.sidebar.caption("Pick object type")
object_options = [
    ("plate", "Plate"),
    ("rod", "Rod"),
    ("irregular", "Irregular"),
]
selected_object_type = st.session_state.get("object_type", "plate")
icon_cols = st.sidebar.columns(3)
for col, (option_value, option_label) in zip(icon_cols, object_options):
    icon_png = make_object_type_icon_png(
        object_type=option_value,
        selected=(selected_object_type == option_value),
    )
    render_icon_image(col, icon_png)
    if col.button(
        option_label,
        key=f"object_type_pick_{option_value}",
        type="primary" if selected_object_type == option_value else "secondary",
        use_container_width=True,
    ):
        st.session_state["object_type"] = option_value
        selected_object_type = option_value
object_type = selected_object_type

mass = st.sidebar.slider(
    "Mass m [kg]",
    min_value=0.005,
    max_value=0.300,
    step=0.005,
    format="%.3f",
    key="mass",
)

drag_coefficient = st.sidebar.slider(
    "Effective drag coefficient Cd [-]",
    min_value=0.0,
    max_value=3.0,
    step=0.05,
    key="drag_coefficient",
)

rod_length = None
rod_radius = None

if object_type == "rod":
    rod_length = st.sidebar.slider(
        "Rod length [m]",
        min_value=0.02,
        max_value=0.50,
        step=0.005,
        format="%.3f",
        key="rod_length",
    )

    rod_radius = st.sidebar.slider(
        "Rod radius [m]",
        min_value=0.002,
        max_value=0.10,
        step=0.001,
        format="%.3f",
        key="rod_radius",
    )

    size_x = rod_length
    size_y = 2.0 * rod_radius
    size_z = 2.0 * rod_radius

    st.sidebar.caption(
        "For rod, the cylinder axis is aligned with the body x-axis. "
        "The app internally uses size_x = length and size_y = size_z = 2*radius."
    )

else:
    size_x = st.sidebar.slider(
        "Object size in x [m]",
        min_value=0.02,
        max_value=0.30,
        step=0.005,
        format="%.3f",
        key="size_x",
    )

    size_y = st.sidebar.slider(
        "Object size in y [m]",
        min_value=0.005,
        max_value=0.30,
        step=0.005,
        format="%.3f",
        key="size_y",
    )

    size_z = st.sidebar.slider(
        "Object size in z [m]",
        min_value=0.002,
        max_value=0.10,
        step=0.002,
        format="%.3f",
        key="size_z",
    )


# ---------------------------------------------------------------------
# Initial motion
# ---------------------------------------------------------------------

st.sidebar.header("Initial Motion")

st.sidebar.button(
    "Reset Initial Motion",
    key="reset_initial_motion_button",
    on_click=reset_keys,
    args=(SECTION_KEYS["initial_motion"],),
)

x0 = st.sidebar.slider(
    "Initial COM x [m]",
    min_value=-0.5,
    max_value=1.5,
    step=0.01,
    key="x0",
)

y0 = st.sidebar.slider(
    "Initial COM y [m]",
    min_value=-0.5,
    max_value=0.5,
    step=0.01,
    key="y0",
)

z0 = st.sidebar.slider(
    "Initial COM z [m]",
    min_value=0.02,
    max_value=1.0,
    step=0.01,
    key="z0",
)

vc = st.sidebar.slider(
    "Conveyor speed vx [m/s]",
    min_value=0.0,
    max_value=5.0,
    step=0.05,
    key="vc",
)

vy_initial = st.sidebar.slider(
    "Initial belt-width velocity vy [m/s]",
    min_value=-2.0,
    max_value=2.0,
    step=0.05,
    key="vy_initial",
)

vz_initial = st.sidebar.slider(
    "Initial vertical velocity vz [m/s]",
    min_value=-2.0,
    max_value=2.0,
    step=0.05,
    key="vz_initial",
)


# ---------------------------------------------------------------------
# Initial orientation
# ---------------------------------------------------------------------

st.sidebar.header("Initial Orientation")

st.sidebar.button(
    "Reset Initial Orientation",
    key="reset_initial_orientation_button",
    on_click=reset_keys,
    args=(SECTION_KEYS["initial_orientation"],),
)

roll0 = st.sidebar.slider(
    "Initial roll angle [deg]",
    min_value=-180.0,
    max_value=180.0,
    step=1.0,
    key="roll0",
)

pitch0 = st.sidebar.slider(
    "Initial pitch angle [deg]",
    min_value=-180.0,
    max_value=180.0,
    step=1.0,
    key="pitch0",
)

yaw0 = st.sidebar.slider(
    "Initial yaw angle [deg]",
    min_value=-180.0,
    max_value=180.0,
    step=1.0,
    key="yaw0",
)

st.sidebar.caption(
    "Roll = rotation around x, pitch = rotation around y, yaw = rotation around z."
)


# ---------------------------------------------------------------------
# Initial angular velocity
# ---------------------------------------------------------------------

st.sidebar.header("Initial Angular Velocity")

st.sidebar.button(
    "Reset Initial Angular Velocity",
    key="reset_initial_angular_velocity_button",
    on_click=reset_keys,
    args=(SECTION_KEYS["initial_angular_velocity"],),
)

omega_x0 = st.sidebar.slider(
    "Initial omega_x [rad/s]",
    min_value=-50.0,
    max_value=50.0,
    step=1.0,
    key="omega_x0",
)

omega_y0 = st.sidebar.slider(
    "Initial omega_y [rad/s]",
    min_value=-50.0,
    max_value=50.0,
    step=1.0,
    key="omega_y0",
)

omega_z0 = st.sidebar.slider(
    "Initial omega_z [rad/s]",
    min_value=-50.0,
    max_value=50.0,
    step=1.0,
    key="omega_z0",
)


# ---------------------------------------------------------------------
# Air jet
# ---------------------------------------------------------------------

st.sidebar.header("Directional Gaussian Air Jet")

st.sidebar.button(
    "Reset Air-Jet Parameters",
    key="reset_air_jet_button",
    on_click=reset_keys,
    args=(SECTION_KEYS["air_jet"],),
)

umax = st.sidebar.slider(
    "Jet maximum velocity Umax [m/s]",
    min_value=0.0,
    max_value=80.0,
    step=1.0,
    key="umax",
)

jet_x_center = st.sidebar.slider(
    "Jet center xj [m]",
    min_value=-0.5,
    max_value=2.0,
    step=0.01,
    key="jet_x_center",
)

jet_y_center = st.sidebar.slider(
    "Jet center yj [m]",
    min_value=-0.5,
    max_value=0.5,
    step=0.01,
    key="jet_y_center",
)

jet_z_center = st.sidebar.slider(
    "Jet center zj [m]",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    key="jet_z_center",
)

sigma = st.sidebar.slider(
    "Radial Gaussian width sigma [m]",
    min_value=0.01,
    max_value=0.30,
    step=0.005,
    format="%.3f",
    key="sigma",
)

axial_decay = st.sidebar.slider(
    "Axial decay length lambda [m]",
    min_value=0.02,
    max_value=1.50,
    step=0.005,
    format="%.3f",
    key="axial_decay",
)

jet_angle_deg = st.sidebar.slider(
    "Jet angle relative to +x [deg]",
    min_value=-45.0,
    max_value=90.0,
    step=1.0,
    key="jet_angle_deg",
)

jet_azimuth_deg = st.sidebar.slider(
    "Jet azimuth around +z [deg]",
    min_value=-180.0,
    max_value=180.0,
    step=1.0,
    key="jet_azimuth_deg",
)

st.sidebar.caption(
    "The jet centerline passes through (xj, yj, zj). "
    "Angle guide: elevation 0 deg = horizontal, 45 deg = diagonal upward, "
    "90 deg = vertical upward. Azimuth rotates the horizontal projection around +z. "
    "The jet uses a radial Gaussian profile around "
    "the centerline and a forward-only axial decay from the jet center. "
    "The velocity magnitude is hard-cut to zero for s < 0."
)

jet_t_on = st.sidebar.slider(
    "Jet activation time t_on [s]",
    min_value=0.0,
    max_value=2.0,
    step=0.01,
    key="jet_t_on",
)

jet_duration = st.sidebar.slider(
    "Jet duration dt_jet [s]",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    key="jet_duration",
)

noise_std = st.sidebar.slider(
    "Jet noise std [-]",
    min_value=0.0,
    max_value=0.50,
    step=0.01,
    key="noise_std",
)


# ---------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------

st.sidebar.header("Simulation Parameters")

st.sidebar.button(
    "Reset Simulation Parameters",
    key="reset_simulation_button",
    on_click=reset_keys,
    args=(SECTION_KEYS["simulation"],),
)

dt = st.sidebar.slider(
    "Time step dt [s]",
    min_value=0.00005,
    max_value=0.00100,
    step=0.00005,
    format="%.5f",
    key="dt",
)
st.sidebar.caption(
    "dt is the integration time step. Smaller dt usually improves numerical accuracy "
    "but increases runtime."
)

t_max = st.sidebar.slider(
    "Maximum simulation time [s]",
    min_value=0.5,
    max_value=5.0,
    step=0.1,
    key="t_max",
)

gravity = st.sidebar.slider(
    "Gravity g [m/s2]",
    min_value=0.0,
    max_value=20.0,
    step=0.01,
    key="gravity",
)

air_density = st.sidebar.slider(
    "Air density rho [kg/m3]",
    min_value=0.0,
    max_value=2.0,
    step=0.005,
    key="air_density",
)

landing_z = st.sidebar.slider(
    "Landing plane z [m]",
    min_value=-0.5,
    max_value=0.5,
    step=0.01,
    key="landing_z",
)

conveyor_length = st.sidebar.slider(
    "Conveyor support length [m]",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    key="conveyor_length",
)

free_fall_start_offset = st.sidebar.slider(
    "Free-fall start offset after conveyor [m]",
    min_value=0.0,
    max_value=0.50,
    step=0.005,
    format="%.3f",
    key="free_fall_start_offset",
)


# ---------------------------------------------------------------------
# Target region
# ---------------------------------------------------------------------

st.sidebar.header("Target Region")

st.sidebar.button(
    "Reset Target Region",
    key="reset_target_button",
    on_click=reset_keys,
    args=(SECTION_KEYS["target"],),
)

target_x_min = st.sidebar.slider(
    "Target x_min [m]",
    min_value=-1.0,
    max_value=5.0,
    step=0.05,
    key="target_x_min",
)

target_x_max = st.sidebar.slider(
    "Target x_max [m]",
    min_value=-1.0,
    max_value=5.0,
    step=0.05,
    key="target_x_max",
)

if target_x_max < target_x_min:
    st.sidebar.warning("Target x_max should be larger than x_min.")


# ---------------------------------------------------------------------
# Plot axis limits
# ---------------------------------------------------------------------

st.sidebar.header("Plot Axis Limits")

st.sidebar.button(
    "Reset Plot Axis Limits",
    key="reset_plot_axes_button",
    on_click=reset_keys,
    args=(SECTION_KEYS["plot_axes"],),
)

use_fixed_axes = st.sidebar.checkbox(
    "Use fixed plot axes",
    key="use_fixed_axes",
)

x_plot_min = st.sidebar.number_input(
    "Plot x_min [m]",
    step=0.05,
    key="x_plot_min",
)

x_plot_max = st.sidebar.number_input(
    "Plot x_max [m]",
    step=0.05,
    key="x_plot_max",
)

y_plot_min = st.sidebar.number_input(
    "Plot y_min [m]",
    step=0.05,
    key="y_plot_min",
)

y_plot_max = st.sidebar.number_input(
    "Plot y_max [m]",
    step=0.05,
    key="y_plot_max",
)

z_plot_min = st.sidebar.number_input(
    "Plot z_min [m]",
    step=0.05,
    key="z_plot_min",
)

z_plot_max = st.sidebar.number_input(
    "Plot z_max [m]",
    step=0.05,
    key="z_plot_max",
)

axis_limits = None

if use_fixed_axes:
    axis_limits = {
        "x": (x_plot_min, x_plot_max),
        "y": (y_plot_min, y_plot_max),
        "z": (z_plot_min, z_plot_max),
    }

    if x_plot_max <= x_plot_min:
        st.sidebar.warning("Plot x_max should be larger than x_min.")
    if y_plot_max <= y_plot_min:
        st.sidebar.warning("Plot y_max should be larger than y_min.")
    if z_plot_max <= z_plot_min:
        st.sidebar.warning("Plot z_max should be larger than z_min.")


# ---------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------

st.sidebar.header("Options")

st.sidebar.button(
    "Reset Options",
    key="reset_options_button",
    on_click=reset_keys,
    args=(SECTION_KEYS["options"],),
)

seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=9999,
    step=1,
    key="seed",
)

show_surface_points = False

auto_run_simulation = st.sidebar.checkbox(
    "Auto run on parameter change",
    key="auto_run_simulation",
)

st.sidebar.subheader("Animation Export")

animation_max_frames = st.sidebar.slider(
    "Maximum GIF frames",
    min_value=20,
    max_value=160,
    step=10,
    key="animation_max_frames",
)

animation_fps = st.sidebar.slider(
    "GIF FPS",
    min_value=5,
    max_value=30,
    step=1,
    key="animation_fps",
)

animation_dpi = st.sidebar.slider(
    "GIF DPI",
    min_value=60,
    max_value=160,
    step=10,
    key="animation_dpi",
)

run_button = main_run_button or sidebar_run_button or bool(auto_run_simulation)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def render_matplotlib_figure(fig, stretch=False):
    """
    Render matplotlib figures with explicit Streamlit width behavior.
    Also keeps legend font sizes consistent across analysis panels.
    """
    for axis in getattr(fig, "axes", []):
        legend = axis.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(ANALYSIS_LEGEND_FONTSIZE)
            legend_title = legend.get_title()
            if legend_title is not None:
                legend_title.set_fontsize(ANALYSIS_LEGEND_FONTSIZE)

    try:
        st.pyplot(fig, width="stretch" if stretch else "content")
    except TypeError:
        st.pyplot(fig, use_container_width=bool(stretch))


def place_legend_outside_right(ax, fig, anchor_x=1.01, right_margin=0.78):
    """Place legend outside right to prevent overlap with data."""
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) == 0:
        return
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(anchor_x, 1.0),
        borderaxespad=0.0,
        framealpha=0.88,
    )
    fig.subplots_adjust(right=right_margin)


def expand_limits(vmin, vmax, pad_ratio=0.12, pad_min=0.01):
    span = float(vmax - vmin)
    if span < 1.0e-9:
        center = 0.5 * (float(vmin) + float(vmax))
        half = max(float(pad_min), max(abs(center), 1.0) * float(pad_ratio))
        return center - half, center + half
    pad = max(float(pad_ratio) * span, float(pad_min))
    return float(vmin - pad), float(vmax + pad)


def compute_auto_axis_limits_3d(result, pad_ratio=0.12):
    position = result["position"]
    obj = result["object"]
    jet = result.get("jet")
    target = result.get("target")
    sim = result.get("simulation")

    x_vals = [position[:, 0]]
    y_vals = [position[:, 1]]
    z_vals = [position[:, 2]]

    if jet is not None:
        x_vals.append(np.array([jet.x_center - jet.sigma, jet.x_center + jet.sigma], dtype=float))
        y_vals.append(np.array([jet.y_center - jet.sigma, jet.y_center + jet.sigma], dtype=float))
        z_vals.append(np.array([jet.z_center - jet.sigma, jet.z_center + jet.sigma], dtype=float))

    if target is not None:
        x_vals.append(np.array([target.x_min, target.x_max], dtype=float))
    if sim is not None:
        z_vals.append(np.array([sim.landing_z], dtype=float))

    x_all = np.concatenate(x_vals)
    y_all = np.concatenate(y_vals)
    z_all = np.concatenate(z_vals)

    body_pad = 0.65 * max(float(obj.size_x), float(obj.size_y), float(obj.size_z))
    x_min, x_max = expand_limits(np.min(x_all) - body_pad, np.max(x_all) + body_pad, pad_ratio=pad_ratio, pad_min=0.03)
    y_min, y_max = expand_limits(np.min(y_all) - body_pad, np.max(y_all) + body_pad, pad_ratio=pad_ratio, pad_min=0.03)
    z_min, z_max = expand_limits(np.min(z_all) - body_pad, np.max(z_all) + body_pad, pad_ratio=pad_ratio, pad_min=0.03)

    return {"x": (x_min, x_max), "y": (y_min, y_max), "z": (z_min, z_max)}


def compute_auto_axis_limits_2d(x_values, y_values, pad_ratio=0.12, pad_min=0.01):
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    x_span = float(np.max(x_arr) - np.min(x_arr))
    y_span = float(np.max(y_arr) - np.min(y_arr))
    ref_span = max(x_span, y_span, 1.0e-12)

    if x_span < 1.0e-9:
        x_center = 0.5 * (float(np.min(x_arr)) + float(np.max(x_arr)))
        x_half = max(float(pad_min), float(pad_ratio) * ref_span)
        x_min, x_max = x_center - x_half, x_center + x_half
    else:
        x_min, x_max = expand_limits(
            np.min(x_arr),
            np.max(x_arr),
            pad_ratio=pad_ratio,
            pad_min=pad_min,
        )

    if y_span < 1.0e-9:
        y_center = 0.5 * (float(np.min(y_arr)) + float(np.max(y_arr)))
        y_half = max(float(pad_min), float(pad_ratio) * ref_span)
        y_min, y_max = y_center - y_half, y_center + y_half
    else:
        y_min, y_max = expand_limits(
            np.min(y_arr),
            np.max(y_arr),
            pad_ratio=pad_ratio,
            pad_min=pad_min,
        )

    return (x_min, x_max), (y_min, y_max)

def result_to_dataframe(result):
    time = result["time"]
    position = result["position"]
    velocity = result["velocity"]
    angular_velocity = result["angular_velocity"]
    force_jet = result["force_jet"]
    force_drag = result["force_drag"]
    force_gravity = result["force_gravity"]
    force_total = result["force_total"]
    torque_jet = result["torque_jet"]

    data = {
        "time_s": time,
        "x_m": position[:, 0],
        "y_m": position[:, 1],
        "z_m": position[:, 2],
        "vx_m_per_s": velocity[:, 0],
        "vy_m_per_s": velocity[:, 1],
        "vz_m_per_s": velocity[:, 2],
        "omega_x_rad_per_s": angular_velocity[:, 0],
        "omega_y_rad_per_s": angular_velocity[:, 1],
        "omega_z_rad_per_s": angular_velocity[:, 2],
        "F_jet_x_N": force_jet[:, 0],
        "F_jet_y_N": force_jet[:, 1],
        "F_jet_z_N": force_jet[:, 2],
        "F_drag_x_N": force_drag[:, 0],
        "F_drag_y_N": force_drag[:, 1],
        "F_drag_z_N": force_drag[:, 2],
        "F_gravity_x_N": force_gravity[:, 0],
        "F_gravity_y_N": force_gravity[:, 1],
        "F_gravity_z_N": force_gravity[:, 2],
        "F_total_x_N": force_total[:, 0],
        "F_total_y_N": force_total[:, 1],
        "F_total_z_N": force_total[:, 2],
        "tau_jet_x_Nm": torque_jet[:, 0],
        "tau_jet_y_Nm": torque_jet[:, 1],
        "tau_jet_z_Nm": torque_jet[:, 2],
    }

    return pd.DataFrame(data)


def make_parameter_dict(hit_offset):
    object_dict = {
        "type": object_type,
        "mass_kg": mass,
        "drag_coefficient": drag_coefficient,
        "size_x_m": size_x,
        "size_y_m": size_y,
        "size_z_m": size_z,
    }

    if object_type == "rod":
        object_dict["rod_length_m"] = rod_length
        object_dict["rod_radius_m"] = rod_radius

    return {
        "object": object_dict,
        "initial_condition": {
            "position_m": [x0, y0, z0],
            "velocity_m_per_s": [vc, vy_initial, vz_initial],
            "orientation_deg": {
                "roll": roll0,
                "pitch": pitch0,
                "yaw": yaw0,
            },
            "angular_velocity_rad_per_s": [omega_x0, omega_y0, omega_z0],
        },
        "jet": {
            "model": "directional Gaussian air jet with forward-only axial decay",
            "equation": (
                "u_jet = Umax(1+epsilon) "
                "exp(-d_perp^2/(2 sigma^2)) "
                "exp(-s/lambda) e_jet for s>=0, and 0 for s<0"
            ),
            "umax_m_per_s": umax,
            "x_center_m": jet_x_center,
            "y_center_m": jet_y_center,
            "z_center_m": jet_z_center,
            "sigma_m": sigma,
            "axial_decay_m": axial_decay,
            "angle_deg": jet_angle_deg,
            "azimuth_deg": jet_azimuth_deg,
            "direction_model": "e_jet = [cos(elevation)cos(azimuth), cos(elevation)sin(azimuth), sin(elevation)]",
            "angle_convention": "elevation: 0=horizontal, 90=vertical; azimuth rotates around +z",
            "t_on_s": jet_t_on,
            "duration_s": jet_duration,
            "noise_std": noise_std,
            "initial_distance_to_jet_centerline_m": hit_offset,
        },
        "simulation": {
            "dt_s": dt,
            "t_max_s": t_max,
            "gravity_m_per_s2": gravity,
            "air_density_kg_per_m3": air_density,
            "landing_z_m": landing_z,
            "conveyor_length_m": conveyor_length,
            "free_fall_start_offset_m": free_fall_start_offset,
        },
        "target_region": {
            "x_min_m": target_x_min,
            "x_max_m": target_x_max,
        },
        "plot_axis_limits": {
            "use_fixed_axes": use_fixed_axes,
            "x": [x_plot_min, x_plot_max],
            "y": [y_plot_min, y_plot_max],
            "z": [z_plot_min, z_plot_max],
        },
        "animation": {
            "max_frames": animation_max_frames,
            "fps": animation_fps,
            "dpi": animation_dpi,
        },
        "coordinate_convention": {
            "x": "conveyor / main jet direction at 0 deg",
            "y": "lateral belt-width direction",
            "z": "vertical direction",
        },
    }


def make_body_box_vertices_from_points(points_body):
    x_min, y_min, z_min = np.min(points_body, axis=0)
    x_max, y_max, z_max = np.max(points_body, axis=0)

    vertices = np.array(
        [
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ],
        dtype=float,
    )

    return vertices


def make_box_faces(vertices_world):
    return [
        [vertices_world[0], vertices_world[1], vertices_world[2], vertices_world[3]],
        [vertices_world[4], vertices_world[5], vertices_world[6], vertices_world[7]],
        [vertices_world[0], vertices_world[1], vertices_world[5], vertices_world[4]],
        [vertices_world[1], vertices_world[2], vertices_world[6], vertices_world[5]],
        [vertices_world[2], vertices_world[3], vertices_world[7], vertices_world[6]],
        [vertices_world[3], vertices_world[0], vertices_world[4], vertices_world[7]],
    ]


def add_body_box_to_axis(
    ax,
    obj,
    position,
    quaternion,
    alpha=0.25,
    facecolor="tab:blue",
    edgecolor="black",
):
    vertices_body = make_body_box_vertices_from_points(obj.surface_points_body)

    vertices_world, _, _ = transform_surface_points(
        position=position,
        quaternion=quaternion,
        points_body=vertices_body,
    )

    faces = make_box_faces(vertices_world)

    body = Poly3DCollection(
        faces,
        alpha=alpha,
        linewidths=1.0,
        edgecolors=edgecolor,
        facecolors=facecolor,
    )

    ax.add_collection3d(body)
    return ax


def add_irregular_points_to_axis(
    ax,
    obj,
    position,
    quaternion,
    alpha=0.5,
    color="tab:purple",
):
    points_world, _, _ = transform_surface_points(
        position=np.asarray(position, dtype=float),
        quaternion=np.asarray(quaternion, dtype=float),
        points_body=obj.surface_points_body,
    )

    if points_world.shape[0] >= 4:
        try:
            hull = ConvexHull(points_world, qhull_options="QJ")
            faces = [points_world[simplex] for simplex in hull.simplices]
            hull_surface = Poly3DCollection(
                faces,
                alpha=alpha,
                linewidths=0.4,
                edgecolors="black",
                facecolors=color,
            )
            ax.add_collection3d(hull_surface)
            return ax
        except Exception:
            pass

    ax.scatter(
        points_world[:, 0],
        points_world[:, 1],
        points_world[:, 2],
        s=12,
        alpha=alpha,
        color=color,
    )

    return ax


def create_cylinder_mesh_body(length, radius, n_length=12, n_theta=24):
    xs = np.linspace(-length / 2.0, length / 2.0, n_length)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=True)

    side_points = []
    for i in range(n_length):
        row = []
        for theta in thetas:
            row.append([xs[i], radius * np.cos(theta), radius * np.sin(theta)])
        side_points.append(row)

    side_points = np.array(side_points, dtype=float)

    left_cap = []
    right_cap = []
    for theta in thetas:
        left_cap.append([-length / 2.0, radius * np.cos(theta), radius * np.sin(theta)])
        right_cap.append([length / 2.0, radius * np.cos(theta), radius * np.sin(theta)])

    left_cap = np.array(left_cap, dtype=float)
    right_cap = np.array(right_cap, dtype=float)

    return side_points, left_cap, right_cap


def transform_body_mesh_points(position, quaternion, points_body):
    original_shape = points_body.shape
    flat = points_body.reshape(-1, 3)

    points_world, _, _ = transform_surface_points(
        position=np.asarray(position, dtype=float),
        quaternion=np.asarray(quaternion, dtype=float),
        points_body=flat,
    )

    return points_world.reshape(original_shape)


def add_rod_cylinder_to_axis(
    ax,
    obj,
    position,
    quaternion,
    alpha=0.35,
    facecolor="tab:purple",
    edgecolor="black",
):
    length = obj.rod_length
    radius = obj.rod_radius

    if length is None or radius is None:
        return add_body_box_to_axis(
            ax=ax,
            obj=obj,
            position=position,
            quaternion=quaternion,
            alpha=alpha,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )

    side_body, left_cap_body, right_cap_body = create_cylinder_mesh_body(
        length=length,
        radius=radius,
        n_length=12,
        n_theta=24,
    )

    side_world = transform_body_mesh_points(position, quaternion, side_body)
    left_cap_world = transform_body_mesh_points(position, quaternion, left_cap_body)
    right_cap_world = transform_body_mesh_points(position, quaternion, right_cap_body)

    X = side_world[:, :, 0]
    Y = side_world[:, :, 1]
    Z = side_world[:, :, 2]

    ax.plot_surface(
        X,
        Y,
        Z,
        alpha=alpha,
        linewidth=0.4,
        edgecolor=edgecolor,
        color=facecolor,
        shade=True,
    )

    left_center = np.array([[-length / 2.0, 0.0, 0.0]], dtype=float)
    right_center = np.array([[length / 2.0, 0.0, 0.0]], dtype=float)
    left_center_world = transform_body_mesh_points(position, quaternion, left_center)[0]
    right_center_world = transform_body_mesh_points(position, quaternion, right_center)[0]

    left_faces = []
    right_faces = []
    for i in range(len(left_cap_world) - 1):
        left_faces.append([left_center_world, left_cap_world[i], left_cap_world[i + 1]])
        right_faces.append([right_center_world, right_cap_world[i], right_cap_world[i + 1]])

    cap_collection = Poly3DCollection(
        left_faces + right_faces,
        alpha=alpha,
        linewidths=0.4,
        edgecolors=edgecolor,
        facecolors=facecolor,
    )
    ax.add_collection3d(cap_collection)

    return ax


def add_body_geometry_to_axis(
    ax,
    obj,
    position,
    quaternion,
    alpha=0.35,
    facecolor="tab:purple",
    edgecolor="black",
):
    if obj.object_type == "rod":
        return add_rod_cylinder_to_axis(
            ax=ax,
            obj=obj,
            position=position,
            quaternion=quaternion,
            alpha=alpha,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )

    if obj.object_type == "irregular":
        return add_irregular_points_to_axis(
            ax=ax,
            obj=obj,
            position=position,
            quaternion=quaternion,
            alpha=max(alpha, 0.45),
            color=facecolor,
        )

    return add_body_box_to_axis(
        ax=ax,
        obj=obj,
        position=position,
        quaternion=quaternion,
        alpha=alpha,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )


def add_target_region_to_3d_axis(
    ax,
    target,
    landing_z,
    axis_limits=None,
    facecolor="lightskyblue",
    edgecolor="tab:blue",
    alpha=0.22,
):
    if axis_limits is not None:
        y_min, y_max = axis_limits["y"]
    else:
        y_min, y_max = -0.50, 0.50

    x_min = target.x_min
    x_max = target.x_max
    z = landing_z

    vertices = [
        [x_min, y_min, z],
        [x_max, y_min, z],
        [x_max, y_max, z],
        [x_min, y_max, z],
    ]

    target_face = Poly3DCollection(
        [vertices],
        alpha=alpha,
        linewidths=1.0,
        edgecolors=edgecolor,
        facecolors=facecolor,
    )

    ax.add_collection3d(target_face)
    return ax


def get_pdf_jet_direction(jet):
    elevation = np.deg2rad(getattr(jet, "angle_deg", 0.0))
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


def build_plane_basis_from_normal(normal):
    """Return orthonormal basis vectors (u, v, n) for a plane normal to n."""
    n = np.asarray(normal, dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm < 1.0e-12:
        n = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        n = n / n_norm

    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(n, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)

    u = np.cross(n, ref)
    u = u / max(np.linalg.norm(u), 1.0e-12)
    v = np.cross(n, u)
    v = v / max(np.linalg.norm(v), 1.0e-12)
    return u, v, n


def make_circle_points_normal_to_direction(center, normal, radius, n_points=96):
    u, v, _ = build_plane_basis_from_normal(normal)

    phi = np.linspace(0.0, 2.0 * np.pi, n_points)
    center = np.asarray(center, dtype=float)
    points = center[None, :] + radius * (np.cos(phi)[:, None] * u[None, :] + np.sin(phi)[:, None] * v[None, :])
    return points


def make_gaussian_jet_volume_traces_for_plotly(
    jet,
    color="darkorange",
    iso_levels=(0.42, 0.20),
    axial_samples=36,
    theta_samples=56,
):
    """
    Build translucent surface traces that visualize the directional Gaussian jet
    as 3D iso-envelope shells (U/Umax = constant).
    The plume is forward-only (s >= 0).
    """
    direction = get_pdf_jet_direction(jet)
    center = np.array([jet.x_center, jet.y_center, jet.z_center], dtype=float)
    sigma_eff = max(float(jet.sigma), 1.0e-9)
    axial_decay_eff = max(float(getattr(jet, "axial_decay", 0.35)), 1.0e-9)
    u, v, _ = build_plane_basis_from_normal(direction)

    traces = []
    theta = np.linspace(0.0, 2.0 * np.pi, int(max(theta_samples, 16)))
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    valid_levels = [float(level) for level in iso_levels if 0.0 < float(level) < 1.0]
    if not valid_levels:
        return traces

    max_extent = 0.0
    for level_idx, level in enumerate(valid_levels):
        log_term = -np.log(max(level, 1.0e-12))
        s_extent = axial_decay_eff * log_term
        max_extent = max(max_extent, float(s_extent))
        if s_extent < 1.0e-9:
            continue

        s_values = np.linspace(0.0, s_extent, int(max(axial_samples, 12)))
        x = np.zeros((s_values.shape[0], theta.shape[0]), dtype=float)
        y = np.zeros_like(x)
        z = np.zeros_like(x)

        for i, s in enumerate(s_values):
            radial_term = max(log_term - float(s) / axial_decay_eff, 0.0)
            radius = sigma_eff * np.sqrt(2.0 * radial_term)
            ring_center = center + float(s) * direction
            ring = ring_center[None, :] + radius * (
                cos_t[:, None] * u[None, :] + sin_t[:, None] * v[None, :]
            )
            x[i, :] = ring[:, 0]
            y[i, :] = ring[:, 1]
            z[i, :] = ring[:, 2]

        opacity = 0.14 if level_idx == 0 else 0.08
        traces.append(
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=np.full_like(x, level, dtype=float),
                colorscale=[[0.0, color], [1.0, color]],
                cmin=0.0,
                cmax=1.0,
                opacity=opacity,
                showscale=False,
                hoverinfo="skip",
                name=f"jet plume iso-surface ({int(round(level * 100))}% Umax)",
                showlegend=(level_idx == 0),
            )
        )

    if max_extent > 1.0e-9:
        s_line = np.linspace(0.0, max_extent, 90)
        centerline = center[None, :] + s_line[:, None] * direction[None, :]
        traces.append(
            go.Scatter3d(
                x=centerline[:, 0],
                y=centerline[:, 1],
                z=centerline[:, 2],
                mode="lines",
                line={"color": color, "width": 4},
                name="jet centerline",
                hoverinfo="skip",
            )
        )

    return traces


def add_directional_gaussian_jet_to_3d_axis(
    ax,
    jet,
    y_ref=0.0,
    color="darkorange",
    alpha=0.45,
    ring_radius_factor=1.0,
):
    direction = get_pdf_jet_direction(jet)
    center = np.array([jet.x_center, getattr(jet, "y_center", y_ref), jet.z_center], dtype=float)
    radius = ring_radius_factor * max(float(jet.sigma), 1.0e-9)
    circle = make_circle_points_normal_to_direction(center, direction, radius)

    ax.plot(
        circle[:, 0],
        circle[:, 1],
        circle[:, 2],
        color=color,
        linewidth=2.0,
        alpha=alpha,
        label="jet Gaussian cross-section (1 sigma)",
    )

    ax.scatter(
        center[0],
        center[1],
        center[2],
        s=70,
        marker="+",
        color=color,
        linewidths=2.0,
        label="jet center",
    )

    arrow_length = min(0.85, max(0.12, 0.010 * float(jet.umax)))
    ax.quiver(
        center[0],
        center[1],
        center[2],
        direction[0],
        direction[1],
        direction[2],
        length=arrow_length,
        normalize=True,
        color=color,
        linewidth=2.2,
        arrow_length_ratio=0.20,
    )
    return ax


def apply_3d_axis_limits(ax, axis_limits):
    if axis_limits is not None:
        x_min, x_max = axis_limits["x"]
        y_min, y_max = axis_limits["y"]
        z_min, z_max = axis_limits["z"]
    else:
        x_min, x_max = -0.10, 1.50
        y_min, y_max = -0.50, 0.50
        z_min, z_max = 0.00, 0.60

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    x_range = max(x_max - x_min, 1.0e-6)
    y_range = max(y_max - y_min, 1.0e-6)
    z_range = max(z_max - z_min, 1.0e-6)

    ax.set_box_aspect((x_range, y_range, z_range))


def plot_3d_trajectory(result, show_points=True, axis_limits=None):
    position = result["position"]
    quaternion = result["quaternion"]
    obj = result["object"]
    jet = result["jet"]

    x = position[:, 0]
    y = position[:, 1]
    z = position[:, 2]

    fig = plt.figure(figsize=(9.8, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.68, bottom=0.14, top=0.86)

    trajectory_color = "tab:blue"
    start_color = "tab:blue"
    landing_color = "tab:orange"
    target_facecolor = "lightskyblue"
    target_edgecolor = "tab:blue"
    jet_zone_facecolor = "gold"
    jet_zone_edgecolor = "darkorange"
    initial_body_color = "tab:red"
    final_body_color = "tab:purple"
    surface_point_color = "tab:green"

    ax.plot(x, y, z, linewidth=2, color=trajectory_color, label="COM trajectory")

    ax.scatter(
        x[0],
        y[0],
        z[0],
        s=60,
        marker="o",
        color=start_color,
        label="start",
    )

    landing_position = result["landing_position"]
    if landing_position is not None:
        ax.scatter(
            landing_position[0],
            landing_position[1],
            landing_position[2],
            s=80,
            marker="x",
            color=landing_color,
            linewidths=2.5,
            label="landing",
        )

    target = result.get("target")
    sim = result.get("simulation")

    if target is not None and sim is not None:
        add_target_region_to_3d_axis(
            ax=ax,
            target=target,
            landing_z=sim.landing_z,
            axis_limits=axis_limits,
            facecolor=target_facecolor,
            edgecolor=target_edgecolor,
            alpha=0.22,
        )
        add_directional_gaussian_jet_to_3d_axis(
            ax=ax,
            jet=jet,
            y_ref=position[0, 1],
            color=jet_zone_edgecolor,
            alpha=0.55,
        )

    add_body_geometry_to_axis(
        ax=ax,
        obj=obj,
        position=position[0],
        quaternion=quaternion[0],
        alpha=0.18,
        facecolor=initial_body_color,
        edgecolor="black",
    )

    add_body_axes_to_axis(
        ax=ax,
        position=position[0],
        quaternion=quaternion[0],
        obj=obj,
        linewidth=2.0,
        alpha=0.65,
        label_prefix="initial body",
    )

    add_body_geometry_to_axis(
        ax=ax,
        obj=obj,
        position=position[-1],
        quaternion=quaternion[-1],
        alpha=0.35,
        facecolor=final_body_color,
        edgecolor="black",
    )

    add_body_axes_to_axis(
        ax=ax,
        position=position[-1],
        quaternion=quaternion[-1],
        obj=obj,
        linewidth=2.6,
        alpha=0.95,
        label_prefix="final body",
    )

    if show_points:
        points = result["final_points_world"]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=8,
            alpha=0.6,
            color=surface_point_color,
            label="final surface points",
        )

    ax.set_xlabel("x conveyor / main jet [m]", fontsize=9, labelpad=10)
    ax.set_ylabel("y lateral / belt width [m]", fontsize=9, labelpad=16)
    ax.set_zlabel("z vertical [m]", fontsize=9, labelpad=14)
    ax.tick_params(axis="both", which="major", labelsize=8, pad=2)
    ax.view_init(elev=24, azim=-58)

    if result.get("has_landed", False):
        ax.set_title(
            "3D COM Trajectory, Object Orientation, Target, and Gaussian Jet",
            fontsize=13,
            pad=16,
        )
    else:
        ax.set_title(
            "3D COM Trajectory and Final Simulated State, Not Landed",
            fontsize=13,
            pad=16,
        )

    local_limits = axis_limits if axis_limits is not None else compute_auto_axis_limits_3d(result, pad_ratio=0.08)
    apply_3d_axis_limits(ax, local_limits)

    legend_handles = [
        Line2D([0], [0], color=trajectory_color, linewidth=2, label="COM trajectory"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=start_color,
            markeredgecolor=start_color,
            markersize=8,
            label="start",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color=landing_color,
            markersize=9,
            markeredgewidth=2.5,
            linestyle="None",
            label="landing",
        ),
        Patch(
            facecolor=target_facecolor,
            edgecolor=target_edgecolor,
            alpha=0.22,
            label="target landing region",
        ),
        Patch(
            facecolor=jet_zone_facecolor,
            edgecolor=jet_zone_edgecolor,
            alpha=0.16,
            label="directional Gaussian jet",
        ),
        Patch(
            facecolor=initial_body_color,
            edgecolor="black",
            alpha=0.18,
            label="initial body",
        ),
        Patch(
            facecolor=final_body_color,
            edgecolor="black",
            alpha=0.35,
            label="final / last simulated body",
        ),
    ]

    if show_points:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=surface_point_color,
                markeredgecolor=surface_point_color,
                markersize=5,
                linestyle="None",
                label="final surface points",
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.03, 0.98),
        borderaxespad=0.0,
        fontsize=8,
        framealpha=0.88,
    )

    return fig


def draw_animation_frame(ax, result, frame_index, axis_limits=None, show_points=False):
    """Draw one animation frame on a 3D axis."""
    position = result["position"]
    quaternion = result["quaternion"]
    obj = result["object"]
    jet = result["jet"]
    target = result.get("target")
    sim = result.get("simulation")
    time = result["time"]

    current_time = float(time[frame_index])

    # Jet valve ON/OFF status based on time condition
    jet_is_on = jet.t_on <= current_time <= jet.t_on + jet.duration

    trajectory_color = "tab:blue"
    start_color = "tab:blue"
    target_facecolor = "lightskyblue"
    target_edgecolor = "tab:blue"

    if jet_is_on:
        jet_zone_facecolor = "limegreen"
        jet_zone_edgecolor = "green"
        jet_zone_alpha = 0.28
        jet_status_text = "● JET ON"
        jet_status_color = "green"
    else:
        jet_zone_facecolor = "lightcoral"
        jet_zone_edgecolor = "red"
        jet_zone_alpha = 0.18
        jet_status_text = "● JET OFF"
        jet_status_color = "red"

    current_body_color = "tab:purple"
    surface_point_color = "tab:green"

    ax.clear()

    # Give extra space for the legend outside the plot.
    fig = ax.figure
    fig.subplots_adjust(
        left=0.02,
        right=0.68,
        bottom=0.14,
        top=0.88,
    )

    x = position[: frame_index + 1, 0]
    y = position[: frame_index + 1, 1]
    z = position[: frame_index + 1, 2]

    ax.plot(
        x,
        y,
        z,
        linewidth=2,
        color=trajectory_color,
        label="COM trajectory",
    )

    ax.scatter(
        position[0, 0],
        position[0, 1],
        position[0, 2],
        s=45,
        marker="o",
        color=start_color,
        label="start",
    )

    if target is not None and sim is not None:
        add_target_region_to_3d_axis(
            ax=ax,
            target=target,
            landing_z=sim.landing_z,
            axis_limits=axis_limits,
            facecolor=target_facecolor,
            edgecolor=target_edgecolor,
            alpha=0.22,
        )
        add_directional_gaussian_jet_to_3d_axis(
            ax=ax,
            jet=jet,
            y_ref=position[0, 1],
            color=jet_zone_edgecolor,
            alpha=jet_zone_alpha,
        )

    add_body_geometry_to_axis(
        ax=ax,
        obj=obj,
        position=position[frame_index],
        quaternion=quaternion[frame_index],
        alpha=0.40,
        facecolor=current_body_color,
        edgecolor="black",
    )

    add_body_axes_to_axis(
        ax=ax,
        position=position[frame_index],
        quaternion=quaternion[frame_index],
        obj=obj,
        linewidth=2.4,
        alpha=0.95,
        label_prefix="current body",
    )

    if show_points:
        points_world, _, _ = transform_surface_points(
            position=position[frame_index],
            quaternion=quaternion[frame_index],
            points_body=obj.surface_points_body,
        )

        ax.scatter(
            points_world[:, 0],
            points_world[:, 1],
            points_world[:, 2],
            s=7,
            alpha=0.55,
            color=surface_point_color,
            label="surface points",
        )

    # Title and axis labels
    ax.set_title(
        f"3D Rigid-Body Motion, t = {current_time:.3f} s",
        fontsize=12,
        pad=14,
    )

    ax.set_xlabel(
        "x conveyor / main jet [m]",
        fontsize=9,
        labelpad=10,
    )

    ax.set_ylabel(
        "y lateral / belt width [m]",
        fontsize=9,
        labelpad=16,
    )

    ax.set_zlabel(
        "z vertical [m]",
        fontsize=9,
        labelpad=14,
    )

    ax.tick_params(axis="both", which="major", labelsize=8, pad=2)

    # A fixed view angle helps reduce y/z label overlap.
    ax.view_init(elev=24, azim=-58)

    local_limits = axis_limits if axis_limits is not None else compute_auto_axis_limits_3d(result, pad_ratio=0.08)
    apply_3d_axis_limits(ax, local_limits)

    # -----------------------------------------------------------------
    # Compact jet ON/OFF status overlay
    # ON  = green
    # OFF = red
    # -----------------------------------------------------------------

    ax.text2D(
        0.015,
        0.965,
        jet_status_text,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color=jet_status_color,
        bbox={
            "facecolor": "white",
            "edgecolor": jet_status_color,
            "boxstyle": "round,pad=0.18",
            "alpha": 0.86,
        },
    )

    ax.text2D(
        0.015,
        0.920,
        f"t_on = {jet.t_on:.3f} s | duration = {jet.duration:.3f} s",
        transform=ax.transAxes,
        fontsize=7,
        color="black",
        bbox={
            "facecolor": "white",
            "edgecolor": "lightgray",
            "boxstyle": "round,pad=0.16",
            "alpha": 0.72,
        },
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=trajectory_color,
            linewidth=2,
            label="COM trajectory",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=start_color,
            markeredgecolor=start_color,
            markersize=6,
            label="start",
        ),
        Patch(
            facecolor=target_facecolor,
            edgecolor=target_edgecolor,
            alpha=0.22,
            label="target landing region",
        ),
        Patch(
            facecolor=jet_zone_facecolor,
            edgecolor=jet_zone_edgecolor,
            alpha=jet_zone_alpha,
            label=f"directional Gaussian jet ({'ON' if jet_is_on else 'OFF'})",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=jet_status_color,
            markeredgecolor=jet_status_color,
            markersize=7,
            linestyle="None",
            label=f"jet status: {'ON' if jet_is_on else 'OFF'}",
        ),
        Patch(
            facecolor=current_body_color,
            edgecolor="black",
            alpha=0.40,
            label="current body",
        ),
    ]

    if show_points:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=surface_point_color,
                markeredgecolor=surface_point_color,
                markersize=4,
                linestyle="None",
                label="surface points",
            )
        )

    # Put legend outside the 3D axes to avoid covering the trajectory.
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.98),
        borderaxespad=0.0,
        fontsize=8,
        framealpha=0.88,
    )


def create_3d_animation_gif(
    result,
    axis_limits,
    max_frames,
    fps,
    dpi,
    show_points=False,
):
    """
    Create GIF animation bytes from a simulation result.

    This function only creates the GIF in memory for preview/download.
    It does NOT save the GIF permanently to the project folder.
    """
    position = result["position"]
    n_steps = len(position)

    if n_steps <= 1:
        raise ValueError("Not enough trajectory points to create an animation.")

    n_frames = int(min(max_frames, n_steps))
    frame_indices = np.linspace(0, n_steps - 1, n_frames, dtype=int)
    frame_indices = np.unique(frame_indices)

    fig = plt.figure(figsize=(10.2, 6.4))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_index):
        draw_animation_frame(
            ax=ax,
            result=result,
            frame_index=int(frame_index),
            axis_limits=axis_limits,
            show_points=show_points,
        )
        return []

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=1000 / max(fps, 1),
        blit=False,
    )

    # Use a temporary file only to generate GIF bytes.
    # The final GIF is not saved to the project folder here.
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp_file:
        temp_path = Path(tmp_file.name)

    writer = PillowWriter(fps=fps)
    animation.save(temp_path, writer=writer, dpi=dpi)

    plt.close(fig)

    gif_bytes = temp_path.read_bytes()

    try:
        temp_path.unlink()
    except OSError:
        pass

    return gif_bytes


def save_gif_to_project_folder(gif_bytes, filename="simulator_3d_animation.gif"):
    """
    Save GIF bytes to the project results folder only when the user clicks Save.
    """
    output_dir = PROJECT_ROOT / "results" / "simulator" / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    output_path.write_bytes(gif_bytes)

    return output_path


def plot_xy_landing_map(result, target, axis_limits=None):
    position = result["position"]
    jet = result["jet"]

    x = position[:, 0]
    y = position[:, 1]

    fig, ax = plt.subplots(figsize=ANALYSIS_FIGSIZE)

    ax.plot(x, y, linewidth=2, label="COM path in x-y")
    ax.scatter(x[0], y[0], s=60, marker="o", label="start")

    landing_position = result["landing_position"]
    if landing_position is not None:
        ax.scatter(
            landing_position[0],
            landing_position[1],
            s=80,
            marker="x",
            label="landing",
        )

    ax.axvspan(
        target.x_min,
        target.x_max,
        alpha=0.22,
        color="lightskyblue",
        label="target x region",
    )
    ax.axvline(target.x_min, linestyle="--", linewidth=1.0)
    ax.axvline(target.x_max, linestyle="--", linewidth=1.0)

    # Use the actual jet center y-coordinate from the slider/model.
    # Do NOT use position[0, 1], because that is the initial COM y position.
    jet_center_y = getattr(jet, "y_center", position[0, 1])

    direction = get_pdf_jet_direction(jet)
    arrow_length = min(0.24, max(0.04, 0.004 * float(jet.umax)))
    arrow_tip_x = jet.x_center + arrow_length * direction[0]
    arrow_tip_y = jet_center_y + arrow_length * direction[1]

    ax.scatter(
        jet.x_center,
        jet_center_y,
        s=80,
        marker="+",
        label="jet center projection",
    )

    ax.arrow(
        jet.x_center,
        jet_center_y,
        arrow_length * direction[0],
        arrow_length * direction[1],
        head_width=0.012,
        head_length=0.016,
        linewidth=1.2,
        length_includes_head=True,
        label="jet direction projection",
    )

    if axis_limits is not None:
        ax.set_xlim(axis_limits["x"])
        ax.set_ylim(axis_limits["y"])
    else:
        x_extra = np.array([target.x_min, target.x_max, jet.x_center, arrow_tip_x], dtype=float)
        y_extra = np.array([jet_center_y, arrow_tip_y], dtype=float)
        x_lim, y_lim = compute_auto_axis_limits_2d(
            np.concatenate([x, x_extra]),
            np.concatenate([y, y_extra]),
            pad_ratio=0.10,
            pad_min=0.01,
        )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    ax.set_xlabel("x conveyor / main jet direction [m]")
    ax.set_ylabel("y lateral / belt-width position [m]")
    ax.set_title("x-y Projection: Jet Center, Direction, and Target x Region")
    ax.grid(True)
    place_legend_outside_right(ax=ax, fig=fig, anchor_x=1.02, right_margin=0.77)

    return fig


def plot_yz_projection_with_jet(result, jet, axis_limits=None):
    position = result["position"]
    y = position[:, 1]
    z = position[:, 2]

    fig, ax = plt.subplots(figsize=ANALYSIS_FIGSIZE)

    ax.plot(y, z, linewidth=2, label="COM path in y-z")
    ax.scatter(y[0], z[0], s=60, marker="o", label="start")

    landing_position = result["landing_position"]
    if landing_position is not None:
        ax.scatter(
            landing_position[1],
            landing_position[2],
            s=80,
            marker="x",
            label="landing",
        )

    # Use the actual jet center y-coordinate from the slider/model.
    jet_center_y = getattr(jet, "y_center", position[0, 1])
    jet_center_z = jet.z_center

    direction = get_pdf_jet_direction(jet)
    arrow_length = min(0.14, max(0.025, 0.0025 * float(jet.umax)))

    ax.scatter(
        jet_center_y,
        jet_center_z,
        s=80,
        marker="+",
        label="jet center (y,z)",
    )

    ax.arrow(
        jet_center_y,
        jet_center_z,
        arrow_length * direction[1],
        arrow_length * direction[2],
        head_width=0.008,
        head_length=0.011,
        linewidth=1.0,
        length_includes_head=True,
        label="jet direction projection",
    )

    arrow_tip_y = jet_center_y + arrow_length * direction[1]
    arrow_tip_z = jet_center_z + arrow_length * direction[2]

    y_extra = [jet_center_y, arrow_tip_y]
    z_extra = [jet_center_z, arrow_tip_z]
    if landing_position is not None:
        y_extra.append(float(landing_position[1]))
        z_extra.append(float(landing_position[2]))

    y_lim, z_lim = compute_auto_axis_limits_2d(
        np.concatenate([y, np.asarray(y_extra, dtype=float)]),
        np.concatenate([z, np.asarray(z_extra, dtype=float)]),
        pad_ratio=0.10,
        pad_min=0.01,
    )
    ax.set_xlim(y_lim)
    ax.set_ylim(z_lim)

    ax.set_xlabel("y lateral / belt-width position [m]")
    ax.set_ylabel("z vertical position [m]")
    ax.set_title("y-z Projection: Jet Lateral Position and Vertical Motion")
    ax.grid(True)
    place_legend_outside_right(ax=ax, fig=fig, anchor_x=1.02, right_margin=0.77)

    return fig


def plot_xz_projection(result, target, axis_limits=None):
    position = result["position"]
    jet = result["jet"]

    x = position[:, 0]
    z = position[:, 2]

    fig, ax = plt.subplots(figsize=ANALYSIS_FIGSIZE)

    ax.plot(x, z, linewidth=2, label="COM path in x-z")
    ax.scatter(x[0], z[0], s=60, marker="o", label="start")

    landing_position = result["landing_position"]
    if landing_position is not None:
        ax.scatter(landing_position[0], landing_position[2], s=80, marker="x", label="landing")

    ax.axvspan(target.x_min, target.x_max, alpha=0.22, color="lightskyblue", label="target x region")
    ax.axvline(target.x_min, linestyle="--", linewidth=1.0)
    ax.axvline(target.x_max, linestyle="--", linewidth=1.0)

    direction = get_pdf_jet_direction(jet)
    ax.scatter(jet.x_center, jet.z_center, s=80, marker="+", label="jet center (x,z)")
    arrow_length = min(0.24, max(0.04, 0.004 * float(jet.umax)))
    arrow_tip_x = jet.x_center + arrow_length * direction[0]
    arrow_tip_z = jet.z_center + arrow_length * direction[2]
    ax.arrow(
        jet.x_center,
        jet.z_center,
        arrow_length * direction[0],
        arrow_length * direction[2],
        head_width=0.012,
        head_length=0.016,
        linewidth=1.2,
        length_includes_head=True,
        label="jet direction projection",
    )

    if axis_limits is not None:
        ax.set_xlim(axis_limits["x"])
        ax.set_ylim(axis_limits["z"])
    else:
        x_extra = np.array([target.x_min, target.x_max, jet.x_center, arrow_tip_x], dtype=float)
        z_extra = np.array([jet.z_center, arrow_tip_z], dtype=float)
        x_lim, z_lim = compute_auto_axis_limits_2d(
            np.concatenate([x, x_extra]),
            np.concatenate([z, z_extra]),
            pad_ratio=0.10,
            pad_min=0.01,
        )
        ax.set_xlim(x_lim)
        ax.set_ylim(z_lim)

    ax.set_xlabel("x conveyor position [m]")
    ax.set_ylabel("z vertical position [m]")
    ax.set_title("x-z Projection: Vertical Motion and Jet Center")
    ax.grid(True)
    place_legend_outside_right(ax=ax, fig=fig, anchor_x=1.02, right_margin=0.77)

    return fig


def plot_gaussian_jet_field_xy_slice(result, jet, axis_limits=None):
    """
    Plot the directional Gaussian jet velocity magnitude on an x-y slice
    at z = jet.z_center.

    Current coordinate convention:
    - x = conveyor / main jet direction at 0 deg
    - y = lateral belt-width direction
    - z = vertical direction

    This plot shows |u_jet| on the plane z = zj.

    The model shown here matches the directional Gaussian model:
        u = Umax * exp(-d_perp^2/(2*sigma^2)) * exp(-s/lambda) * e_jet, s >= 0
        u = 0, s < 0
    """
    position = result["position"]
    direction = get_pdf_jet_direction(jet)
    arrow_length = min(0.24, max(0.04, 0.004 * float(jet.umax)))
    arrow_tip_x = jet.x_center + arrow_length * direction[0]
    arrow_tip_y = jet.y_center + arrow_length * direction[1]

    x_extent = np.array(
        [jet.x_center - 3.0 * jet.sigma, jet.x_center + 3.0 * jet.sigma, arrow_tip_x],
        dtype=float,
    )
    y_extent = np.array(
        [jet.y_center - 3.0 * jet.sigma, jet.y_center + 3.0 * jet.sigma, arrow_tip_y],
        dtype=float,
    )

    landing_position = result["landing_position"]
    if landing_position is not None:
        x_extent = np.append(x_extent, float(landing_position[0]))
        y_extent = np.append(y_extent, float(landing_position[1]))

    x_minmax, y_minmax = compute_auto_axis_limits_2d(
        np.concatenate([position[:, 0], x_extent]),
        np.concatenate([position[:, 1], y_extent]),
        pad_ratio=0.10,
        pad_min=0.01,
    )
    x_min, x_max = x_minmax
    y_min, y_max = y_minmax

    x_grid = np.linspace(x_min, x_max, 120)
    y_grid = np.linspace(y_min, y_max, 120)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Slice at the jet center height.
    Z = np.full_like(X, jet.z_center)

    centerline_point = np.array(
        [jet.x_center, jet.y_center, jet.z_center],
        dtype=float,
    )

    points = np.stack(
        [X.ravel(), Y.ravel(), Z.ravel()],
        axis=1,
    )

    rel = points - centerline_point[None, :]

    # s is the signed axial coordinate along the jet direction.
    # Forward-only jet: no magnitude for s < 0.
    axial = rel @ direction

    # d_perp is distance from the jet centerline.
    perp = rel - axial[:, None] * direction[None, :]
    d_perp2 = np.sum(perp * perp, axis=1)

    sigma_eff = max(float(jet.sigma), 1.0e-12)
    axial_decay_eff = max(float(getattr(jet, "axial_decay", 0.35)), 1.0e-12)

    radial_profile = np.exp(-d_perp2 / (2.0 * sigma_eff ** 2))
    forward_mask = axial >= 0.0
    axial_profile = np.zeros_like(axial, dtype=float)
    axial_profile[forward_mask] = np.exp(-axial[forward_mask] / axial_decay_eff)

    speed = float(jet.umax) * radial_profile * axial_profile
    speed_field = speed.reshape(X.shape)

    # Use a wider canvas for this panel to separate axes/colorbar/legend cleanly.
    fig, ax = plt.subplots(figsize=(7.8, 3.9))

    contour = ax.contourf(
        X,
        Y,
        speed_field,
        levels=30,
    )

    cbar = fig.colorbar(contour, ax=ax, pad=0.02, fraction=0.055)
    cbar.set_label("|u_jet| [m/s]", labelpad=6, fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.yaxis.tick_right()

    ax.scatter(
        jet.x_center,
        jet.y_center,
        s=90,
        marker="+",
        linewidths=2.5,
        label="jet center",
    )

    ax.arrow(
        jet.x_center,
        jet.y_center,
        arrow_length * direction[0],
        arrow_length * direction[1],
        head_width=0.012,
        head_length=0.016,
        linewidth=1.2,
        length_includes_head=True,
        label="jet direction projection",
    )

    # Show COM trajectory projected onto the same x-y plane with jet ON/OFF split.
    time_values = result["time"]
    jet_on_mask = (time_values >= jet.t_on) & (time_values <= jet.t_on + jet.duration)
    x_off = np.where(jet_on_mask, np.nan, position[:, 0])
    y_off = np.where(jet_on_mask, np.nan, position[:, 1])
    x_on = np.where(jet_on_mask, position[:, 0], np.nan)
    y_on = np.where(jet_on_mask, position[:, 1], np.nan)
    ax.plot(x_off, y_off, linewidth=1.8, color="tab:blue", label="COM path projection (jet OFF)")
    ax.plot(x_on, y_on, linewidth=1.8, color="tab:red", label="COM path projection (jet ON)")

    ax.scatter(
        position[0, 0],
        position[0, 1],
        s=50,
        marker="o",
        label="start",
    )

    if landing_position is not None:
        ax.scatter(
            landing_position[0],
            landing_position[1],
            s=70,
            marker="x",
            label="landing",
        )

    ax.set_xlabel("x conveyor / main jet direction [m]", fontsize=11, labelpad=8)
    ax.set_ylabel("y lateral / belt-width position [m]", fontsize=11, labelpad=8)
    ax.set_title(
        f"Gaussian Jet Field Slice in x-y Plane at z = {jet.z_center:.3f} m",
        fontsize=14,
        pad=8,
    )
    ax.tick_params(axis="both", which="major", labelsize=10)

    ax.grid(True)
    place_legend_outside_right(
        ax=ax,
        fig=fig,
        anchor_x=1.42,
        right_margin=0.50,
    )

    return fig


def build_gaussian_jet_field_3d_plot(result, jet, axis_limits=None):
    """Build a dedicated 3D visualization of the forward-only Gaussian jet plume."""
    direction = get_pdf_jet_direction(jet)
    jet_center = np.array([jet.x_center, jet.y_center, jet.z_center], dtype=float)

    if axis_limits is not None:
        limits = axis_limits
    else:
        sigma_eff = max(float(jet.sigma), 1.0e-9)
        axial_decay_eff = max(float(getattr(jet, "axial_decay", 0.35)), 1.0e-9)
        s_extent = axial_decay_eff * (-np.log(0.06))
        plume_tip = jet_center + s_extent * direction
        envelope = np.vstack(
            [
                jet_center[None, :],
                plume_tip[None, :],
                (jet_center + 3.0 * sigma_eff * np.array([1.0, 0.0, 0.0]))[None, :],
                (jet_center - 3.0 * sigma_eff * np.array([1.0, 0.0, 0.0]))[None, :],
                (jet_center + 3.0 * sigma_eff * np.array([0.0, 1.0, 0.0]))[None, :],
                (jet_center - 3.0 * sigma_eff * np.array([0.0, 1.0, 0.0]))[None, :],
                (jet_center + 3.0 * sigma_eff * np.array([0.0, 0.0, 1.0]))[None, :],
                (jet_center - 3.0 * sigma_eff * np.array([0.0, 0.0, 1.0]))[None, :],
            ]
        )
        p_min = np.min(envelope, axis=0)
        p_max = np.max(envelope, axis=0)
        p_center = 0.5 * (p_min + p_max)
        p_span = np.maximum(p_max - p_min, 1.0e-9)
        half_range = np.maximum(0.60 * p_span, 0.05)
        limits = {
            "x": (float(p_center[0] - half_range[0]), float(p_center[0] + half_range[0])),
            "y": (float(p_center[1] - half_range[1]), float(p_center[1] + half_range[1])),
            "z": (float(p_center[2] - half_range[2]), float(p_center[2] + half_range[2])),
        }

    traces = []
    traces.extend(
        make_gaussian_jet_volume_traces_for_plotly(
            jet=jet,
            color="darkorange",
            iso_levels=(0.60, 0.35, 0.15),
            axial_samples=42,
            theta_samples=64,
        )
    )

    traces.append(
        go.Scatter3d(
            x=[jet_center[0]],
            y=[jet_center[1]],
            z=[jet_center[2]],
            mode="markers",
            marker={"size": 6, "color": "darkorange", "symbol": "cross"},
            name="jet center",
            hoverinfo="skip",
        )
    )

    arrow_length = min(0.85, max(0.16, 0.010 * float(jet.umax)))
    traces.append(
        go.Scatter3d(
            x=[jet_center[0], jet_center[0] + arrow_length * direction[0]],
            y=[jet_center[1], jet_center[1] + arrow_length * direction[1]],
            z=[jet_center[2], jet_center[2] + arrow_length * direction[2]],
            mode="lines",
            line={"color": "orange", "width": 7},
            name="jet direction",
            hoverinfo="skip",
        )
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_white",
        height=620,
        margin={"l": 16, "r": 16, "t": 52, "b": 10},
        title="Gaussian Jet Field (3D, Forward-Only)",
        legend={
            "x": 1.01,
            "y": 0.98,
            "bgcolor": "rgba(255,255,255,0.92)",
            "bordercolor": "rgba(170,170,170,0.8)",
            "borderwidth": 1,
            "font": {"size": 13, "color": "black"},
        },
        scene={
            "xaxis": {
                "title": "x conveyor / main jet direction [m]",
                "range": list(limits["x"]),
                "gridcolor": "#dde4ef",
                "showspikes": False,
                "spikesides": False,
            },
            "yaxis": {
                "title": "y lateral / belt width [m]",
                "range": list(limits["y"]),
                "gridcolor": "#dde4ef",
                "showspikes": False,
                "spikesides": False,
            },
            "zaxis": {
                "title": "z vertical [m]",
                "range": list(limits["z"]),
                "gridcolor": "#dde4ef",
                "showspikes": False,
                "spikesides": False,
            },
            "aspectmode": "manual",
            "aspectratio": {
                "x": max(limits["x"][1] - limits["x"][0], 1.0e-6),
                "y": max(limits["y"][1] - limits["y"][0], 1.0e-6),
                "z": max(limits["z"][1] - limits["z"][0], 1.0e-6),
            },
            "camera": {"eye": {"x": 0.92, "y": -1.16, "z": 0.78}},
        },
    )
    return fig


def plot_time_history(result, key, ylabel, title, labels):
    time = result["time"]
    values = result[key]

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, label in enumerate(labels):
        ax.plot(time, values[:, i], label=label)

    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    return fig


def plot_magnitude_history(result, key, ylabel, title):
    time = result["time"]
    values = result[key]
    magnitude = np.linalg.norm(values, axis=1)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(time, magnitude, linewidth=2)

    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

    return fig


def plot_total_force_history(result):
    time = result["time"]
    force_total = result["force_total"]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(time, force_total[:, 0], label="Ftotal,x")
    ax.plot(time, force_total[:, 1], label="Ftotal,y")
    ax.plot(time, force_total[:, 2], label="Ftotal,z")

    ax.set_xlabel("time [s]")
    ax.set_ylabel("total force [N]")
    ax.set_title("Total Force = Jet + Drag + Gravity")
    ax.grid(True)
    ax.legend()

    return fig


def plot_force_breakdown_z(result):
    time = result["time"]
    force_jet = result["force_jet"]
    force_drag = result["force_drag"]
    force_gravity = result["force_gravity"]
    force_total = result["force_total"]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(time, force_jet[:, 2], label="Fjet,z")
    ax.plot(time, force_drag[:, 2], label="Fdrag,z")
    ax.plot(time, force_gravity[:, 2], label="Fgravity,z")
    ax.plot(time, force_total[:, 2], label="Ftotal,z", linewidth=2)

    ax.set_xlabel("time [s]")
    ax.set_ylabel("z-force [N]")
    ax.set_title("Z-Force Breakdown")
    ax.grid(True)
    ax.legend()

    return fig



def plot_angular_velocity_and_speed(result):
    """Plot angular velocity components and angular speed over time."""
    time = result["time"]
    omega = result["angular_velocity"]
    omega_mag = np.linalg.norm(omega, axis=1)
    jet = result.get("jet")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(time, omega[:, 0], label="omega_x")
    ax.plot(time, omega[:, 1], label="omega_y")
    ax.plot(time, omega[:, 2], label="omega_z")
    ax.plot(time, omega_mag, linewidth=2.5, label="|omega|")

    if jet is not None:
        ax.axvspan(
            jet.t_on,
            jet.t_on + jet.duration,
            alpha=0.15,
            label="jet ON window",
        )

    ax.set_xlabel("time [s]")
    ax.set_ylabel("angular velocity [rad/s]")
    ax.set_title("Angular Velocity History")
    ax.grid(True)
    ax.legend()

    return fig


def plot_jet_torque_and_magnitude(result):
    """Plot jet torque components and torque magnitude over time."""
    time = result["time"]
    torque = result["torque_jet"]
    torque_mag = np.linalg.norm(torque, axis=1)
    jet = result.get("jet")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(time, torque[:, 0], label="tau_x")
    ax.plot(time, torque[:, 1], label="tau_y")
    ax.plot(time, torque[:, 2], label="tau_z")
    ax.plot(time, torque_mag, linewidth=2.5, label="|tau|")

    if jet is not None:
        ax.axvspan(
            jet.t_on,
            jet.t_on + jet.duration,
            alpha=0.15,
            label="jet ON window",
        )

    ax.set_xlabel("time [s]")
    ax.set_ylabel("jet torque [N m]")
    ax.set_title("Jet Torque History")
    ax.grid(True)
    ax.legend()

    return fig


def compute_rotational_invariants(result):
    """
    Compute angular momentum and rotational kinetic energy histories.

    H = I_world * omega
    T_rot = 0.5 * omega dot I_world omega

    If jet torque is zero after jet shutoff, |H| and T_rot should remain
    approximately constant in an ideal torque-free rigid-body simulation.
    """
    obj = result["object"]
    quaternion = result["quaternion"]
    omega = result["angular_velocity"]

    h_world = np.zeros_like(omega)
    h_mag = np.zeros(len(omega))
    rotational_energy = np.zeros(len(omega))

    for i in range(len(omega)):
        q = quaternion[i]
        w = omega[i]

        # Convert quaternion to rotation matrix.
        # This duplicates the standard formula so we do not need to modify source/core_3d.py.
        q = np.asarray(q, dtype=float)
        q_norm = np.linalg.norm(q)

        if q_norm < 1.0e-12:
            q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            q = q / q_norm

        q0, q1, q2, q3 = q

        rotation_matrix = np.array(
            [
                [
                    1.0 - 2.0 * (q2 * q2 + q3 * q3),
                    2.0 * (q1 * q2 - q0 * q3),
                    2.0 * (q1 * q3 + q0 * q2),
                ],
                [
                    2.0 * (q1 * q2 + q0 * q3),
                    1.0 - 2.0 * (q1 * q1 + q3 * q3),
                    2.0 * (q2 * q3 - q0 * q1),
                ],
                [
                    2.0 * (q1 * q3 - q0 * q2),
                    2.0 * (q2 * q3 + q0 * q1),
                    1.0 - 2.0 * (q1 * q1 + q2 * q2),
                ],
            ],
            dtype=float,
        )

        inertia_world = rotation_matrix @ obj.inertia_body @ rotation_matrix.T

        h_i = inertia_world @ w

        h_world[i] = h_i
        h_mag[i] = np.linalg.norm(h_i)
        rotational_energy[i] = 0.5 * float(w @ h_i)

    return h_world, h_mag, rotational_energy


def plot_rotational_invariant_history(result):
    """
    Plot angular momentum magnitude and rotational kinetic energy.

    These plots help diagnose whether angular velocity changes after jet shutoff
    are physically plausible torque-free motion or numerical energy drift.
    """
    time = result["time"]
    jet = result.get("jet")

    _, h_mag, rotational_energy = compute_rotational_invariants(result)

    fig1, ax1 = plt.subplots(figsize=(8, 4))

    ax1.plot(time, h_mag, linewidth=2.5, label="|H|")

    if jet is not None:
        ax1.axvspan(
            jet.t_on,
            jet.t_on + jet.duration,
            alpha=0.15,
            label="jet ON window",
        )

    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("angular momentum magnitude [kg m^2/s]")
    ax1.set_title("Angular Momentum Magnitude")
    ax1.grid(True)
    ax1.legend()

    fig2, ax2 = plt.subplots(figsize=(8, 4))

    ax2.plot(time, rotational_energy, linewidth=2.5, label="T_rot")

    if jet is not None:
        ax2.axvspan(
            jet.t_on,
            jet.t_on + jet.duration,
            alpha=0.15,
            label="jet ON window",
        )

    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("rotational kinetic energy [J]")
    ax2.set_title("Rotational Kinetic Energy")
    ax2.grid(True)
    ax2.legend()

    return fig1, fig2


def summarize_rotational_invariants(result):
    """
    Return a compact dataframe for checking drift after jet shutoff.
    """
    time = result["time"]
    jet = result.get("jet")

    _, h_mag, rotational_energy = compute_rotational_invariants(result)

    if len(time) == 0:
        return pd.DataFrame([])

    def nearest_index(t_query):
        return int(np.argmin(np.abs(time - t_query)))

    rows = []

    check_times = [("start", float(time[0]))]

    if jet is not None:
        check_times.extend(
            [
                ("jet on", float(jet.t_on)),
                ("jet off", float(jet.t_on + jet.duration)),
            ]
        )

    check_times.append(("final", float(time[-1])))

    used_labels = set()

    for label, t_query in check_times:
        if label in used_labels:
            continue

        used_labels.add(label)
        idx = nearest_index(t_query)

        rows.append(
            {
                "state": label,
                "time_s": float(time[idx]),
                "angular_momentum_mag": float(h_mag[idx]),
                "rotational_energy_J": float(rotational_energy[idx]),
            }
        )

    return pd.DataFrame(rows)


def plot_angular_diagnostics_summary(result):
    """Return a compact dataframe for checking whether rotation continues after jet shutoff."""
    time = result["time"]
    omega = result["angular_velocity"]
    torque = result["torque_jet"]
    jet = result.get("jet")

    omega_mag = np.linalg.norm(omega, axis=1)
    torque_mag = np.linalg.norm(torque, axis=1)

    rows = []

    if len(time) == 0:
        return pd.DataFrame(rows)

    def nearest_index(t_query):
        return int(np.argmin(np.abs(time - t_query)))

    check_times = [("start", float(time[0]))]

    if jet is not None:
        check_times.extend(
            [
                ("jet on", float(jet.t_on)),
                ("jet off", float(jet.t_on + jet.duration)),
            ]
        )

    check_times.append(("final", float(time[-1])))

    used_labels = set()
    for label, t_query in check_times:
        if label in used_labels:
            continue
        used_labels.add(label)
        idx = nearest_index(t_query)
        rows.append(
            {
                "state": label,
                "time_s": float(time[idx]),
                "omega_x_rad_s": float(omega[idx, 0]),
                "omega_y_rad_s": float(omega[idx, 1]),
                "omega_z_rad_s": float(omega[idx, 2]),
                "omega_mag_rad_s": float(omega_mag[idx]),
                "torque_mag_Nm": float(torque_mag[idx]),
            }
        )

    return pd.DataFrame(rows)


def quaternion_to_rotation_matrix_for_plot(q):
    """
    Convert quaternion [w, x, y, z] to rotation matrix.
    This is used only for visualization and analysis.
    """
    q = np.asarray(q, dtype=float)
    q_norm = np.linalg.norm(q)

    if q_norm < 1.0e-12:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        q = q / q_norm

    q0, q1, q2, q3 = q

    return np.array(
        [
            [
                1.0 - 2.0 * (q2 * q2 + q3 * q3),
                2.0 * (q1 * q2 - q0 * q3),
                2.0 * (q1 * q3 + q0 * q2),
            ],
            [
                2.0 * (q1 * q2 + q0 * q3),
                1.0 - 2.0 * (q1 * q1 + q3 * q3),
                2.0 * (q2 * q3 - q0 * q1),
            ],
            [
                2.0 * (q1 * q3 - q0 * q2),
                2.0 * (q2 * q3 + q0 * q1),
                1.0 - 2.0 * (q1 * q1 + q2 * q2),
            ],
        ],
        dtype=float,
    )


def add_body_axes_to_axis(
    ax,
    position,
    quaternion,
    obj=None,
    axis_length=None,
    linewidth=2.4,
    alpha=0.95,
    label_prefix="body",
):
    """
    Draw body-fixed x/y/z directions in the world frame.

    body x-axis:
        plate: normal direction of +/-x face pair
        rod  : rod length direction

    body y-axis:
        plate: normal direction of +/-y face pair
        rod  : radial direction 1

    body z-axis:
        plate: normal direction of +/-z face pair
        rod  : radial direction 2
    """
    rotation_matrix = quaternion_to_rotation_matrix_for_plot(quaternion)
    origin = np.asarray(position, dtype=float)

    if axis_length is None:
        if obj is not None:
            max_size = max(float(obj.size_x), float(obj.size_y), float(obj.size_z))
            axis_length = 0.75 * max_size
        else:
            axis_length = 0.12

    axis_length = max(float(axis_length), 0.03)

    body_axes = [
        (rotation_matrix @ np.array([1.0, 0.0, 0.0]), "red", f"{label_prefix} x-axis"),
        (rotation_matrix @ np.array([0.0, 1.0, 0.0]), "green", f"{label_prefix} y-axis"),
        (rotation_matrix @ np.array([0.0, 0.0, 1.0]), "blue", f"{label_prefix} z-axis"),
    ]

    for direction, color, label in body_axes:
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            direction[0],
            direction[1],
            direction[2],
            length=axis_length,
            normalize=True,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            arrow_length_ratio=0.22,
            label=label,
        )

    return ax


# ---------------------------------------------------------------------
# Plotly interactive 3D playback
# ---------------------------------------------------------------------

def make_box_mesh_and_wire_traces_for_plotly(
    obj,
    position,
    quaternion,
    mesh_color="mediumpurple",
    mesh_opacity=0.44,
    edge_color="indigo",
    showlegend=True,
):
    """Create mesh + wireframe traces for box-like bodies."""
    vertices_body = make_body_box_vertices_from_points(obj.surface_points_body)
    vertices_world, _, _ = transform_surface_points(
        position=np.asarray(position, dtype=float),
        quaternion=np.asarray(quaternion, dtype=float),
        points_body=vertices_body,
    )

    x = vertices_world[:, 0]
    y = vertices_world[:, 1]
    z = vertices_world[:, 2]

    i = [0, 0, 0, 4, 4, 5, 1, 1, 2, 2, 3, 3]
    j = [1, 2, 3, 5, 6, 6, 2, 5, 3, 6, 0, 7]
    k = [2, 3, 1, 6, 7, 2, 5, 4, 6, 7, 7, 4]

    mesh_trace = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color=mesh_color,
        opacity=mesh_opacity,
        flatshading=True,
        hoverinfo="skip",
        name="current body",
        showlegend=showlegend,
    )

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    ex, ey, ez = [], [], []
    for a, b in edges:
        ex.extend([x[a], x[b], None])
        ey.extend([y[a], y[b], None])
        ez.extend([z[a], z[b], None])

    wire_trace = go.Scatter3d(
        x=ex,
        y=ey,
        z=ez,
        mode="lines",
        line={"color": edge_color, "width": 4},
        hoverinfo="skip",
        name="current body edges",
        showlegend=False,
    )

    return [mesh_trace, wire_trace]


def make_cylinder_cap_mesh_trace_for_plotly(
    cap_world,
    color="mediumpurple",
    opacity=0.44,
    name="current body",
    showlegend=False,
):
    cap_arr = np.asarray(cap_world, dtype=float)
    if cap_arr.shape[0] < 3:
        return None
    if np.linalg.norm(cap_arr[0] - cap_arr[-1]) < 1.0e-10:
        ring = cap_arr[:-1]
    else:
        ring = cap_arr
    if ring.shape[0] < 3:
        return None
    center = np.mean(ring, axis=0)
    vertices = np.vstack([center, ring])
    n_ring = ring.shape[0]

    i_vals = []
    j_vals = []
    k_vals = []
    for idx in range(1, n_ring):
        i_vals.append(0)
        j_vals.append(idx)
        k_vals.append(idx + 1)
    i_vals.append(0)
    j_vals.append(n_ring)
    k_vals.append(1)

    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=i_vals,
        j=j_vals,
        k=k_vals,
        color=color,
        opacity=opacity,
        flatshading=True,
        hoverinfo="skip",
        name=name,
        showlegend=showlegend,
    )


def make_rod_cylinder_traces_for_plotly(
    obj,
    position,
    quaternion,
    mesh_color="mediumpurple",
    mesh_opacity=0.44,
):
    """Create cylinder traces (side + caps) for rod rendering in Plotly."""
    rod_length = obj.rod_length if obj.rod_length is not None else float(obj.size_x)
    rod_radius = obj.rod_radius if obj.rod_radius is not None else 0.5 * max(float(obj.size_y), float(obj.size_z))

    side_body, left_cap_body, right_cap_body = create_cylinder_mesh_body(
        length=float(rod_length),
        radius=float(rod_radius),
        n_length=12,
        n_theta=24,
    )
    side_world = transform_body_mesh_points(position, quaternion, side_body)
    left_cap_world = transform_body_mesh_points(position, quaternion, left_cap_body)
    right_cap_world = transform_body_mesh_points(position, quaternion, right_cap_body)

    x_side = side_world[:, :, 0]
    y_side = side_world[:, :, 1]
    z_side = side_world[:, :, 2]

    side_trace = go.Surface(
        x=x_side,
        y=y_side,
        z=z_side,
        surfacecolor=np.ones_like(x_side, dtype=float),
        colorscale=[[0.0, mesh_color], [1.0, mesh_color]],
        cmin=0.0,
        cmax=1.0,
        opacity=mesh_opacity,
        showscale=False,
        hoverinfo="skip",
        name="current body",
        showlegend=True,
    )

    left_cap_trace = make_cylinder_cap_mesh_trace_for_plotly(
        cap_world=left_cap_world,
        color=mesh_color,
        opacity=mesh_opacity,
        name="current body",
        showlegend=False,
    )
    right_cap_trace = make_cylinder_cap_mesh_trace_for_plotly(
        cap_world=right_cap_world,
        color=mesh_color,
        opacity=mesh_opacity,
        name="current body",
        showlegend=False,
    )

    traces = [side_trace]
    if left_cap_trace is not None:
        traces.append(left_cap_trace)
    if right_cap_trace is not None:
        traces.append(right_cap_trace)
    return traces


def make_irregular_hull_traces_for_plotly(
    obj,
    position,
    quaternion,
    mesh_color="mediumpurple",
    mesh_opacity=0.50,
    edge_color="indigo",
):
    """Create irregular-body hull mesh traces for Plotly."""
    points_world, _, _ = transform_surface_points(
        position=np.asarray(position, dtype=float),
        quaternion=np.asarray(quaternion, dtype=float),
        points_body=obj.surface_points_body,
    )

    if points_world.shape[0] < 4:
        return [
            go.Scatter3d(
                x=points_world[:, 0],
                y=points_world[:, 1],
                z=points_world[:, 2],
                mode="markers",
                marker={"size": 4, "color": mesh_color, "opacity": 0.9},
                hoverinfo="skip",
                name="current body",
            )
        ]

    try:
        hull = ConvexHull(points_world, qhull_options="QJ")
        triangles = np.asarray(hull.simplices, dtype=int)

        mesh_trace = go.Mesh3d(
            x=points_world[:, 0],
            y=points_world[:, 1],
            z=points_world[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            color=mesh_color,
            opacity=mesh_opacity,
            flatshading=True,
            hoverinfo="skip",
            name="current body",
            showlegend=True,
        )

        edge_pairs = set()
        for i0, i1, i2 in triangles:
            for a, b in ((i0, i1), (i1, i2), (i2, i0)):
                if a > b:
                    a, b = b, a
                edge_pairs.add((int(a), int(b)))

        ex, ey, ez = [], [], []
        for a, b in edge_pairs:
            ex.extend([points_world[a, 0], points_world[b, 0], None])
            ey.extend([points_world[a, 1], points_world[b, 1], None])
            ez.extend([points_world[a, 2], points_world[b, 2], None])

        wire_trace = go.Scatter3d(
            x=ex,
            y=ey,
            z=ez,
            mode="lines",
            line={"color": edge_color, "width": 3},
            hoverinfo="skip",
            name="current body edges",
            showlegend=False,
        )

        return [mesh_trace, wire_trace]
    except Exception:
        return [
            go.Scatter3d(
                x=points_world[:, 0],
                y=points_world[:, 1],
                z=points_world[:, 2],
                mode="markers",
                marker={"size": 4, "color": mesh_color, "opacity": 0.9},
                hoverinfo="skip",
                name="current body",
            )
        ]


def make_body_axes_traces_for_plotly(position, quaternion, obj, label_prefix="current body"):
    rotation_matrix = quaternion_to_rotation_matrix_for_plot(quaternion)
    origin = np.asarray(position, dtype=float)
    max_size = max(float(obj.size_x), float(obj.size_y), float(obj.size_z))
    axis_length = max(0.03, 0.75 * max_size)

    axis_defs = [
        (rotation_matrix @ np.array([1.0, 0.0, 0.0]), "red", f"{label_prefix} x-axis"),
        (rotation_matrix @ np.array([0.0, 1.0, 0.0]), "green", f"{label_prefix} y-axis"),
        (rotation_matrix @ np.array([0.0, 0.0, 1.0]), "blue", f"{label_prefix} z-axis"),
    ]
    traces = []
    for direction, color, name in axis_defs:
        traces.append(
            go.Scatter3d(
                x=[origin[0], origin[0] + axis_length * direction[0]],
                y=[origin[1], origin[1] + axis_length * direction[1]],
                z=[origin[2], origin[2] + axis_length * direction[2]],
                mode="lines",
                line={"color": color, "width": 7},
                name=name,
                hoverinfo="skip",
            )
        )
    return traces


def make_current_body_traces_for_plotly(obj, position, quaternion):
    if obj.object_type == "irregular":
        return make_irregular_hull_traces_for_plotly(
            obj=obj,
            position=position,
            quaternion=quaternion,
            mesh_color="mediumpurple",
            mesh_opacity=0.50,
            edge_color="indigo",
        )

    if obj.object_type == "rod":
        return make_rod_cylinder_traces_for_plotly(
            obj=obj,
            position=position,
            quaternion=quaternion,
            mesh_color="mediumpurple",
            mesh_opacity=0.44,
        )

    return make_box_mesh_and_wire_traces_for_plotly(
        obj=obj,
        position=position,
        quaternion=quaternion,
        mesh_color="mediumpurple",
        mesh_opacity=0.44,
        edge_color="indigo",
    )


def build_interactive_plotly_3d(result, axis_limits=None):
    """Build interactive Plotly 3D playback with mouse rotation + play/pause."""
    position = result["position"]
    quaternion = result["quaternion"]
    time = result["time"]
    obj = result["object"]
    jet = result["jet"]
    target = result["target"]
    sim = result["simulation"]

    if axis_limits is not None:
        limits = axis_limits
    else:
        limits = compute_auto_axis_limits_3d(result, pad_ratio=0.08)

    jet_on_mask = (time >= jet.t_on) & (time <= jet.t_on + jet.duration)
    x_off = np.where(jet_on_mask, np.nan, position[:, 0])
    y_off = np.where(jet_on_mask, np.nan, position[:, 1])
    z_off = np.where(jet_on_mask, np.nan, position[:, 2])
    x_on = np.where(jet_on_mask, position[:, 0], np.nan)
    y_on = np.where(jet_on_mask, position[:, 1], np.nan)
    z_on = np.where(jet_on_mask, position[:, 2], np.nan)

    traces = []
    traces.append(
        go.Scatter3d(
            x=x_off, y=y_off, z=z_off,
            mode="lines",
            line={"color": "royalblue", "width": 6},
            name="COM trajectory (jet OFF)",
            hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
        )
    )
    traces.append(
        go.Scatter3d(
            x=x_on, y=y_on, z=z_on,
            mode="lines",
            line={"color": "crimson", "width": 6},
            name="COM trajectory (jet ON)",
            hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
        )
    )
    traces.append(
        go.Scatter3d(
            x=[position[0, 0]], y=[position[0, 1]], z=[position[0, 2]],
            mode="markers",
            marker={"size": 6, "color": "royalblue", "symbol": "circle"},
            name="start",
        )
    )
    if result["landing_position"] is not None:
        lp = result["landing_position"]
        traces.append(
            go.Scatter3d(
                x=[lp[0]], y=[lp[1]], z=[lp[2]],
                mode="markers",
                marker={"size": 7, "color": "darkorange", "symbol": "x"},
                name="landing",
            )
        )

    # Target plane
    y_min, y_max = limits["y"]
    z0 = sim.landing_z
    traces.append(
        go.Mesh3d(
            x=[target.x_min, target.x_max, target.x_max, target.x_min],
            y=[y_min, y_min, y_max, y_max],
            z=[z0, z0, z0, z0],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="lightskyblue",
            opacity=0.24,
            name="target landing region",
            hoverinfo="skip",
        )
    )

    # Conveyor belt support region before release.
    release_x = float(result.get("support_release_x", position[0, 0]))
    belt_z = float(result.get("conveyor_surface_z", position[0, 2]))
    belt_x_min = float(min(position[0, 0], release_x))
    belt_x_max = float(max(position[0, 0], release_x))
    if abs(belt_x_max - belt_x_min) < 1.0e-9:
        belt_x_max = belt_x_min + 1.0e-3
    traces.append(
        go.Mesh3d(
            x=[belt_x_min, belt_x_max, belt_x_max, belt_x_min],
            y=[y_min, y_min, y_max, y_max],
            z=[belt_z, belt_z, belt_z, belt_z],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="lightgray",
            opacity=0.34,
            name="conveyor belt",
            hoverinfo="skip",
        )
    )

    # Jet direction arrow only (keep main playback uncluttered).
    direction = get_pdf_jet_direction(jet)
    jet_center = np.array([jet.x_center, jet.y_center, jet.z_center], dtype=float)

    # 1-sigma concentric rings around the jet axis (outer ring = 1σ).
    sigma_ring = max(float(jet.sigma), 1.0e-9)
    for ring_idx, ring_scale in enumerate([0.34, 0.67, 1.00]):
        ring = make_circle_points_normal_to_direction(
            center=jet_center,
            normal=direction,
            radius=sigma_ring * ring_scale,
            n_points=120,
        )
        traces.append(
            go.Scatter3d(
                x=ring[:, 0],
                y=ring[:, 1],
                z=ring[:, 2],
                mode="lines",
                line={
                    "color": "darkorange",
                    "width": 4 if ring_idx == 2 else 2,
                },
                opacity=0.70 if ring_idx == 2 else 0.25,
                name="jet 1σ (concentric rings)" if ring_idx == 2 else "jet ring guide",
                showlegend=(ring_idx == 2),
                hoverinfo="skip",
            )
        )

    arrow_length = min(0.85, max(0.12, 0.010 * float(jet.umax)))
    arrow_direction = direction.copy()

    # Keep the displayed arrow tip below the main object pass-height so the
    # direction cue does not visually overshoot above the trajectory cloud.
    z_pass_ref = float(np.percentile(position[:, 2], 80.0))
    z_tip_target = z_pass_ref - 0.004
    if arrow_direction[2] > 1.0e-9 and jet_center[2] < z_tip_target:
        max_len_by_z = (z_tip_target - jet_center[2]) / arrow_direction[2]
        if max_len_by_z > 0.0:
            arrow_length = min(arrow_length, 0.95 * max_len_by_z)
            arrow_length = max(arrow_length, 0.04)

    arrow_tip = jet_center + arrow_length * arrow_direction
    traces.append(
        go.Scatter3d(
            x=[jet_center[0], arrow_tip[0]],
            y=[jet_center[1], arrow_tip[1]],
            z=[jet_center[2], arrow_tip[2]],
            mode="lines",
            line={"color": "orange", "width": 7},
            name="jet direction",
            hoverinfo="skip",
        )
    )
    traces.append(
        go.Cone(
            x=[arrow_tip[0]],
            y=[arrow_tip[1]],
            z=[arrow_tip[2]],
            u=[arrow_direction[0]],
            v=[arrow_direction[1]],
            w=[arrow_direction[2]],
            anchor="tip",
            sizemode="absolute",
            sizeref=max(0.010, min(0.035, 0.28 * arrow_length)),
            colorscale=[[0.0, "orange"], [1.0, "orange"]],
            showscale=False,
            hoverinfo="skip",
            name="jet direction (arrow head)",
            showlegend=False,
        )
    )

    # Dynamic traces
    body_traces_0 = make_current_body_traces_for_plotly(obj, position[0], quaternion[0])
    axes_traces_0 = make_body_axes_traces_for_plotly(position[0], quaternion[0], obj, label_prefix="current body")
    dyn0 = body_traces_0 + axes_traces_0

    traces.extend(dyn0)

    static_count = len(traces) - len(dyn0)
    dynamic_count = len(dyn0)

    # Frames
    n_steps = len(position)
    n_frames = int(min(140, n_steps))
    frame_indices = np.linspace(0, n_steps - 1, n_frames, dtype=int)
    frame_indices = np.unique(frame_indices)
    frame_duration_ms = int(max(1, round(1000.0 / max(float(12), 1.0))))
    frames = []

    for idx in frame_indices:
        frame_data = make_current_body_traces_for_plotly(obj, position[idx], quaternion[idx])
        frame_data.extend(make_body_axes_traces_for_plotly(position[idx], quaternion[idx], obj, label_prefix="current body"))
        for tr in frame_data:
            tr.showlegend = False
        frames.append(
            go.Frame(
                data=frame_data,
                traces=list(range(static_count, static_count + dynamic_count)),
                name=f"{time[idx]:.3f}",
            )
        )

    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(
        template="plotly_white",
        height=760,
        margin={"l": 20, "r": 20, "t": 70, "b": 10},
        title=(
            f"3D Interactive Playback, t = {time[0]:.3f} s | "
            f"jet: {'ON' if jet.t_on <= time[0] <= jet.t_on + jet.duration else 'OFF'}"
        ),
        legend={
            "x": 1.02,
            "y": 0.98,
            "bgcolor": "rgba(255,255,255,0.90)",
            "bordercolor": "rgba(170,170,170,0.8)",
            "borderwidth": 1,
            "font": {"size": 15, "color": "black"},
        },
        scene={
            "xaxis": {
                "title": "x conveyor / main jet direction [m]",
                "range": list(limits["x"]),
                "gridcolor": "#dde4ef",
                "showspikes": False,
                "spikesides": False,
            },
            "yaxis": {
                "title": "y lateral / belt width [m]",
                "range": list(limits["y"]),
                "gridcolor": "#dde4ef",
                "showspikes": False,
                "spikesides": False,
            },
            "zaxis": {
                "title": "z vertical [m]",
                "range": list(limits["z"]),
                "gridcolor": "#dde4ef",
                "showspikes": False,
                "spikesides": False,
            },
            "aspectmode": "manual",
            "aspectratio": {
                "x": max(limits["x"][1] - limits["x"][0], 1.0e-6),
                "y": max(limits["y"][1] - limits["y"][0], 1.0e-6),
                "z": max(limits["z"][1] - limits["z"][0], 1.0e-6),
            },
            # Slightly left-looking and closer default view for more zoomed-in playback.
            "camera": {"eye": {"x": 0.72, "y": -1.34, "z": 0.76}},
        },
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 0.96,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": frame_duration_ms, "redraw": True}, "transition": {"duration": 0}}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.10,
                "y": 0.03,
                "len": 0.78,
                "currentvalue": {"prefix": "time: ", "suffix": " s", "font": {"size": 15}},
                "steps": [
                    {"label": f"{time[idx]:.2f}", "method": "animate", "args": [[f"{time[idx]:.3f}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]}
                    for idx in frame_indices
                ],
            }
        ],
    )
    return fig


def build_surface_points_analysis_plot(result, axis_limits=None):
    """Build a dedicated 3D surface-point analysis plot."""
    position = result["position"]
    quaternion = result["quaternion"]
    obj = result["object"]

    points_start, _, _ = transform_surface_points(
        position=position[0],
        quaternion=quaternion[0],
        points_body=obj.surface_points_body,
    )
    n_surface_points = int(points_start.shape[0])

    if axis_limits is not None:
        limits = axis_limits
    else:
        # Auto-frame around the surface cloud itself so the object is visible
        # immediately without manual zoom.
        # Use robust quantiles to avoid a few outliers forcing a zoomed-out view.
        q_low = np.percentile(points_start, 2.0, axis=0)
        q_high = np.percentile(points_start, 98.0, axis=0)
        p_center = 0.5 * (q_low + q_high)
        p_span = np.maximum(q_high - q_low, 1.0e-9)
        span_ref = max(float(np.max(p_span)), 1.0e-3)
        # Aggressive default framing so the surface cloud appears large at first view.
        half_range = np.maximum(0.5 * p_span * 0.86, 0.08 * span_ref)
        limits = {
            "x": (float(p_center[0] - half_range[0]), float(p_center[0] + half_range[0])),
            "y": (float(p_center[1] - half_range[1]), float(p_center[1] + half_range[1])),
            "z": (float(p_center[2] - half_range[2]), float(p_center[2] + half_range[2])),
        }

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points_start[:, 0],
                y=points_start[:, 1],
                z=points_start[:, 2],
                mode="markers",
                marker={
                    "size": 2.2 if n_surface_points >= 500 else 3.2,
                    "color": "royalblue",
                    "opacity": 0.56 if n_surface_points >= 500 else 0.72,
                },
                name="surface points",
                hoverinfo="skip",
            ),
        ]
    )

    fig.update_layout(
        template="plotly_white",
        hovermode=False,
        height=560,
        margin={"l": 16, "r": 16, "t": 56, "b": 10},
        title="Surface Point Analysis",
        showlegend=False,
        legend={
            "x": 1.01,
            "y": 0.98,
            "bgcolor": "rgba(255,255,255,0.92)",
            "bordercolor": "rgba(170,170,170,0.8)",
            "borderwidth": 1,
            "font": {"size": 13, "color": "black"},
        },
        scene={
            "xaxis": {
                "title": "x conveyor / main jet direction [m]",
                "range": list(limits["x"]),
                "gridcolor": "#dde4ef",
                "showspikes": False,
                "spikesides": False,
            },
            "yaxis": {
                "title": "y lateral / belt width [m]",
                "range": list(limits["y"]),
                "gridcolor": "#dde4ef",
                "showspikes": False,
                "spikesides": False,
            },
            "zaxis": {
                "title": "z vertical [m]",
                "range": list(limits["z"]),
                "gridcolor": "#dde4ef",
                "showspikes": False,
                "spikesides": False,
            },
            "aspectmode": "manual",
            "aspectratio": {
                "x": max(limits["x"][1] - limits["x"][0], 1.0e-6),
                "y": max(limits["y"][1] - limits["y"][0], 1.0e-6),
                "z": max(limits["z"][1] - limits["z"][0], 1.0e-6),
            },
            "camera": {
                "projection": {"type": "perspective"},
                "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                "eye": {"x": 0.30, "y": -0.42, "z": 0.26},
            },
        },
    )
    return fig


# ---------------------------------------------------------------------
# Week 2 Analysis helper functions
# These functions reuse the current directional Gaussian jet model.
# They do not change the simulator physics in source/core_3d.py.
# ---------------------------------------------------------------------

ANALYSIS_SESSION_KEYS = [
    "hit_offset_df",
    "hit_offset_axis",
    "noise_df",
    "noise_summary",
    "timing_map_result",
]


def clear_analysis_results():
    """Clear stored analysis outputs when a new base simulation is run."""
    for key in ANALYSIS_SESSION_KEYS:
        if key in st.session_state:
            del st.session_state[key]


def run_case_with_overrides(overrides=None, seed_override=None, force_noise_std=None):
    """
    Re-run the current Week 2 simulation with selected parameter overrides.

    This helper is used for hit-offset sensitivity, noise sensitivity,
    and timing-performance maps. It uses the current sidebar values as
    the base case and changes only values provided in overrides.
    """
    overrides = overrides or {}

    p_case = {
        "object_type": object_type,
        "mass": mass,
        "drag_coefficient": drag_coefficient,
        "size_x": size_x,
        "size_y": size_y,
        "size_z": size_z,
        "rod_length": rod_length,
        "rod_radius": rod_radius,
        "x0": x0,
        "y0": y0,
        "z0": z0,
        "vc": vc,
        "vy_initial": vy_initial,
        "vz_initial": vz_initial,
        "roll0": roll0,
        "pitch0": pitch0,
        "yaw0": yaw0,
        "omega_x0": omega_x0,
        "omega_y0": omega_y0,
        "omega_z0": omega_z0,
        "umax": umax,
        "jet_x_center": jet_x_center,
        "jet_y_center": jet_y_center,
        "jet_z_center": jet_z_center,
        "sigma": sigma,
        "axial_decay": axial_decay,
        "jet_angle_deg": jet_angle_deg,
        "jet_azimuth_deg": jet_azimuth_deg,
        "jet_t_on": jet_t_on,
        "jet_duration": jet_duration,
        "noise_std": noise_std,
        "dt": dt,
        "t_max": t_max,
        "gravity": gravity,
        "air_density": air_density,
        "landing_z": landing_z,
        "conveyor_length": conveyor_length,
        "free_fall_start_offset": free_fall_start_offset,
        "target_x_min": target_x_min,
        "target_x_max": target_x_max,
        "seed": seed,
    }

    p_case.update(overrides)

    if force_noise_std is not None:
        p_case["noise_std"] = force_noise_std

    local_seed = int(seed_override if seed_override is not None else p_case["seed"])

    if p_case["object_type"] == "rod":
        local_rod_length = p_case["rod_length"]
        local_rod_radius = p_case["rod_radius"]
        local_size_x = local_rod_length
        local_size_y = 2.0 * local_rod_radius
        local_size_z = 2.0 * local_rod_radius
    else:
        local_rod_length = None
        local_rod_radius = None
        local_size_x = p_case["size_x"]
        local_size_y = p_case["size_y"]
        local_size_z = p_case["size_z"]

    obj_case = create_object_3d(
        object_type=p_case["object_type"],
        mass=p_case["mass"],
        size_x=local_size_x,
        size_y=local_size_y,
        size_z=local_size_z,
        drag_coefficient=p_case["drag_coefficient"],
        rod_length=local_rod_length,
        rod_radius=local_rod_radius,
        seed=local_seed,
    )

    jet_case = Jet3D(
        umax=p_case["umax"],
        x_center=p_case["jet_x_center"],
        y_center=p_case["jet_y_center"],
        z_center=p_case["jet_z_center"],
        sigma=p_case["sigma"],
        axial_decay=p_case["axial_decay"],
        angle_deg=p_case["jet_angle_deg"],
        azimuth_deg=p_case["jet_azimuth_deg"],
        t_on=p_case["jet_t_on"],
        duration=p_case["jet_duration"],
        noise_std=p_case["noise_std"],
    )

    sim_case = Simulation3D(
        dt=p_case["dt"],
        t_max=p_case["t_max"],
        gravity=p_case["gravity"],
        air_density=p_case["air_density"],
        landing_z=p_case["landing_z"],
        conveyor_length=p_case["conveyor_length"],
        free_fall_start_offset=p_case["free_fall_start_offset"],
    )

    initial_quaternion_case = euler_degrees_to_quaternion(
        roll_deg=p_case["roll0"],
        pitch_deg=p_case["pitch0"],
        yaw_deg=p_case["yaw0"],
    )

    initial_case = InitialCondition3D(
        position=(p_case["x0"], p_case["y0"], p_case["z0"]),
        velocity=(p_case["vc"], p_case["vy_initial"], p_case["vz_initial"]),
        quaternion=initial_quaternion_case,
        angular_velocity=(p_case["omega_x0"], p_case["omega_y0"], p_case["omega_z0"]),
    )

    target_case = TargetRegion3D(
        x_min=p_case["target_x_min"],
        x_max=p_case["target_x_max"],
    )

    return simulate_rigid_body_3d(
        obj=obj_case,
        jet=jet_case,
        sim=sim_case,
        initial=initial_case,
        target=target_case,
        seed=local_seed,
    )


def compute_jet_influence_score(result):
    """
    Compute surface-point-weighted overlap with the current directional jet plume.

    score(t) = sum_i A_i
            * exp(-d_perp_i^2/(2*sigma^2))
            * exp(-abs(s_i)/lambda)

    where:
    - s_i is the signed axial coordinate from the jet center along e_jet.
    - d_perp_i is the perpendicular distance from each surface point to the jet centerline.
    """
    obj_local = result["object"]
    jet_local = result["jet"]

    time_arr = result["time"]
    pos_arr = result["position"]
    quat_arr = result["quaternion"]

    sigma_eff = max(float(jet_local.sigma), 1.0e-12)
    axial_decay_eff = max(float(getattr(jet_local, "axial_decay", 0.35)), 1.0e-12)

    direction = get_pdf_jet_direction(jet_local)
    centerline_point = np.array(
        [jet_local.x_center, jet_local.y_center, jet_local.z_center],
        dtype=float,
    )

    scores = []

    for i in range(len(time_arr)):
        points_world, _, _ = transform_surface_points(
            position=pos_arr[i],
            quaternion=quat_arr[i],
            points_body=obj_local.surface_points_body,
        )

        rel = points_world - centerline_point[None, :]
        axial = rel @ direction

        perp = rel - axial[:, None] * direction[None, :]
        d_perp2 = np.sum(perp * perp, axis=1)

        radial_profile = np.exp(-d_perp2 / (2.0 * sigma_eff ** 2))
        forward_mask = axial >= 0.0
        axial_profile = np.zeros_like(axial, dtype=float)
        axial_profile[forward_mask] = np.exp(-axial[forward_mask] / axial_decay_eff)

        profile = radial_profile * axial_profile

        score = float(np.sum(obj_local.area_weights * profile))
        scores.append(score)

    return np.asarray(time_arr), np.asarray(scores)


def recommend_jet_timing(result):
    """Recommend t_on by aligning the jet pulse center with peak overlap."""
    time_arr, scores = compute_jet_influence_score(result)
    jet_local = result["jet"]

    if len(scores) == 0 or float(np.max(scores)) <= 1.0e-15:
        return {
            "ok": False,
            "recommended_t_on": None,
            "t_peak": None,
            "peak_score": 0.0,
            "current_t_on": jet_local.t_on,
            "duration": jet_local.duration,
            "message": "No meaningful jet-object overlap was found. Adjust jet center, angle, sigma, or t_max.",
            "time": time_arr,
            "score": scores,
        }

    peak_idx = int(np.argmax(scores))
    t_peak = float(time_arr[peak_idx])
    peak_score = float(scores[peak_idx])
    recommended_t_on = max(0.0, t_peak - 0.5 * jet_local.duration)

    return {
        "ok": True,
        "recommended_t_on": recommended_t_on,
        "t_peak": t_peak,
        "peak_score": peak_score,
        "current_t_on": jet_local.t_on,
        "duration": jet_local.duration,
        "message": (
            "The recommended t_on aligns the center of the jet pulse with "
            "the peak surface-point-weighted jet influence time."
        ),
        "time": time_arr,
        "score": scores,
    }


def plot_influence_score(timing_info):
    """Plot jet influence score and timing markers."""
    fig, ax = plt.subplots(figsize=ANALYSIS_FIGSIZE)

    ax.plot(
        timing_info["time"],
        timing_info["score"],
        color="tab:blue",
        linewidth=2,
        label="jet influence score",
    )

    current_t_on = timing_info["current_t_on"]
    duration_local = timing_info["duration"]

    ax.axvspan(
        current_t_on,
        current_t_on + duration_local,
        color="tab:red",
        alpha=0.08,
        label="current jet window",
    )
    ax.axvline(
        current_t_on,
        linestyle="--",
        color="tab:red",
        linewidth=1.7,
        label="current t_on",
    )
    ax.axvline(
        current_t_on + duration_local,
        linestyle="--",
        color="tab:orange",
        linewidth=1.7,
        label="current t_on + duration",
    )

    if timing_info["ok"]:
        ax.axvline(
            timing_info["t_peak"],
            linestyle="-",
            color="tab:green",
            linewidth=2,
            label="peak overlap time",
        )
        ax.axvline(
            timing_info["recommended_t_on"],
            linestyle=":",
            color="tab:purple",
            linewidth=2,
            label="recommended t_on",
        )

    ax.set_xlabel("time [s]")
    ax.set_ylabel("surface-weighted influence score")
    ax.set_title("Recommended Jet Timing Analysis")
    y_formatter = ScalarFormatter(useMathText=False)
    y_formatter.set_scientific(True)
    y_formatter.set_powerlimits((-3, -3))
    ax.yaxis.set_major_formatter(y_formatter)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True)
    place_legend_outside_right(ax=ax, fig=fig, anchor_x=1.02, right_margin=0.77)

    return fig


def run_hit_offset_sensitivity(offset_axis="x", offset_span=0.12, n_samples=17):
    """Sweep jet center offset and collect landing/impulse/rotation metrics."""
    n_samples = int(max(3, n_samples))
    offsets = np.linspace(-float(offset_span), float(offset_span), n_samples)
    rows = []

    for offset in offsets:
        if offset_axis == "x":
            overrides = {"jet_x_center": jet_x_center + float(offset), "noise_std": 0.0}
        elif offset_axis == "y":
            overrides = {"jet_y_center": jet_y_center + float(offset), "noise_std": 0.0}
        elif offset_axis == "z":
            overrides = {"jet_z_center": jet_z_center + float(offset), "noise_std": 0.0}
        else:
            raise ValueError(f"Unsupported offset_axis: {offset_axis}")

        case_result = run_case_with_overrides(
            overrides=overrides,
            seed_override=seed,
            force_noise_std=0.0,
        )

        landing_position = case_result["landing_position"]
        if landing_position is None:
            landing_x = np.nan
            landing_y = np.nan
            landing_z = np.nan
        else:
            landing_x = float(landing_position[0])
            landing_y = float(landing_position[1])
            landing_z = float(landing_position[2])

        rows.append(
            {
                "offset_m": float(offset),
                "landing_x_m": landing_x,
                "landing_y_m": landing_y,
                "landing_z_m": landing_z,
                "success": bool(case_result["success"]),
                "linear_impulse_Ns": float(np.linalg.norm(case_result["jet_impulse"])),
                "angular_impulse_Nms": float(np.linalg.norm(case_result["angular_impulse"])),
                "max_angular_speed_rad_s": float(case_result["max_angular_speed"]),
                "has_landed": bool(case_result["has_landed"]),
            }
        )

    return pd.DataFrame(rows)


def summarize_hit_offset_sensitivity(df):
    """Return compact summary metrics for hit-offset sensitivity."""
    total = int(len(df))
    success_count = int(df["success"].sum()) if total > 0 else 0
    success_rate = (success_count / total) if total > 0 else 0.0

    valid_x = df["landing_x_m"].dropna()
    if len(valid_x) > 0:
        landing_x_span = float(valid_x.max() - valid_x.min())
        target_center = 0.5 * (float(target_x_min) + float(target_x_max))
        best_idx = (df["landing_x_m"] - target_center).abs().idxmin()
        best_offset = float(df.loc[best_idx, "offset_m"])
    else:
        landing_x_span = np.nan
        best_offset = np.nan

    success_df = df.loc[df["success"].astype(bool)]
    if len(success_df) > 0:
        success_offset_min = float(success_df["offset_m"].min())
        success_offset_max = float(success_df["offset_m"].max())
        success_offset_band = success_offset_max - success_offset_min
    else:
        success_offset_min = np.nan
        success_offset_max = np.nan
        success_offset_band = np.nan

    return {
        "total_cases": total,
        "success_count": success_count,
        "success_rate": success_rate,
        "landing_x_span_m": landing_x_span,
        "best_offset_m": best_offset,
        "success_offset_min_m": success_offset_min,
        "success_offset_max_m": success_offset_max,
        "success_offset_band_m": success_offset_band,
    }


def plot_hit_offset_sensitivity(df, offset_axis="x"):
    """Plot hit-offset sensitivity results."""
    fig1, ax1 = plt.subplots(figsize=ANALYSIS_FIGSIZE)
    ax1.plot(df["offset_m"], df["landing_x_m"], marker="o", label="landing x")
    ax1.axhspan(target_x_min, target_x_max, alpha=0.12, label="target x region")
    ax1.set_xlabel(f"jet center offset along {offset_axis} [m]")
    ax1.set_ylabel("landing x [m]")
    ax1.set_title("Hit-Offset Sensitivity: Landing Position")
    ax1.grid(True)
    place_legend_outside_right(ax=ax1, fig=fig1, anchor_x=1.02, right_margin=0.77)

    fig2, ax2 = plt.subplots(figsize=ANALYSIS_FIGSIZE)
    ax2.plot(df["offset_m"], df["linear_impulse_Ns"], marker="o", label="|J|")
    ax2.plot(df["offset_m"], df["angular_impulse_Nms"], marker="s", label="|L|")
    ax2.set_xlabel(f"jet center offset along {offset_axis} [m]")
    ax2.set_ylabel("impulse magnitude")
    ax2.set_title("Hit-Offset Sensitivity: Linear and Angular Impulse")
    ax2.grid(True)
    place_legend_outside_right(ax=ax2, fig=fig2, anchor_x=1.02, right_margin=0.77)

    fig3, ax3 = plt.subplots(figsize=ANALYSIS_FIGSIZE)
    ax3.plot(
        df["offset_m"],
        df["max_angular_speed_rad_s"],
        marker="o",
        label="max |omega|",
    )
    ax3.set_xlabel(f"jet center offset along {offset_axis} [m]")
    ax3.set_ylabel("max angular speed [rad/s]")
    ax3.set_title("Hit-Offset Sensitivity: Rotation Response")
    ax3.grid(True)
    place_legend_outside_right(ax=ax3, fig=fig3, anchor_x=1.02, right_margin=0.77)

    return fig1, fig2, fig3


def run_noise_sensitivity(n_trials=30, analysis_noise_std=0.05):
    """Run repeated simulations with random jet noise."""
    n_trials = int(max(1, n_trials))
    analysis_noise_std = float(max(0.0, analysis_noise_std))
    rows = []

    for i in range(n_trials):
        case_seed = int(seed + 101 * i + 17)
        case_result = run_case_with_overrides(
            overrides={},
            seed_override=case_seed,
            force_noise_std=analysis_noise_std,
        )

        landing_position = case_result["landing_position"]
        if landing_position is None:
            landing_x = np.nan
            landing_y = np.nan
            landing_z = np.nan
        else:
            landing_x = float(landing_position[0])
            landing_y = float(landing_position[1])
            landing_z = float(landing_position[2])

        rows.append(
            {
                "trial": i + 1,
                "seed": case_seed,
                "landing_x_m": landing_x,
                "landing_y_m": landing_y,
                "landing_z_m": landing_z,
                "success": bool(case_result["success"]),
                "has_landed": bool(case_result["has_landed"]),
            }
        )

    return pd.DataFrame(rows)


def summarize_noise_sensitivity(df):
    """Return summary statistics for noise sensitivity results."""
    valid = df.dropna(subset=["landing_x_m"])
    total = len(df)
    success_count = int(df["success"].sum())
    success_rate = success_count / total if total > 0 else 0.0

    return {
        "total_trials": total,
        "success_count": success_count,
        "success_rate": success_rate,
        "mean_landing_x_m": float(valid["landing_x_m"].mean()) if len(valid) else np.nan,
        "std_landing_x_m": float(valid["landing_x_m"].std()) if len(valid) > 1 else np.nan,
        "mean_landing_y_m": float(valid["landing_y_m"].mean()) if len(valid) else np.nan,
        "std_landing_y_m": float(valid["landing_y_m"].std()) if len(valid) > 1 else np.nan,
    }


def plot_noise_sensitivity(df):
    """Plot noise sensitivity landing distribution."""
    fig1, ax1 = plt.subplots(figsize=ANALYSIS_FIGSIZE)
    success_mask = df["success"].astype(bool)

    ax1.scatter(
        df.loc[~success_mask, "landing_x_m"],
        df.loc[~success_mask, "landing_y_m"],
        marker="x",
        label="fail",
    )
    ax1.scatter(
        df.loc[success_mask, "landing_x_m"],
        df.loc[success_mask, "landing_y_m"],
        marker="o",
        label="success",
    )
    ax1.set_xlabel("landing x [m]")
    ax1.set_ylabel("landing y [m]")
    ax1.set_title("Noise Sensitivity: Landing Scatter")
    ax1.grid(True)
    place_legend_outside_right(ax=ax1, fig=fig1, anchor_x=1.02, right_margin=0.77)

    fig2, ax2 = plt.subplots(figsize=ANALYSIS_FIGSIZE)
    valid_x = df["landing_x_m"].dropna()
    if len(valid_x) > 0:
        x_min = float(valid_x.min())
        x_max = float(valid_x.max())
        x_span = max(0.0, x_max - x_min)
        if x_span < 1.0e-8:
            center = float(valid_x.mean())
            pad = max(1.0e-4, abs(center) * 2.0e-4)
            hist_range = (center - pad, center + pad)
            bins = 6
        else:
            pad = max(0.25 * x_span, 1.0e-5)
            hist_range = (x_min - pad, x_max + pad)
            bins = int(min(16, max(6, np.ceil(np.sqrt(len(valid_x)) * 1.6))))

        ax2.hist(
            valid_x,
            bins=bins,
            range=hist_range,
            rwidth=0.92,
            color="tab:blue",
            edgecolor="white",
            linewidth=0.8,
            alpha=0.88,
        )
        ax2.set_xlim(hist_range)
    ax2.axvspan(target_x_min, target_x_max, alpha=0.18, label="target x region")
    ax2.set_xlabel("landing x [m]")
    ax2.set_ylabel("count")
    ax2.set_title("Noise Sensitivity: Landing-x Distribution")
    ax2.grid(True)
    place_legend_outside_right(ax=ax2, fig=fig2, anchor_x=1.02, right_margin=0.77)

    return fig1, fig2


def run_timing_map(grid_size=8, t_on_half_width=0.25, duration_min_factor=0.25, duration_max_factor=1.8):
    """Scan t_on and duration and collect landing x and success maps."""
    grid_size = int(max(3, grid_size))

    t_on_values = np.linspace(
        max(0.0, jet_t_on - float(t_on_half_width)),
        max(0.0, jet_t_on + float(t_on_half_width)),
        grid_size,
    )

    base_duration = max(float(jet_duration), 1.0e-4)
    duration_values = np.linspace(
        max(0.0, float(duration_min_factor) * base_duration),
        max(0.005, float(duration_max_factor) * base_duration),
        grid_size,
    )

    landing_x_map = np.full((grid_size, grid_size), np.nan)
    success_map = np.zeros((grid_size, grid_size), dtype=float)
    rows = []

    for j, duration_value in enumerate(duration_values):
        for i, t_on_value in enumerate(t_on_values):
            case_result = run_case_with_overrides(
                overrides={
                    "jet_t_on": float(t_on_value),
                    "jet_duration": float(duration_value),
                    "noise_std": 0.0,
                },
                seed_override=seed,
                force_noise_std=0.0,
            )

            landing_position = case_result["landing_position"]
            landing_x = float(landing_position[0]) if landing_position is not None else np.nan
            success_value = bool(case_result["success"])

            landing_x_map[j, i] = landing_x
            success_map[j, i] = 1.0 if success_value else 0.0

            rows.append(
                {
                    "t_on_s": float(t_on_value),
                    "duration_s": float(duration_value),
                    "landing_x_m": landing_x,
                    "success": success_value,
                    "has_landed": bool(case_result["has_landed"]),
                }
            )

    return {
        "t_on_values": t_on_values,
        "duration_values": duration_values,
        "landing_x_map": landing_x_map,
        "success_map": success_map,
        "dataframe": pd.DataFrame(rows),
    }


def summarize_timing_map(timing_result):
    """Return compact summary metrics for timing-map results."""
    df = timing_result.get("dataframe", pd.DataFrame())
    total = int(len(df))
    success_count = int(df["success"].sum()) if total > 0 else 0
    success_rate = (success_count / total) if total > 0 else 0.0

    valid = df.dropna(subset=["landing_x_m"])
    if len(valid) > 0:
        landing_x_min = float(valid["landing_x_m"].min())
        landing_x_max = float(valid["landing_x_m"].max())
    else:
        landing_x_min = np.nan
        landing_x_max = np.nan

    success_df = valid.loc[valid["success"].astype(bool)]
    target_center = 0.5 * (float(target_x_min) + float(target_x_max))
    if len(success_df) > 0:
        best_idx = (success_df["landing_x_m"] - target_center).abs().idxmin()
        best_row = success_df.loc[best_idx]
        best_t_on = float(best_row["t_on_s"])
        best_duration = float(best_row["duration_s"])
        best_landing_x = float(best_row["landing_x_m"])
    else:
        best_t_on = np.nan
        best_duration = np.nan
        best_landing_x = np.nan

    return {
        "total_cases": total,
        "success_count": success_count,
        "success_rate": success_rate,
        "landing_x_min_m": landing_x_min,
        "landing_x_max_m": landing_x_max,
        "best_t_on_s": best_t_on,
        "best_duration_s": best_duration,
        "best_landing_x_m": best_landing_x,
    }


def plot_timing_map(timing_result):
    """Plot timing-performance maps."""
    t_on_values = timing_result["t_on_values"]
    duration_values = timing_result["duration_values"]

    fig1, ax1 = plt.subplots(figsize=ANALYSIS_FIGSIZE)
    im1 = ax1.imshow(
        timing_result["landing_x_map"],
        origin="lower",
        aspect="auto",
        extent=[t_on_values[0], t_on_values[-1], duration_values[0], duration_values[-1]],
    )
    ax1.scatter([jet_t_on], [jet_duration], marker="x", s=80, label="current setting")
    ax1.set_xlabel("t_on [s]")
    ax1.set_ylabel("jet duration [s]")
    ax1.set_title("Timing Performance Map: Landing x")
    fig1.colorbar(im1, ax=ax1, label="landing x [m]")
    place_legend_outside_right(ax=ax1, fig=fig1, anchor_x=1.02, right_margin=0.77)

    fig2, ax2 = plt.subplots(figsize=ANALYSIS_FIGSIZE)
    im2 = ax2.imshow(
        timing_result["success_map"],
        origin="lower",
        aspect="auto",
        vmin=0,
        vmax=1,
        extent=[t_on_values[0], t_on_values[-1], duration_values[0], duration_values[-1]],
    )
    ax2.scatter([jet_t_on], [jet_duration], marker="x", s=80, label="current setting")
    ax2.set_xlabel("t_on [s]")
    ax2.set_ylabel("jet duration [s]")
    ax2.set_title("Timing Performance Map: Success")
    fig2.colorbar(im2, ax=ax2, label="success")
    place_legend_outside_right(ax=ax2, fig=fig2, anchor_x=1.02, right_margin=0.77)

    return fig1, fig2


def compute_body_axis_relative_angle_history(result):
    """
    Compute how much each body-fixed direction has changed relative to its initial direction.

    For each body axis:
        angle_axis(t) = arccos( body_axis_world(0) dot body_axis_world(t) )

    Returns angles in degrees.
    """
    quaternion = result["quaternion"]

    body_x = np.array([1.0, 0.0, 0.0], dtype=float)
    body_y = np.array([0.0, 1.0, 0.0], dtype=float)
    body_z = np.array([0.0, 0.0, 1.0], dtype=float)

    x_world = []
    y_world = []
    z_world = []

    for q in quaternion:
        rotation_matrix = quaternion_to_rotation_matrix_for_plot(q)
        x_world.append(rotation_matrix @ body_x)
        y_world.append(rotation_matrix @ body_y)
        z_world.append(rotation_matrix @ body_z)

    x_world = np.asarray(x_world)
    y_world = np.asarray(y_world)
    z_world = np.asarray(z_world)

    def angle_from_initial(axis_history):
        initial_axis = axis_history[0]
        dots = axis_history @ initial_axis
        dots = np.clip(dots, -1.0, 1.0)
        return np.rad2deg(np.arccos(dots))

    x_angle = angle_from_initial(x_world)
    y_angle = angle_from_initial(y_world)
    z_angle = angle_from_initial(z_world)

    return {
        "x_axis_world": x_world,
        "y_axis_world": y_world,
        "z_axis_world": z_world,
        "x_angle_deg": x_angle,
        "y_angle_deg": y_angle,
        "z_angle_deg": z_angle,
    }


def plot_body_axis_relative_angle_history(result):
    """
    Plot the relative direction change of body-fixed x/y/z axes
    from their initial directions.
    """
    time = result["time"]
    jet = result.get("jet")
    axis_data = compute_body_axis_relative_angle_history(result)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        time,
        axis_data["x_angle_deg"],
        linewidth=2,
        label="body x-axis change",
    )
    ax.plot(
        time,
        axis_data["y_angle_deg"],
        linewidth=2,
        label="body y-axis change",
    )
    ax.plot(
        time,
        axis_data["z_angle_deg"],
        linewidth=2,
        label="body z-axis change",
    )

    if jet is not None:
        ax.axvspan(
            jet.t_on,
            jet.t_on + jet.duration,
            alpha=0.15,
            label="jet ON window",
        )

    ax.set_xlabel("time [s]")
    ax.set_ylabel("angle from initial direction [deg]")
    ax.set_title("Body-Axis Direction Change from Initial Orientation")
    ax.grid(True)
    ax.legend()

    return fig


def summarize_body_axis_relative_angle_history(result):
    """
    Summarize body-axis direction changes at key times.
    """
    time = result["time"]
    jet = result.get("jet")
    axis_data = compute_body_axis_relative_angle_history(result)

    if len(time) == 0:
        return pd.DataFrame([])

    def nearest_index(t_query):
        return int(np.argmin(np.abs(time - t_query)))

    check_times = [("start", float(time[0]))]

    if jet is not None:
        check_times.extend(
            [
                ("jet on", float(jet.t_on)),
                ("jet off", float(jet.t_on + jet.duration)),
            ]
        )

    check_times.append(("final", float(time[-1])))

    rows = []
    used_labels = set()

    for label, t_query in check_times:
        if label in used_labels:
            continue

        used_labels.add(label)
        idx = nearest_index(t_query)

        rows.append(
            {
                "state": label,
                "time_s": float(time[idx]),
                "body_x_axis_change_deg": float(axis_data["x_angle_deg"][idx]),
                "body_y_axis_change_deg": float(axis_data["y_angle_deg"][idx]),
                "body_z_axis_change_deg": float(axis_data["z_angle_deg"][idx]),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Run simulation and store result
# ---------------------------------------------------------------------

if run_button:
    obj = create_object_3d(
        object_type=object_type,
        mass=mass,
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        drag_coefficient=drag_coefficient,
        rod_length=rod_length,
        rod_radius=rod_radius,
        seed=seed,
    )

    jet = Jet3D(
        umax=umax,
        x_center=jet_x_center,
        y_center=jet_y_center,
        z_center=jet_z_center,
        sigma=sigma,
        axial_decay=axial_decay,
        angle_deg=jet_angle_deg,
        azimuth_deg=jet_azimuth_deg,
        t_on=jet_t_on,
        duration=jet_duration,
        noise_std=noise_std,
    )

    sim = Simulation3D(
        dt=dt,
        t_max=t_max,
        gravity=gravity,
        air_density=air_density,
        landing_z=landing_z,
        conveyor_length=conveyor_length,
        free_fall_start_offset=free_fall_start_offset,
    )

    initial_quaternion = euler_degrees_to_quaternion(
        roll_deg=roll0,
        pitch_deg=pitch0,
        yaw_deg=yaw0,
    )

    initial = InitialCondition3D(
        position=(x0, y0, z0),
        velocity=(vc, vy_initial, vz_initial),
        quaternion=initial_quaternion,
        angular_velocity=(omega_x0, omega_y0, omega_z0),
    )

    target = TargetRegion3D(
        x_min=target_x_min,
        x_max=target_x_max,
    )

    hit_offset = compute_hit_offset(initial.position, jet)

    result = simulate_rigid_body_3d(
        obj=obj,
        jet=jet,
        sim=sim,
        initial=initial,
        target=target,
        seed=seed,
    )

    df = result_to_dataframe(result)
    csv_data = df.to_csv(index=False).encode("utf-8")
    parameter_dict = make_parameter_dict(hit_offset)
    parameter_json = json.dumps(parameter_dict, indent=2)

    st.session_state["last_result"] = result
    st.session_state["last_hit_offset"] = hit_offset
    st.session_state["last_dataframe"] = df
    st.session_state["last_csv_data"] = csv_data
    st.session_state["last_parameter_json"] = parameter_json

    if "last_gif_bytes" in st.session_state:
        del st.session_state["last_gif_bytes"]

    clear_analysis_results()


# ---------------------------------------------------------------------
# Display result
# ---------------------------------------------------------------------

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    hit_offset = st.session_state["last_hit_offset"]
    df = st.session_state["last_dataframe"]
    csv_data = st.session_state["last_csv_data"]
    parameter_json = st.session_state["last_parameter_json"]

    landing_position = result["landing_position"]
    landing_time = result["landing_time"]
    has_landed = result["has_landed"]
    final_position = result["final_position"]
    final_time = result["final_time"]
    success = result["success"]

    obj = result["object"]
    jet = result["jet"]
    target = result["target"]

    st.subheader("3D Interactive Playback")
    fig_plotly = build_interactive_plotly_3d(
        result=result,
        axis_limits=axis_limits,
    )
    st.plotly_chart(fig_plotly, use_container_width=True, config={"displaylogo": False})
    st.caption("Drag to rotate, scroll to zoom, and use Play/Pause or the time slider for playback without GIF export.")

    st.subheader("Simulation Result")
    row1 = st.columns(4)
    if not has_landed:
        with row1[0]:
            st.metric("Final time [s]", f"{final_time:.4f}")
        with row1[1]:
            st.metric("Final x [m]", f"{final_position[0]:.4f}")
        with row1[2]:
            st.metric("Final y [m]", f"{final_position[1]:.4f}")
        with row1[3]:
            st.metric("Final z [m]", f"{final_position[2]:.4f}")
        st.warning("NOT LANDED: The object did not reach the landing plane within t_max.")
        st.info(
            "Landing time and landing position are not defined for this run. "
            "Increase t_max, reduce upward jet force, reduce jet duration, "
            "or adjust the jet angle if you want the object to land."
        )
    else:
        with row1[0]:
            st.metric("Landing x [m]", f"{landing_position[0]:.4f}")
        with row1[1]:
            st.metric("Landing y [m]", f"{landing_position[1]:.4f}")
        with row1[2]:
            st.metric("Landing z [m]", f"{landing_position[2]:.4f}")
        with row1[3]:
            st.metric("Landing time [s]", f"{landing_time:.4f}")
        if success:
            st.success("SUCCESS: landing x is inside target region.")
        else:
            st.error("FAIL: landing x missed the target region.")

    row2 = st.columns(4)
    with row2[0]:
        st.metric("Initial hit offset [m]", f"{hit_offset:.4f}")
    with row2[1]:
        st.metric("Max angular speed [rad/s]", f"{result['max_angular_speed']:.4f}")
    with row2[2]:
        st.metric("Linear impulse |J| [N s]", f"{np.linalg.norm(result['jet_impulse']):.4e}")
    with row2[3]:
        st.metric("Angular impulse |L| [N m s]", f"{np.linalg.norm(result['angular_impulse']):.4e}")

    if has_landed and landing_time is not None and jet.t_on > landing_time:
        st.warning(
            "Jet activates after landing. Decrease t_on, increase initial height, "
            "or increase conveyor speed."
        )
    if hit_offset > 3.0 * jet.sigma:
        st.warning(
            "The initial COM is far from the jet centerline relative to the radial Gaussian width. "
            "The jet may barely interact with the object."
        )

    object_summary = {
        "object name": obj.name,
        "object type": obj.object_type,
        "number of surface points": int(obj.surface_points_body.shape[0]),
        "total area weight [m2]": float(np.sum(obj.area_weights)),
        "mass [kg]": obj.mass,
        "Cd [-]": obj.drag_coefficient,
        "size_x [m]": obj.size_x,
        "size_y [m]": obj.size_y,
        "size_z [m]": obj.size_z,
    }
    if obj.object_type == "rod":
        object_summary["rod length [m]"] = obj.rod_length
        object_summary["rod radius [m]"] = obj.rod_radius

    with st.expander("Object Summary", expanded=False):
        st.write(object_summary)
        st.download_button(
            label="Download parameters as JSON",
            data=parameter_json,
            file_name="simulator_parameters.json",
            mime="application/json",
        )

    st.subheader("Animation Export")

    st.write(
        """
        Generate a GIF preview of the 3D rigid-body motion first.  
        The GIF will be shown on this page before saving.  
        Click **Save GIF to Project Folder** only if you want to save it locally.
        """
    )

    col_anim_1, col_anim_2 = st.columns([1, 2])

    with col_anim_1:
        generate_gif_button = st.button(
            "Generate GIF Preview",
            type="primary",
            key="generate_3d_gif_preview_button",
        )

    with col_anim_2:
        st.caption(
            f"Current settings: max frames = {animation_max_frames}, "
            f"FPS = {animation_fps}, DPI = {animation_dpi}"
        )

    if generate_gif_button:
        try:
            with st.spinner("Generating GIF preview. This may take a moment..."):
                gif_bytes = create_3d_animation_gif(
                    result=result,
                    axis_limits=axis_limits,
                    max_frames=animation_max_frames,
                    fps=animation_fps,
                    dpi=animation_dpi,
                    show_points=show_surface_points,
                )

            st.session_state["last_gif_bytes"] = gif_bytes

            if "last_gif_path" in st.session_state:
                del st.session_state["last_gif_path"]

            st.success("GIF preview generated. Review it below before saving.")

        except Exception as exc:
            st.error(f"Failed to generate GIF preview: {exc}")

    if "last_gif_bytes" in st.session_state:
        st.subheader("GIF Preview")

        st.image(
            st.session_state["last_gif_bytes"],
            caption="3D rigid-body motion preview",
        )

        col_save_1, col_save_2 = st.columns(2)

        with col_save_1:
            st.download_button(
                label="Download GIF",
                data=st.session_state["last_gif_bytes"],
                file_name="simulator_3d_animation.gif",
                mime="image/gif",
            )

        with col_save_2:
            save_gif_button = st.button(
                "Save GIF to Project Folder",
                key="save_3d_gif_to_project_folder_button",
            )

        if save_gif_button:
            try:
                gif_path = save_gif_to_project_folder(
                    st.session_state["last_gif_bytes"],
                    filename="simulator_3d_animation.gif",
                )

                st.session_state["last_gif_path"] = str(gif_path)
                st.success(f"GIF saved to: {gif_path}")

            except Exception as exc:
                st.error(f"Failed to save GIF: {exc}")

    if "last_gif_path" in st.session_state:
        st.info(f"Last saved GIF path: {st.session_state['last_gif_path']}")

    st.subheader("2D Projections")
    fig_xy = plot_xy_landing_map(
        result,
        target,
        axis_limits=axis_limits,
    )
    render_matplotlib_figure(fig_xy, stretch=True)

    fig_xz = plot_xz_projection(
        result,
        target,
        axis_limits=axis_limits,
    )
    render_matplotlib_figure(fig_xz, stretch=True)

    fig_yz = plot_yz_projection_with_jet(
        result,
        jet,
        axis_limits=axis_limits,
    )
    render_matplotlib_figure(fig_yz, stretch=True)

    st.subheader("Week 2 Analysis")

    st.write(
        """
        This section summarizes the result and provides analysis tools inspired by the
        Week 2 HTML analyzer: recommended jet timing, hit-offset sensitivity,
        noise sensitivity, and timing-performance maps. Heavy analyses run only
        when their buttons are clicked.
        """
    )

    st.markdown("### Summary Metrics")

    metric_cols = st.columns(4)

    if has_landed and landing_position is not None:
        with metric_cols[0]:
            st.metric("Landing x [m]", f"{landing_position[0]:.4f}")
        with metric_cols[1]:
            st.metric("Landing y [m]", f"{landing_position[1]:.4f}")
        with metric_cols[2]:
            st.metric("Landing z [m]", f"{landing_position[2]:.4f}")
        with metric_cols[3]:
            st.metric("Landing time [s]", f"{landing_time:.4f}")
    else:
        with metric_cols[0]:
            st.metric("Final x [m]", f"{final_position[0]:.4f}")
        with metric_cols[1]:
            st.metric("Final y [m]", f"{final_position[1]:.4f}")
        with metric_cols[2]:
            st.metric("Final z [m]", f"{final_position[2]:.4f}")
        with metric_cols[3]:
            st.metric("Final time [s]", f"{final_time:.4f}")

    metric_cols_2 = st.columns(4)

    with metric_cols_2[0]:
        st.metric("Success", "YES" if success else "NO")
    with metric_cols_2[1]:
        st.metric("Linear impulse |J| [N s]", f"{np.linalg.norm(result['jet_impulse']):.4e}")
    with metric_cols_2[2]:
        st.metric("Angular impulse |L| [N m s]", f"{np.linalg.norm(result['angular_impulse']):.4e}")
    with metric_cols_2[3]:
        st.metric("Max angular speed [rad/s]", f"{result['max_angular_speed']:.4f}")

    st.markdown("### Surface Point 3D Analysis")

    st.write(
        """
        Dedicated static 3D view of object surface points at the start of the simulation.
        """
    )

    fig_surface_points_3d = build_surface_points_analysis_plot(
        result=result,
    )
    st.plotly_chart(fig_surface_points_3d, use_container_width=True, config={"displaylogo": False})
    rendered_surface_count = int(obj.surface_points_body.shape[0])
    unique_surface_count = int(
        np.unique(np.round(obj.surface_points_body, 12), axis=0).shape[0]
    )
    st.caption(
        f"Rendered surface samples: {rendered_surface_count} "
        f"(unique body-coordinate samples: {unique_surface_count})"
    )
    if rendered_surface_count < 500:
        st.warning(
            "This result appears to use an older surface-sampling setup (<500 points). "
            "Click Run Simulation again to regenerate with the current 560-point setting."
        )

    st.markdown("### Gaussian Jet Field 3D")

    st.write(
        """
        This plot shows the directional Gaussian jet plume as a 3D forward-only
        field envelope. The plume exists only in the downstream direction
        (`s >= 0`) and decays axially away from the jet center.
        """
    )

    fig_jet_field_3d = build_gaussian_jet_field_3d_plot(
        result=result,
        jet=jet,
    )

    st.plotly_chart(fig_jet_field_3d, use_container_width=True, config={"displaylogo": False})


    st.markdown("### Angular Dynamics Check")

    st.write(
        """
        Use these plots to check whether the object keeps rotating after the jet turns off.  
        The torque shown here is **jet torque only**. After the jet turns off, jet torque should
        usually drop near zero, but angular velocity should not be forcibly reset to zero.
        If the angular velocity becomes flat after jet shutoff, that can simply mean the object is
        continuing to rotate at an approximately constant angular speed.
        """
    )

    fig_omega = plot_angular_velocity_and_speed(result)
    fig_torque = plot_jet_torque_and_magnitude(result)

    render_matplotlib_figure(fig_omega, stretch=True)
    render_matplotlib_figure(fig_torque, stretch=True)

    angular_diag_df = plot_angular_diagnostics_summary(result)
    if not angular_diag_df.empty:
        st.dataframe(angular_diag_df, use_container_width=True)

    st.markdown("#### Body-Axis Direction Change")

    st.write(
        """
        This plot tracks how much each body-fixed direction has rotated away from
        its initial direction. For a plate, the three directions correspond to the
        normal directions of the three opposite face pairs. For a rod, body x is
        the rod length direction, while body y and body z are two radial directions.
        """
    )

    fig_body_axis_change = plot_body_axis_relative_angle_history(result)
    render_matplotlib_figure(fig_body_axis_change)

    body_axis_summary_df = summarize_body_axis_relative_angle_history(result)
    if not body_axis_summary_df.empty:
        st.dataframe(body_axis_summary_df, use_container_width=True)
        
    st.markdown("#### Rotational Invariant Check")

    st.write(
        """
        These plots help check whether the post-jet rotation is physically reasonable.  
        After the jet turns off, if there is no external rotational damping or contact torque,
        angular momentum magnitude and rotational kinetic energy should remain approximately
        constant. If they grow continuously, the rotation integration may have numerical drift.
        """
    )

    fig_h, fig_energy = plot_rotational_invariant_history(result)

    render_matplotlib_figure(fig_h, stretch=True)
    render_matplotlib_figure(fig_energy, stretch=True)

    rotational_invariant_df = summarize_rotational_invariants(result)
    if not rotational_invariant_df.empty:
        st.dataframe(rotational_invariant_df, use_container_width=True)

    st.markdown("### Recommended Jet Timing")

    timing_info = recommend_jet_timing(result)

    if timing_info["ok"]:
        timing_cols = st.columns(4)
        with timing_cols[0]:
            st.metric("Recommended t_on [s]", f"{timing_info['recommended_t_on']:.4f}")
        with timing_cols[1]:
            st.metric("Peak overlap time [s]", f"{timing_info['t_peak']:.4f}")
        with timing_cols[2]:
            st.metric("Current t_on [s]", f"{timing_info['current_t_on']:.4f}")
        with timing_cols[3]:
            delta_t_on = timing_info["current_t_on"] - timing_info["recommended_t_on"]
            st.metric("Current - recommended [s]", f"{delta_t_on:.4f}")

        st.info(timing_info["message"])
    else:
        st.warning(timing_info["message"])

    influence_fig = plot_influence_score(timing_info)
    render_matplotlib_figure(influence_fig)

    st.markdown("### Hit-Offset Sensitivity")

    hit_cols = st.columns(4)
    with hit_cols[0]:
        offset_axis = st.selectbox(
            "Offset axis",
            options=["x", "y", "z"],
            index=1,
            key="analysis_offset_axis",
        )
    with hit_cols[1]:
        offset_span = st.number_input(
            "Offset span [m]",
            min_value=0.0,
            max_value=1.0,
            value=0.12,
            step=0.01,
            key="analysis_offset_span",
        )
    with hit_cols[2]:
        offset_samples = st.number_input(
            "Number of offset samples",
            min_value=3,
            max_value=51,
            value=17,
            step=2,
            key="analysis_offset_samples",
        )
    with hit_cols[3]:
        st.write("")
        st.write("")
        run_hit_offset_button = st.button(
            "Run Hit-Offset Sensitivity",
            key="run_hit_offset_sensitivity_button",
        )

    if run_hit_offset_button:
        with st.spinner("Running hit-offset sensitivity analysis..."):
            hit_df = run_hit_offset_sensitivity(
                offset_axis=offset_axis,
                offset_span=offset_span,
                n_samples=offset_samples,
            )
        st.session_state["hit_offset_df"] = hit_df
        st.session_state["hit_offset_axis"] = offset_axis

    if "hit_offset_df" in st.session_state:
        hit_df = st.session_state["hit_offset_df"]
        stored_axis = st.session_state.get("hit_offset_axis", offset_axis)
        hit_summary = summarize_hit_offset_sensitivity(hit_df)

        st.markdown("#### Result Summary")
        hit_summary_cols = st.columns(4)
        with hit_summary_cols[0]:
            st.metric("Success count", f"{hit_summary['success_count']} / {hit_summary['total_cases']}")
        with hit_summary_cols[1]:
            st.metric("Success rate", f"{100.0 * hit_summary['success_rate']:.1f}%")
        with hit_summary_cols[2]:
            if np.isfinite(hit_summary["best_offset_m"]):
                st.metric("Best offset [m]", f"{hit_summary['best_offset_m']:.4f}")
            else:
                st.metric("Best offset [m]", "N/A")
        with hit_summary_cols[3]:
            if np.isfinite(hit_summary["landing_x_span_m"]):
                st.metric("Landing x span [m]", f"{hit_summary['landing_x_span_m']:.4f}")
            else:
                st.metric("Landing x span [m]", "N/A")

        if np.isfinite(hit_summary["success_offset_band_m"]):
            st.info(
                "Success offset band: "
                f"[{hit_summary['success_offset_min_m']:.4f}, {hit_summary['success_offset_max_m']:.4f}] m "
                f"(width {hit_summary['success_offset_band_m']:.4f} m)."
            )
        else:
            st.info("No successful offset sample in the current sweep.")

        fig_h1, fig_h2, fig_h3 = plot_hit_offset_sensitivity(hit_df, stored_axis)
        render_matplotlib_figure(fig_h1, stretch=True)
        render_matplotlib_figure(fig_h2, stretch=True)
        render_matplotlib_figure(fig_h3, stretch=True)
        st.dataframe(hit_df, use_container_width=True)

    st.markdown("### Noise Sensitivity")

    noise_cols = st.columns(3)
    with noise_cols[0]:
        noise_trials = st.number_input(
            "Number of noise trials",
            min_value=1,
            max_value=200,
            value=30,
            step=1,
            key="analysis_noise_trials",
        )
    with noise_cols[1]:
        analysis_noise_std = st.number_input(
            "Analysis noise std [-]",
            min_value=0.0,
            max_value=1.0,
            value=max(float(noise_std), 0.05),
            step=0.01,
            key="analysis_noise_std",
        )
    with noise_cols[2]:
        st.write("")
        st.write("")
        run_noise_button = st.button(
            "Run Noise Sensitivity",
            key="run_noise_sensitivity_button",
        )

    if run_noise_button:
        with st.spinner("Running noise sensitivity analysis..."):
            noise_df = run_noise_sensitivity(
                n_trials=noise_trials,
                analysis_noise_std=analysis_noise_std,
            )
            noise_summary = summarize_noise_sensitivity(noise_df)
        st.session_state["noise_df"] = noise_df
        st.session_state["noise_summary"] = noise_summary

    if "noise_df" in st.session_state:
        noise_df = st.session_state["noise_df"]
        noise_summary = st.session_state["noise_summary"]

        noise_metric_cols = st.columns(4)
        with noise_metric_cols[0]:
            st.metric(
                "Success count",
                f"{noise_summary['success_count']} / {noise_summary['total_trials']}",
            )
        with noise_metric_cols[1]:
            st.metric("Success rate", f"{100.0 * noise_summary['success_rate']:.1f}%")
        with noise_metric_cols[2]:
            st.metric(
                "Landing x mean +/- std [m]",
                f"{noise_summary['mean_landing_x_m']:.4f} +/- {noise_summary['std_landing_x_m']:.4f}",
            )
        with noise_metric_cols[3]:
            st.metric(
                "Landing y mean +/- std [m]",
                f"{noise_summary['mean_landing_y_m']:.4f} +/- {noise_summary['std_landing_y_m']:.4f}",
            )

        st.info(
            "Noise summary: "
            f"{noise_summary['success_count']}/{noise_summary['total_trials']} succeeded "
            f"({100.0 * noise_summary['success_rate']:.1f}%), "
            f"landing x = {noise_summary['mean_landing_x_m']:.4f} +/- {noise_summary['std_landing_x_m']:.4f} m."
        )

        fig_n1, fig_n2 = plot_noise_sensitivity(noise_df)
        render_matplotlib_figure(fig_n1, stretch=True)
        render_matplotlib_figure(fig_n2, stretch=True)
        st.dataframe(noise_df, use_container_width=True)

    st.markdown("### Timing Performance Map")

    timing_cols_2 = st.columns(5)
    with timing_cols_2[0]:
        timing_grid_size = st.number_input(
            "Timing grid size",
            min_value=3,
            max_value=25,
            value=8,
            step=1,
            key="analysis_timing_grid_size",
        )
    with timing_cols_2[1]:
        t_on_half_width = st.number_input(
            "t_on scan half-width [s]",
            min_value=0.0,
            max_value=2.0,
            value=0.25,
            step=0.05,
            key="analysis_t_on_half_width",
        )
    with timing_cols_2[2]:
        duration_min_factor = st.number_input(
            "Duration min factor",
            min_value=0.0,
            max_value=5.0,
            value=0.25,
            step=0.05,
            key="analysis_duration_min_factor",
        )
    with timing_cols_2[3]:
        duration_max_factor = st.number_input(
            "Duration max factor",
            min_value=0.05,
            max_value=5.0,
            value=1.80,
            step=0.05,
            key="analysis_duration_max_factor",
        )
    with timing_cols_2[4]:
        st.write("")
        st.write("")
        run_timing_button = st.button(
            "Run Timing Map",
            key="run_timing_map_button",
        )

    if run_timing_button:
        with st.spinner("Running timing performance map..."):
            timing_map_result = run_timing_map(
                grid_size=timing_grid_size,
                t_on_half_width=t_on_half_width,
                duration_min_factor=duration_min_factor,
                duration_max_factor=duration_max_factor,
            )
        st.session_state["timing_map_result"] = timing_map_result

    if "timing_map_result" in st.session_state:
        timing_map_result = st.session_state["timing_map_result"]
        timing_summary = summarize_timing_map(timing_map_result)

        st.markdown("#### Result Summary")
        timing_summary_cols = st.columns(4)
        with timing_summary_cols[0]:
            st.metric(
                "Success count",
                f"{timing_summary['success_count']} / {timing_summary['total_cases']}",
            )
        with timing_summary_cols[1]:
            st.metric("Success rate", f"{100.0 * timing_summary['success_rate']:.1f}%")
        with timing_summary_cols[2]:
            if np.isfinite(timing_summary["landing_x_min_m"]) and np.isfinite(timing_summary["landing_x_max_m"]):
                st.metric(
                    "Landing x range [m]",
                    f"{timing_summary['landing_x_min_m']:.4f} to {timing_summary['landing_x_max_m']:.4f}",
                )
            else:
                st.metric("Landing x range [m]", "N/A")
        with timing_summary_cols[3]:
            if np.isfinite(timing_summary["best_t_on_s"]) and np.isfinite(timing_summary["best_duration_s"]):
                st.metric(
                    "Best (t_on, duration) [s]",
                    f"{timing_summary['best_t_on_s']:.3f}, {timing_summary['best_duration_s']:.3f}",
                )
            else:
                st.metric("Best (t_on, duration) [s]", "N/A")

        if np.isfinite(timing_summary["best_landing_x_m"]):
            st.info(
                "Best successful landing x in map: "
                f"{timing_summary['best_landing_x_m']:.4f} m "
                f"(target center {(0.5 * (target_x_min + target_x_max)):.4f} m)."
            )
        else:
            st.info("No successful timing pair in the current map.")

        fig_t1, fig_t2 = plot_timing_map(timing_map_result)
        render_matplotlib_figure(fig_t1, stretch=True)
        render_matplotlib_figure(fig_t2, stretch=True)
        st.dataframe(timing_map_result["dataframe"], use_container_width=True)

    st.subheader("Trajectory Data")

    st.dataframe(df.head(30), use_container_width=True)

    st.download_button(
        label="Download trajectory data as CSV",
        data=csv_data,
        file_name="simulator_trajectory.csv",
        mime="text/csv",
    )


else:
    st.info("Adjust the sliders in the sidebar and click **Run 3D Simulation**.")

    st.subheader("Recommended Starting Values")

    st.write(
        {
            "object type": "plate",
            "mass [kg]": 0.050,
            "plate/irregular size [m]": [0.10, 0.10, 0.01],
            "rod length/radius [m]": [0.15, 0.025],
            "initial COM position [m]": [0.0, 0.0, 0.20],
            "initial COM velocity [m/s]": [1.0, 0.0, 0.0],
            "initial orientation [roll, pitch, yaw] [deg]": [0.0, 0.0, 0.0],
            "jet Umax [m/s]": 25.0,
            "jet center [x, y, z] [m]": [0.10, 0.0, 0.12],
            "jet Gaussian sigma [m]": 0.08,
            "jet axial decay lambda [m]": 0.35,
            "jet angle elevation [deg]": 45.0,
            "jet azimuth [deg]": 0.0,
            "jet t_on [s]": 0.12,
            "jet duration [s]": 0.15,
            "conveyor length [m]": 0.15,
            "free-fall start offset [m]": 0.03,
            "target x region [m]": [0.30, 0.80],
            "fixed plot axes": {
                "x": [-0.10, 1.50],
                "y": [-0.50, 0.50],
                "z": [0.00, 0.60],
            },
            "animation": {
                "max frames": 80,
                "fps": 12,
                "dpi": 100,
            },
        }
    )
