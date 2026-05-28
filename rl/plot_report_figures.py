"""Generate report-ready figures from PPO evaluation results.

Usage (explicit):
    python -m rl.plot_report_figures \\
        --eval-csv outputs/rl_results/eval_50k_boundary_elev0_60.csv \\
        --train-log outputs/rl_results/train_boundary_elev0_60_50k.log \\
        --run-name boundary_elev0_60_50k

Usage (default / backward-compatible):
    python -m rl.plot_report_figures
"""

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd

# ── defaults ───────────────────────────────────────────────────────────────────

_DEFAULT_CSV      = "outputs/rl_results/eval_50k_boundary_elev0_60.csv"
_DEFAULT_LOG      = "outputs/rl_results/train_boundary_elev0_60_50k.log"
_DEFAULT_RUN_NAME = "boundary_elev0_60_50k"
_DEFAULT_OUT_ROOT = "outputs/rl_results/report_figures"
_DEFAULT_TARGET_X = 0.42

# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate report figures from a PPO evaluation CSV + training log."
    )
    p.add_argument(
        "--eval-csv",
        default=_DEFAULT_CSV,
        help="Path to evaluation CSV (required data source).",
    )
    p.add_argument(
        "--train-log",
        default=_DEFAULT_LOG,
        help="Path to training log. If absent, learning-curve figures are skipped.",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help=(
            "Name of this run (used as subfolder). "
            "If omitted, inferred from the eval CSV filename "
            "(e.g. eval_50k_boundary_elev0_60.csv → 50k_boundary_elev0_60)."
        ),
    )
    p.add_argument(
        "--out-root",
        default=_DEFAULT_OUT_ROOT,
        help=f"Root output directory. Default: {_DEFAULT_OUT_ROOT}",
    )
    p.add_argument(
        "--target-x-min",
        type=float,
        default=_DEFAULT_TARGET_X,
        help=f"Success boundary threshold on x axis (m). Default: {_DEFAULT_TARGET_X}",
    )
    return p.parse_args()


def _infer_run_name(csv_path: str) -> str:
    """eval_50k_boundary_elev0_60.csv  →  50k_boundary_elev0_60"""
    stem = os.path.splitext(os.path.basename(csv_path))[0]  # eval_50k_boundary_elev0_60
    if stem.startswith("eval_"):
        return stem[len("eval_"):]
    return stem


# ── helpers ────────────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, out_dir: str, name: str) -> None:
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def _bar_fig(categories, values, xlabel, ylabel, title, color="steelblue"):
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(categories, values, color=color, edgecolor="black", linewidth=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
    fig.tight_layout()
    return fig


def _has(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()

    csv_path  = args.eval_csv
    log_path  = args.train_log
    target_x  = args.target_x_min
    out_root  = args.out_root
    run_name  = args.run_name or (
        _DEFAULT_RUN_NAME
        if csv_path == _DEFAULT_CSV
        else _infer_run_name(csv_path)
    )
    out_dir   = os.path.join(out_root, run_name)

    # ── validate CSV (required) ────────────────────────────────────────────────

    if not os.path.isfile(csv_path):
        print(f"ERROR: eval CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # ── validate log (optional) ────────────────────────────────────────────────

    use_log = os.path.isfile(log_path) if log_path else False
    if log_path and not use_log:
        print(f"  WARNING: train log not found: {log_path} — skipping learning curves")

    os.makedirs(out_dir, exist_ok=True)

    # ── load CSV ───────────────────────────────────────────────────────────────

    df = pd.read_csv(csv_path)
    df["success"] = df["success"].astype(bool)

    # ── parse log ──────────────────────────────────────────────────────────────

    log_steps, log_success, log_reward = [], [], []
    if use_log:
        _pat = re.compile(
            r"\[step\s+(\d+)\]\s+success_rate=([\d.]+)%\s+mean_reward=([-\d.]+)"
        )
        with open(log_path) as fh:
            for line in fh:
                m = _pat.search(line)
                if m:
                    log_steps.append(int(m.group(1)))
                    log_success.append(float(m.group(2)))
                    log_reward.append(float(m.group(3)))
        if not log_steps:
            print(f"  WARNING: no callback lines found in {log_path} — skipping learning curves")

    log_df = pd.DataFrame(
        {"step": log_steps, "success_rate": log_success, "mean_reward": log_reward}
    )

    # ── compute summary stats ──────────────────────────────────────────────────

    n_ep      = len(df)
    suc_rate  = df["success"].mean() * 100
    land_rate = df["has_landed"].mean() * 100 if _has(df, "has_landed") else None
    mean_lx   = df["landing_x"].mean() if _has(df, "landing_x") else None
    mean_umax = df["umax"].mean()       if _has(df, "umax")      else None
    mean_dur  = df["duration"].mean()   if _has(df, "duration")  else None
    mean_elev = df["elevation_deg"].mean() if _has(df, "elevation_deg") else None

    # columns available for action tables
    action_cols = [c for c in ("umax", "duration", "elevation_deg") if _has(df, c)]

    # ── build summary text ─────────────────────────────────────────────────────

    lines = []
    lines.append("=" * 60)
    lines.append("  EVALUATION SUMMARY")
    lines.append("=" * 60)
    lines.append(f"  Run name        : {run_name}")
    lines.append(f"  Eval CSV        : {csv_path}")
    lines.append(f"  Train log       : {log_path if use_log else '(not provided)'}")
    lines.append(f"  Episodes        : {n_ep}")
    lines.append(f"  Success rate    : {suc_rate:.2f}%")
    if land_rate is not None:
        lines.append(f"  Landing rate    : {land_rate:.2f}%")
    if mean_lx is not None:
        lines.append(f"  Mean landing_x  : {mean_lx:.4f} m")
    if mean_umax is not None:
        lines.append(f"  Mean Umax       : {mean_umax:.4f} m/s")
    if mean_dur is not None:
        lines.append(f"  Mean duration   : {mean_dur:.4f} s")
    if mean_elev is not None:
        lines.append(f"  Mean elev_deg   : {mean_elev:.4f} deg")

    if _has(df, "object_type"):
        lines.append("")
        lines.append("  Success rate by object type:")
        for ot, grp in df.groupby("object_type"):
            lines.append(f"    {ot:12s}: {grp['success'].mean()*100:.2f}%")

    if action_cols:
        lines.append("")
        lines.append("  Mean actions — success vs. failure:")
        grp_sf = df.groupby("success")[action_cols].mean()
        grp_sf.index = grp_sf.index.map({True: "success", False: "failure"})
        lines.append(grp_sf.to_string())

    lines.append("=" * 60)
    summary_text = "\n".join(lines)

    print("\n" + summary_text + "\n")

    # ── save summary.txt ───────────────────────────────────────────────────────

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as fh:
        fh.write(summary_text + "\n")
    print(f"  saved: {summary_path}")

    # ── figure 1 — top-view landing map ───────────────────────────────────────

    if _has(df, "landing_x") and _has(df, "landing_y"):
        suc  = df[df["success"]]
        fail = df[~df["success"]]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(fail["landing_x"], fail["landing_y"],
                   c="red",   alpha=0.6, s=25, label="failure", zorder=3)
        ax.scatter(suc["landing_x"],  suc["landing_y"],
                   c="green", alpha=0.7, s=25, label="success", zorder=4)
        ax.axvline(target_x, color="navy", linestyle="--", linewidth=1.2,
                   label=f"target x = {target_x} m")
        ax.set_xlabel("landing_x (m)")
        ax.set_ylabel("landing_y (m)")
        ax.set_title("Top-view landing position map")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        _save(fig, out_dir, "fig_landing_map")
    elif _has(df, "landing_x"):
        print("  WARNING: landing_y missing — skipping top-view map (fig_landing_map)")
    else:
        print("  WARNING: landing_x missing — skipping fig_landing_map")

    # ── figure 2 — landing_x histogram ────────────────────────────────────────

    if _has(df, "landing_x"):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df["landing_x"], bins=20, color="steelblue",
                edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.axvline(target_x, color="red", linestyle="--", linewidth=1.3,
                   label=f"target x = {target_x} m")
        ax.set_xlabel("landing_x (m)")
        ax.set_ylabel("episode count")
        ax.set_title("Landing x distribution")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        _save(fig, out_dir, "fig_landing_x_hist")
    else:
        print("  WARNING: landing_x missing — skipping fig_landing_x_hist")

    # ── figures 3 & 4 — learning curves ───────────────────────────────────────

    if not log_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(log_df["step"], log_df["success_rate"],
                marker="o", markersize=5, color="steelblue", linewidth=1.5)
        ax.set_xlabel("training timestep")
        ax.set_ylabel("success rate (%)")
        ax.set_title("PPO learning curve — success rate")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        _save(fig, out_dir, "fig_learning_curve_success")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(log_df["step"], log_df["mean_reward"],
                marker="o", markersize=5, color="darkorange", linewidth=1.5)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("training timestep")
        ax.set_ylabel("mean episode reward")
        ax.set_title("PPO learning curve — mean reward")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        _save(fig, out_dir, "fig_learning_curve_reward")

    # ── figures 5 — mean action by object type ────────────────────────────────

    if _has(df, "object_type"):
        grp_cols = [c for c in ("umax", "duration", "elevation_deg") if _has(df, c)]
        obj_grp  = df.groupby("object_type")[grp_cols].mean()
        obj_types = obj_grp.index.tolist()

        for col, ylabel, fname in [
            ("umax",          "mean Umax (m/s)",       "fig_mean_umax_by_object"),
            ("duration",      "mean duration (s)",      "fig_mean_duration_by_object"),
            ("elevation_deg", "mean elevation (deg)",   "fig_mean_elevation_deg_by_object"),
        ]:
            if col not in grp_cols:
                print(f"  WARNING: column '{col}' missing — skipping {fname}")
                continue
            fig = _bar_fig(obj_types, obj_grp[col].values,
                           "object type", ylabel, f"Mean {col} by object type")
            _save(fig, out_dir, fname)

    # ── figures 6 — mean action by success/failure ────────────────────────────

    sf_cols = [c for c in ("umax", "duration", "elevation_deg") if _has(df, c)]
    if sf_cols:
        sf_grp = df.groupby("success")[sf_cols].mean()
        sf_grp.index = sf_grp.index.map({True: "success", False: "failure"})
        sf_cats = sf_grp.index.tolist()

        for col, ylabel, fname in [
            ("umax",          "mean Umax (m/s)",      "fig_mean_umax_success_vs_failure"),
            ("duration",      "mean duration (s)",     "fig_mean_duration_success_vs_failure"),
            ("elevation_deg", "mean elevation (deg)",  "fig_mean_elevation_deg_success_vs_failure"),
        ]:
            if col not in sf_cols:
                print(f"  WARNING: column '{col}' missing — skipping {fname}")
                continue
            fig, ax = plt.subplots(figsize=(5, 4))
            colors = ["green" if c == "success" else "red" for c in sf_cats]
            bars = ax.bar(sf_cats, sf_grp[col].values,
                          color=colors, edgecolor="black", linewidth=0.6)
            ax.set_xlabel("outcome")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Mean {col}: success vs. failure")
            ax.grid(axis="y", alpha=0.3)
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
            fig.tight_layout()
            _save(fig, out_dir, fname)

    # ── figure 7 — success rate by object type ────────────────────────────────

    if _has(df, "object_type"):
        srate_by_obj = df.groupby("object_type")["success"].mean() * 100
        fig = _bar_fig(
            srate_by_obj.index.tolist(), srate_by_obj.values,
            "object type", "success rate (%)",
            "Success rate by object type", color="teal",
        )
        _save(fig, out_dir, "fig_success_rate_by_object")

    # ── figure 8 — reward component summary ───────────────────────────────────

    reward_cols = ["reward_success", "reward_undershoot", "reward_overshoot", "reward_energy"]
    present      = [c for c in reward_cols if _has(df, c)]
    missing_rcols = [c for c in reward_cols if not _has(df, c)]

    if missing_rcols:
        print(f"  WARNING: missing reward columns {missing_rcols} — partial/skipped reward plot")

    if present:
        means = df[present].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["green" if v >= 0 else "tomato" for v in means.values]
        bars = ax.bar(means.index, means.values,
                      color=colors, edgecolor="black", linewidth=0.6)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("reward component")
        ax.set_ylabel("mean value")
        ax.set_title("Mean reward components")
        ax.grid(axis="y", alpha=0.3)
        ax.bar_label(bars, fmt="%.4f", padding=2, fontsize=9)
        fig.tight_layout()
        _save(fig, out_dir, "fig_reward_components")
    else:
        print("  WARNING: no reward component columns found — skipping fig_reward_components")

    print(f"\nAll outputs saved to: {out_dir}/\n")


if __name__ == "__main__":
    main()
