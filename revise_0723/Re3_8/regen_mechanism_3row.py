#!/usr/bin/env python3
"""Regenerate failure_mechanism_diagnostics_v10 with 3 rows (drop overload row).

Rows: (top) range d, (mid) range-rate d_dot, (bottom) LOS angular rate.
The original 4th row (|n_y|,|n_z| overload command) is removed per revision.
Reads the local frozen-policy npz trajectories; styling mirrors the original
analyze_failure_mechanisms.py so the curves are identical to the verified data.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CF_ROOT = (
    HERE.parent
    / "Re_xiaorong/V2/FailureCases_Case1_Case2_20260730"
    / "artifacts_remote_v2/failure_cause_counterfactuals_v2"
)
OUT_DIRS = [
    HERE,  # Re3_8/  (canonical copy used by manuscript + letter)
    HERE.parent
    / "Re_xiaorong/V2/FailureCases_Case1_Case2_20260730"
    / "reviewer_response_assets/mechanism",  # keep source-draft folder in sync
]
DT = 0.05
HIT_RADIUS_M = 3.0
CASES = {"case1": 76048, "case2": 77008}
MISSED_DEFENDER = {"case1": 5, "case2": 1}
ASSIGNMENT = np.array(
    [20, 21, 22, 23, 24, 25, 26, 27, 20, 21,
     22, 23, 24, 25, 26, 27, 20, 21, 22, 23],
    dtype=int,
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "mathtext.fontset": "stix",
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.3,
        "grid.linestyle": "--",
        "grid.color": "#CCCCCC",
        "grid.alpha": 0.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.framealpha": 0.95,
        "legend.edgecolor": "0.7",
        "legend.fancybox": False,
        "legend.frameon": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "text.usetex": False,
    }
)


def vec_norm(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=-1)


def load_curves(case: str) -> dict:
    folder = CF_ROOT / case / "observed_boundary" / case
    z = np.load(folder / f"{case}_selected_episode.npz")
    md = MISSED_DEFENDER[case]
    ti = int(ASSIGNMENT[md] - 20)
    p_d = np.asarray(z["rep_def"], dtype=float)[:, md]
    p_a = np.asarray(z["rep_att"], dtype=float)[:, ti]
    v_d = np.gradient(p_d, DT, axis=0)
    v_a = np.gradient(p_a, DT, axis=0)
    rel = p_a - p_d
    rel_vel = v_a - v_d
    distance = vec_norm(rel)
    safe_d = np.maximum(distance, 1e-12)
    range_rate = np.einsum("ij,ij->i", rel, rel_vel) / safe_d
    los_rate = vec_norm(np.cross(rel, rel_vel)) / (safe_d ** 2)
    time = np.arange(len(distance)) * DT
    closest = int(np.argmin(distance))
    return {
        "time": time,
        "closest": closest,
        "distance": distance,
        "range_rate": range_rate,
        "los_rate": los_rate,
    }


def main() -> None:
    plot_data = {c: load_curves(c) for c in CASES}
    colors = {"case1": "#0072B2", "case2": "#D55E00"}
    fig, axes = plt.subplots(3, 2, figsize=(7.16, 5.35), sharex="col")
    for col, case in enumerate(("case1", "case2")):
        data = plot_data[case]
        center = data["closest"]
        rel_t = data["time"] - data["time"][center]
        mask = (rel_t >= -0.5) & (rel_t <= 0.5)
        color = colors[case]

        axes[0, col].plot(rel_t[mask], data["distance"][mask], color=color)
        axes[0, col].axhline(
            HIT_RADIUS_M, color="#000000", ls="--", lw=1.0,
            label=r"$d_{\rm kill}=3$ m",
        )
        axes[0, col].scatter(
            [0.0], [data["distance"][center]], color="#D55E00",
            marker="X", s=35, zorder=5, label="Closest approach",
        )
        axes[0, col].set_ylabel(r"$d$ (m)")
        axes[0, col].set_title(f"Case {col + 1}", pad=2)

        axes[1, col].plot(rel_t[mask], data["range_rate"][mask], color=color)
        axes[1, col].axhline(0.0, color="#000000", ls="--", lw=1.0)
        axes[1, col].set_ylabel(r"$\dot d$ (m/s)")

        axes[2, col].plot(rel_t[mask], data["los_rate"][mask], color="#009E73")
        axes[2, col].set_ylabel(r"$|\dot{\lambda}|$ (rad/s)")
        axes[2, col].set_xlabel("Time relative to closest approach (s)")

        for row in range(3):
            axes[row, col].axvline(0.0, color="#777777", ls=":", lw=0.8)
            axes[row, col].grid(True)

    axes[0, 0].legend(loc="upper right", fontsize=7)
    labels = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")
    for index, (label, ax) in enumerate(zip(labels, axes.flat)):
        y_position = 0.08 if index in (0, 1) else 0.96
        ax.text(
            0.02, y_position, label, transform=ax.transAxes,
            ha="left", va="top", fontweight="bold",
        )
    fig.align_ylabels()
    fig.subplots_adjust(hspace=0.12, wspace=0.28)
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 600})):
            fig.savefig(
                out_dir / f"failure_mechanism_diagnostics_v10.{suffix}", **kwargs
            )
    plt.close(fig)
    print("regenerated 3-row mechanism figure in:")
    for out_dir in OUT_DIRS:
        print(" ", out_dir)


if __name__ == "__main__":
    main()
