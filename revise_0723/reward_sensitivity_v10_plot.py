#!/usr/bin/env python3
"""Redraw the archived reward-sensitivity results in the paper's v10 style.

This script is deliberately plot-only: it reads the CSV files produced by
``reward_sensitivity_analysis.py`` and never changes or smooths the numerical
results.  It is therefore safe to rerun without policy inference or training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCALINGS = np.asarray([0.50, 0.75, 1.00, 1.25, 1.50])
CURVE_SPECS = (
    ("distance_curve", "alpha_dist", r"$\alpha_d$", "Relative distance $d$ (m)", "Distance reward"),
    (
        "coordination_curve",
        "w_coord",
        r"$w_{\mathrm{coord}}$",
        r"Time-to-go mismatch $|t_{go,i}-\bar{t}_{go}|$ (s)",
        "Coordination reward",
    ),
    (
        "energy_curve",
        "w_energy",
        r"$w_{\mathrm{energy}}$",
        r"Control effort $n_x^2+n_y^2+n_z^2$",
        "Energy/control penalty",
    ),
)
PARAMETERS = ("w_hit", "w_coord", "w_energy", "alpha_dist")
PARAMETER_LABELS = {
    "w_hit": r"$w_{\mathrm{hit}}$",
    "w_coord": r"$w_{\mathrm{coord}}$",
    "w_energy": r"$w_{\mathrm{energy}}$",
    "alpha_dist": r"$\alpha_d$",
}

# Okabe--Ito plus black for the nominal theoretical curve.  The combination
# remains distinguishable in grayscale through line styles and markers.
THEORY_COLORS = ("#0072B2", "#56B4E9", "#000000", "#E69F00", "#D55E00")
THEORY_STYLES = ("-", "--", "-", "-.", ":")
TRAJECTORY_COLORS = {
    "w_hit": "#0072B2",
    "w_coord": "#D55E00",
    "w_energy": "#009E73",
    "alpha_dist": "#CC79A7",
}
TRAJECTORY_MARKERS = {
    "w_hit": "o",
    "w_coord": "s",
    "w_energy": "D",
    "alpha_dist": "^",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("figures/reward_sensitivity"),
        help="Directory containing reward_sensitivity_data.csv and summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/reward_sensitivity_v10"),
    )
    return parser.parse_args()


def configure_v10_style() -> None:
    """Mirror the serif, high-contrast, uncluttered style used by v10 plots."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Tinos", "Times New Roman", "Times", "Nimbus Roman"],
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.2,
            "lines.linewidth": 1.45,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.6,
            "ytick.major.size": 3.6,
            "xtick.major.width": 0.85,
            "ytick.major.width": 0.85,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_results(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = input_dir / "reward_sensitivity_data.csv"
    summary_path = input_dir / "reward_sensitivity_summary.csv"
    data = pd.read_csv(data_path)
    summary = pd.read_csv(summary_path)
    if data.empty or summary.empty:
        raise ValueError("Sensitivity input CSV is empty.")
    numeric = data.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()[~numeric.isna().to_numpy()]).all():
        raise ValueError("Sensitivity input contains NaN/Inf in populated numeric fields.")
    return data, summary


def theoretical_panel(
    axis: plt.Axes,
    data: pd.DataFrame,
    data_type: str,
    symbol: str,
    xlabel: str,
    ylabel: str,
) -> None:
    panel = data.loc[data["data_type"].eq(data_type)].copy()
    for index, scaling in enumerate(SCALINGS):
        curve = panel.loc[np.isclose(panel["scaling"], scaling)].sort_values("x_value")
        if curve.empty:
            raise ValueError(f"Missing {data_type} data for scaling {scaling}.")
        width = 2.0 if np.isclose(scaling, 1.0) else 1.35
        zorder = 5 if np.isclose(scaling, 1.0) else 2
        label = rf"${scaling:.2f}\,{symbol[1:-1]}^0$"
        axis.plot(
            curve["x_value"],
            curve["reward_value"],
            color=THEORY_COLORS[index],
            linestyle=THEORY_STYLES[index],
            linewidth=width,
            label=label,
            zorder=zorder,
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.margins(x=0.01)


def trajectory_panel(axis: plt.Axes, data: pd.DataFrame) -> None:
    panel = data.loc[data["data_type"].eq("trajectory_summary")].copy()
    for parameter in PARAMETERS:
        curve = panel.loc[panel["parameter"].eq(parameter)].sort_values("scaling")
        if len(curve) != len(SCALINGS):
            raise ValueError(f"Expected five trajectory summaries for {parameter}.")
        x = curve["scaling"].to_numpy(dtype=float)
        mean = curve["mean"].to_numpy(dtype=float)
        ci = curve["ci95"].to_numpy(dtype=float)
        color = TRAJECTORY_COLORS[parameter]
        axis.plot(
            x,
            mean,
            color=color,
            marker=TRAJECTORY_MARKERS[parameter],
            markersize=4.2,
            markeredgewidth=0.6,
            linewidth=1.55,
            label=PARAMETER_LABELS[parameter],
            zorder=3,
        )
        axis.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.18, linewidth=0)

    axis.axhline(0.0, color="0.32", linewidth=0.75, linestyle="--", zorder=1)
    axis.axvline(1.0, color="0.52", linewidth=0.75, linestyle=":", zorder=1)
    axis.set_xticks(SCALINGS)
    axis.set_xlabel(r"Parameter scaling factor $\theta/\theta_0$")
    axis.set_ylabel("Normalized cumulative-reward variation")

    # The sparse hit term dominates the global scale.  This inset uses exactly
    # the same mean and 95% CI values to reveal the smaller responses.
    inset = axis.inset_axes([0.105, 0.555, 0.47, 0.34])
    extent = 0.0
    for parameter in ("w_coord", "w_energy", "alpha_dist"):
        curve = panel.loc[panel["parameter"].eq(parameter)].sort_values("scaling")
        x = curve["scaling"].to_numpy(dtype=float)
        mean = curve["mean"].to_numpy(dtype=float)
        ci = curve["ci95"].to_numpy(dtype=float)
        extent = max(extent, float(np.max(np.abs(mean) + ci)))
        color = TRAJECTORY_COLORS[parameter]
        inset.plot(
            x,
            mean,
            color=color,
            marker=TRAJECTORY_MARKERS[parameter],
            markersize=2.8,
            linewidth=1.0,
        )
        inset.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.18, linewidth=0)
    extent *= 1.16
    inset.set_xlim(0.48, 1.52)
    inset.set_ylim(-extent, extent)
    inset.set_xticks(SCALINGS)
    inset.set_xticklabels([".50", ".75", "1", "1.25", "1.50"])
    inset.tick_params(labelsize=5.8, top=True, right=True, length=2.4, width=0.65)
    inset.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    inset.yaxis.get_offset_text().set_fontsize(5.6)
    inset.axhline(0.0, color="0.42", linewidth=0.55, linestyle="--")
    inset.axvline(1.0, color="0.55", linewidth=0.55, linestyle=":")
    inset.grid(True, color="0.88", linewidth=0.4, linestyle="--", dashes=(2.2, 2.2))
    inset.set_title("Magnified near-zero responses", fontsize=6.2, pad=1.5)
    for spine in inset.spines.values():
        spine.set_linewidth(0.75)


def create_figure(data: pd.DataFrame, output_dir: Path) -> None:
    configure_v10_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.10, 5.05))
    ax_a, ax_b, ax_c, ax_d = axes.flat
    for axis, spec in zip((ax_a, ax_b, ax_c), CURVE_SPECS):
        data_type, _parameter, symbol, xlabel, ylabel = spec
        theoretical_panel(axis, data, data_type, symbol, xlabel, ylabel)
    trajectory_panel(ax_d, data)

    ax_c.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))
    ax_c.yaxis.get_offset_text().set_fontsize(7.2)

    for label, axis in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
        axis.text(
            0.018,
            0.975,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            zorder=10,
        )
        axis.tick_params(top=True, right=True)
        axis.grid(True, color="0.87", linewidth=0.48, linestyle="--", dashes=(2.4, 2.4))
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_linewidth(0.9)

    ax_a.legend(loc="lower left", frameon=False, ncol=2, handlelength=2.5, columnspacing=1.0)
    ax_b.legend(loc="lower left", frameon=False, ncol=2, handlelength=2.5, columnspacing=1.0)
    ax_c.legend(loc="lower left", frameon=False, ncol=2, handlelength=2.5, columnspacing=1.0)
    ax_d.legend(loc="lower right", frameon=False, ncol=2, handlelength=2.1, columnspacing=0.9)

    fig.subplots_adjust(
        left=0.105,
        right=0.988,
        bottom=0.105,
        top=0.985,
        wspace=0.29,
        hspace=0.31,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension, options in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 600}),
    ):
        fig.savefig(
            output_dir / f"reward_sensitivity_v10.{extension}",
            format=extension,
            bbox_inches="tight",
            pad_inches=0.025,
            **options,
        )
    plt.close(fig)


def validate_nominal_zero(data: pd.DataFrame) -> None:
    rows = data.loc[
        data["data_type"].eq("trajectory_summary") & np.isclose(data["scaling"], 1.0)
    ]
    if len(rows) != 4 or not np.allclose(rows["mean"], 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("Nominal scaling factor does not give zero reward variation.")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    data, _summary = load_results(input_dir)
    validate_nominal_zero(data)
    create_figure(data, output_dir)
    print(f"[ok] Read archived data from {input_dir}")
    print("[ok] Nominal trajectory variation is exactly zero for all four parameters")
    print(f"[ok] Saved v10-style PDF/SVG/PNG to {output_dir}")


if __name__ == "__main__":
    main()
