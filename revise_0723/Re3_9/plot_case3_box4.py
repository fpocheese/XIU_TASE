#!/usr/bin/env python3
"""Case-3 bang-bang four-metric boxplot in the paper's V9 figure style.

The typography and drawing parameters intentionally follow
``ieee_plot_v9_tase.py::plot_mc_boxplot_compare`` so this figure is visually
consistent with ``sin_mc_boxplot_compare.png``.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DOUBLE_COL_WIDTH = 7.16
SINGLE_COL_WIDTH = 3.5
COLOR_MAPPO = "#0072B2"


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(rows, key):
    return np.asarray([float(r[key]) for r in rows], dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.input)
    if not rows:
        raise RuntimeError("empty Case-3 metrics CSV")
    args.outdir.mkdir(parents=True, exist_ok=True)
    success = np.asarray(
        [r["mission_success"].strip().lower() in {"1", "true"} for r in rows]
    )
    successful_rows = [r for r, ok in zip(rows, success) if ok]
    if not successful_rows:
        raise RuntimeError("no successful Case-3 trials in metrics CSV")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
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
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "text.usetex": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(
        1, 4, figsize=(DOUBLE_COL_WIDTH, SINGLE_COL_WIDTH * 0.6)
    )

    metrics = [
        ("E_co_time_s", r"$E_{co-time}$ (s)"),
        ("E_n_g", r"$E_n$ (g)"),
        ("E_miss_m", r"$E_{miss}$ (m)"),
        ("E_t_s", r"$E_t$ (s)"),
    ]
    for ax, (key, ylabel) in zip(axes, metrics):
        data = as_float(successful_rows, key)
        data = data[np.isfinite(data)]
        if data.size == 0:
            ax.text(0.5, 0.5, "No successful trials", ha="center", va="center", transform=ax.transAxes)
        else:
            bp = ax.boxplot(
                [data],
                labels=["Case 3"],
                widths=0.5,
                patch_artist=True,
                showfliers=True,
                medianprops={"color": "black", "linewidth": 1.5},
                flierprops={"marker": "o", "markersize": 2, "alpha": 0.3},
            )
            bp["boxes"][0].set_facecolor(COLOR_MAPPO)
            bp["boxes"][0].set_alpha(0.6)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.2,
                color="#CCCCCC", alpha=0.5)
        # Keep the complete rectangular frame used by the V9 comparison plots.
        # Set every spine explicitly so an external Matplotlib style cannot
        # suppress the top or right border.
        ax.set_frame_on(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)
            spine.set_zorder(10)

    plt.tight_layout()
    stem = args.outdir / "case3_bangbang_metrics_box4"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"))
    plt.close(fig)
    print("wrote", stem.with_suffix(".pdf"), "from", len(successful_rows), "cooperative-success trials")


if __name__ == "__main__":
    main()
