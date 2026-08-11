#!/usr/bin/env python3
"""Create an IEEE-style 2x2 boxplot from genuine MC episode CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("E_co_time_s", r"$E_{\mathrm{co\!-\!time}}$ (s)"),
    ("E_n_g", r"$E_n$ (g)"),
    ("E_miss_m", r"$E_{miss}$ (m)"),
    ("E_t_s", r"$E_t$ (s)"),
]
COLORS = ["#0072B2", "#D55E00"]


def load_successful(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["interception_success"])]
    data = {}
    for key, _ in METRICS:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        data[key] = values[np.isfinite(values)]
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case1", type=Path, required=True)
    parser.add_argument("--case2", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    case_data = [load_successful(args.case1), load_successful(args.case2)]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # Match the manuscript's V10 Monte Carlo layout: one compact
    # double-column row with the four terminal metrics.
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.25))
    for panel, (axis, (key, ylabel)) in enumerate(zip(axes.flat, METRICS)):
        values = [case_data[0][key], case_data[1][key]]
        boxes = axis.boxplot(
            values,
            labels=["Case 1", "Case 2"],
            widths=0.48,
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 3.2},
            medianprops={"color": "black", "linewidth": 1.15},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
            flierprops={"marker": "o", "markersize": 2.2, "markerfacecolor": "none", "markeredgecolor": "#666666", "alpha": 0.55},
        )
        for patch, color in zip(boxes["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
            patch.set_alpha(0.65)
            patch.set_linewidth(0.8)
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", linestyle="--", linewidth=0.35, color="#B8B8B8", alpha=0.65)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(
            -0.17,
            1.03,
            f"({chr(97 + panel)})",
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
        )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.22, top=0.94, wspace=0.52)
    stem = args.outdir / "two_defender_failure_mc_boxplots"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
