#!/usr/bin/env python3
"""Case-1 four-metric ablation boxplot (1x4 horizontal) for Comment 2.6.

Reads the already-recomputed per-episode ablation metrics (the same episode
table that feeds the Comment 3.3 figure, so the two responses stay consistent)
and draws the four guidance metrics of the paper -- E_co-time, E_n, E_miss, E_t
-- for the four architectural variants, restricted to Case 1 (the scenario that
matches the manuscript's reported operating point). Boxes are over the
strictly-complete Case-1 episodes, i.e. the successful trials, exactly as the
manuscript Monte-Carlo boxplots are conditioned.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VARIANTS = ["full", "no_trust", "no_gru", "no_attention_residual"]
DISPLAY = {
    "full": "ART-MAPPO",
    "no_trust": "w/o Trust",
    "no_gru": "w/o GRU",
    "no_attention_residual": "w/o A-R",
}
COLORS = {
    "full": "#0072B2",
    "no_trust": "#D55E00",
    "no_gru": "#009E73",
    "no_attention_residual": "#CC79A7",
}
METRICS = [
    ("E_co_time_s", r"$E_{\mathrm{co\mathrm{-}time}}$ (s)", "(a)"),
    ("E_n_terminal_sample_g", r"$E_n$ (g)", "(b)"),
    ("E_miss_m", r"$E_{\mathrm{miss}}$ (m)", "(c)"),
    ("E_t_s", r"$E_t$ (s)", "(d)"),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 10,
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "text.usetex": False,
        }
    )


def to_float(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("."))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    configure_style()

    rows = list(csv.DictReader(args.input.open(newline="", encoding="utf-8")))
    # Case-1 successful trials: strictly-complete-group episodes.
    kept = [
        r
        for r in rows
        if r["case"] == "case1" and r["strict_all_8_groups_complete"] == "1"
    ]

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.05), constrained_layout=True)
    for ax, (col, label, tag) in zip(axes, METRICS):
        values = [
            np.asarray(
                [to_float(r[col]) for r in kept if r["variant"] == v], dtype=float
            )
            for v in VARIANTS
        ]
        values = [v[np.isfinite(v)] for v in values]
        bp = ax.boxplot(
            values,
            widths=0.58,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 3.2,
            },
            medianprops={"color": "black", "linewidth": 1.4},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
            flierprops={"marker": "o", "markersize": 2, "alpha": 0.28},
        )
        for box, v in zip(bp["boxes"], VARIANTS):
            box.set_facecolor(COLORS[v])
            box.set_alpha(0.68)
            box.set_linewidth(0.8)
        ax.set_ylabel(label, fontsize=9.5)
        ax.set_xticks(range(1, len(VARIANTS) + 1))
        ax.set_xticklabels(
            [DISPLAY[v] for v in VARIANTS], fontsize=6.8, rotation=20, ha="right"
        )
        ax.grid(True, axis="y", linestyle="--", linewidth=0.25, color="#CCCCCC", alpha=0.55)
        despine(ax)
        ax.set_title(tag, fontsize=9, loc="center")

    stem = args.outdir / "ablation_case1_metrics_box4"
    fig.savefig(f"{stem}.pdf", format="pdf")
    fig.savefig(f"{stem}.png", dpi=600)
    plt.close(fig)
    counts = ",".join(
        str(sum(1 for r in kept if r["variant"] == v)) for v in VARIANTS
    )
    print(f"wrote {stem}.pdf from Case-1 successful trials (N per variant: {counts})")


if __name__ == "__main__":
    main()
