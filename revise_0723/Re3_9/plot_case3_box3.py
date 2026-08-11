#!/usr/bin/env python3
"""Case-4 bang-bang four-metric box figure (1x4) for the generalization study."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]

def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def as_float(rows, key):
    return np.asarray([float(r[key]) for r in rows], dtype=float)

def main():
    data_csv = Path("/home/uav/00gao_xueshu/DT_PAPER/XIU_tase_paper_V1/revise_0723/Re_newcase/case4_bangbang/data_n100_clean/episodes.csv")
    outdir   = Path("/home/uav/00gao_xueshu/DT_PAPER/XIU_tase_paper_V1/revise_0723/Re3_9")
    rows = read_rows(data_csv)
    # all 100 episodes are mission_success; use all
    ok_rows = [r for r in rows if r.get("mission_success","").strip().lower() in ("1","true")]
    print(f"success rows: {len(ok_rows)}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.05), constrained_layout=True)

    metrics = [
        ("E_co_time_s",  r"$E_{\mathrm{co{-}time}}$ (s)", "(a)"),
        ("E_n_g",        r"$E_n$ (g)",                         "(b)"),
        ("E_miss_m",     r"$E_{\mathrm{miss}}$ (m)",           "(c)"),
        ("E_t_s",        r"$E_t$ (s)",                         "(d)"),
    ]
    for idx, (ax, (key, ylabel, panel)) in enumerate(zip(axes, metrics)):
        data = as_float(ok_rows, key)
        data = data[np.isfinite(data)]
        bp = ax.boxplot([data], widths=0.42, patch_artist=True, showfliers=True,
            medianprops={"color":"black","linewidth":1.2},
            whiskerprops={"linewidth":0.8}, capprops={"linewidth":0.8},
            boxprops={"linewidth":0.8},
            flierprops={"markersize":2.3,"markerfacecolor":"none","markeredgecolor":"0.35"})
        bp["boxes"][0].set_facecolor(COLORS[idx])
        bp["boxes"][0].set_alpha(0.72)
        ax.scatter([1],[np.mean(data)], marker="D", s=16, color="white",
                   edgecolor="black", linewidth=0.6, zorder=4)
        ax.set_xticks([1], ["Case 3"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="0.88", lw=0.55)
        ax.text(-0.30, 1.04, panel, transform=ax.transAxes, fontweight="bold")

    for ax in axes:
        ax.tick_params(direction="in", length=3.0, width=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    stem = outdir / "case4_bangbang_metrics_box4"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("wrote", stem.with_suffix(".pdf"))

if __name__ == "__main__":
    main()
