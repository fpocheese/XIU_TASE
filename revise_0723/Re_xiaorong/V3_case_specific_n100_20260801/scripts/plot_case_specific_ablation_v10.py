#!/usr/bin/env python3
"""IEEE-TASE V10 plots for the case-specific ART-MAPPO ablation.

Only visual smoothing is applied to per-update training traces.  Monte-Carlo
episode values and boxplot samples are never smoothed or imputed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


DOUBLE = 7.16
SINGLE = 3.5
VARIANTS = ("full", "no_trust", "no_gru", "no_attention_residual")
COLORS = {
    "full": "#0072B2",
    "no_trust": "#D55E00",
    "no_gru": "#009E73",
    "no_attention_residual": "#CC79A7",
}
LABELS = {
    "full": "Full ART-MAPPO",
    "no_trust": "No trust",
    "no_gru": "No GRU",
    "no_attention_residual": "No attention-residual",
}
SHORT = {
    "full": "Full",
    "no_trust": "No trust",
    "no_gru": "No GRU",
    "no_attention_residual": "No A-R",
}
MARKERS = {"full": "o", "no_trust": "s", "no_gru": "^", "no_attention_residual": "D"}
# A 51-update centered moving average is used only for visualization.  The
# recorded returns alternate among episode-completion cohorts in this vector
# environment; a shorter window leaves a dense comb pattern that obscures the
# comparison.  All numerical summaries use the unsmoothed CSV values.
TRAIN_WINDOW = 51


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "mathtext.fontset": "stix",
        "lines.linewidth": 1.45,
        "lines.markersize": 4.5,
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
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "text.usetex": False,
    })


def save(fig, outdir: Path, stem: str):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf")
    fig.savefig(outdir / f"{stem}.svg")
    fig.savefig(outdir / f"{stem}.png", dpi=600)
    plt.close(fig)


def panel(ax, text):
    ax.text(0.02, 0.97, text, transform=ax.transAxes, ha="left", va="top",
            fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.6})


def rolling(values, window=TRAIN_WINDOW):
    return pd.Series(np.asarray(values, float)).rolling(window, min_periods=1, center=True).mean().to_numpy()


def load_training(root: Path):
    frames = []
    for path in sorted(root.glob("*/*/seed*/training_metrics.csv")):
        variant, case, seed = path.parts[-4:-1]
        if variant not in VARIANTS or case not in ("case1", "case2"):
            continue
        data = pd.read_csv(path).sort_values("environment_steps")
        data["variant"] = variant
        data["case"] = case
        data["seed"] = int(seed.replace("seed", ""))
        frames.append(data)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_one_training_axis(ax, subset, metric, ylabel):
    for variant in VARIANTS:
        data = subset[subset.variant == variant].sort_values("environment_steps")
        if data.empty:
            continue
        x = data.environment_steps.to_numpy(float) / 1000.0
        y = rolling(data[metric].to_numpy(float))
        ax.plot(x, y, color=COLORS[variant], marker=MARKERS[variant],
                markevery=max(1, len(x) // 8), label=LABELS[variant])
    ax.set_xlabel(r"Environment steps ($\times 10^3$)")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    ax.grid(True)
    if metric == "value_loss":
        positive = subset[metric].to_numpy(float)
        positive = positive[np.isfinite(positive) & (positive > 0)]
        if len(positive) and positive.max() / positive.min() > 100:
            ax.set_yscale("log")


def plot_training(root: Path, outdir: Path):
    data = load_training(root)
    stems = []
    definitions = (("mean_episode_return", "Episode return", "reward"),
                   ("value_loss", "Critic loss", "critic_loss"),
                   ("entropy", "Policy entropy", "policy_entropy"))
    for case in ("case1", "case2"):
        subset = data[data.case == case]
        if subset.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.30), constrained_layout=True)
        for ax, (metric, ylabel, _), letter in zip(axes, definitions, ("(a)", "(b)", "(c)")):
            plot_one_training_axis(ax, subset, metric, ylabel)
            panel(ax, letter)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=4,
                   columnspacing=0.75, handlelength=1.45)
        stem = f"ablation_training_{case}_v10"
        save(fig, outdir, stem); stems.append(stem)
        for metric, ylabel, suffix in definitions:
            fig, ax = plt.subplots(figsize=(SINGLE, 2.45), constrained_layout=True)
            plot_one_training_axis(ax, subset, metric, ylabel)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2,
                      columnspacing=0.7, handlelength=1.4)
            stem = f"ablation_training_{suffix}_{case}_v10"
            save(fig, outdir, stem); stems.append(stem)
    return stems


def wilson(values):
    values = np.asarray(values, float)
    n = len(values)
    p = float(values.mean())
    z = 1.959963984540054
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return p, max(0.0, center-half), min(1.0, center+half)


def boxplot(ax, groups, variants, ylabel):
    positions = np.arange(1, len(variants)+1)
    nonempty, used_positions, used_colors = [], [], []
    counts = []
    for pos, variant, values in zip(positions, variants, groups):
        clean = np.asarray(values, float)
        clean = clean[np.isfinite(clean)]
        counts.append(len(clean))
        if len(clean):
            nonempty.append(clean); used_positions.append(pos); used_colors.append(COLORS[variant])
    if nonempty:
        bp = ax.boxplot(nonempty, positions=used_positions, widths=0.56, patch_artist=True,
                        showfliers=False, medianprops={"color":"black","linewidth":1.2},
                        whiskerprops={"linewidth":0.8}, capprops={"linewidth":0.8},
                        boxprops={"linewidth":0.8})
        for box, color in zip(bp["boxes"], used_colors):
            box.set_facecolor(color); box.set_alpha(0.62)
    ax.set_xticks(positions, [f"{SHORT[v]}\n($n$={n})" for v, n in zip(variants, counts)])
    ax.set_xlim(0.5, len(variants)+0.5)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    ax.grid(True, axis="y")


def plot_mc(episodes_csv: Path, outdir: Path):
    episodes = pd.read_csv(episodes_csv)
    stems = []
    for case in ("case1", "case2"):
        data = episodes[episodes.case == case]
        variants = [v for v in VARIANTS if v in set(data.variant)]
        if data.empty:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.35), constrained_layout=True)
        for ax, metric, ylabel, letter in (
            (axes[0,0], "target_coverage_success", "Target-coverage rate", "(a)"),
            (axes[0,1], "cooperative_success", "Cooperative-success rate", "(b)")):
            stats = [wilson(data[data.variant == v][metric]) for v in variants]
            means = np.asarray([s[0] for s in stats])
            # Wilson endpoints can differ from an exact 0/1 mean by a few
            # ulps.  Matplotlib rejects tiny negative error lengths, so clip
            # only those roundoff artifacts without changing any rate or CI.
            errors = np.maximum(
                np.asarray(
                    [
                        [s[0] - s[1] for s in stats],
                        [s[2] - s[0] for s in stats],
                    ]
                ),
                0.0,
            )
            x = np.arange(len(variants))
            ax.bar(x, means, yerr=errors, color=[COLORS[v] for v in variants], alpha=0.76,
                   capsize=2.5, linewidth=0.6, edgecolor="black")
            ax.set_xticks(x, [SHORT[v] for v in variants]); ax.set_ylim(0,1.05)
            ax.set_ylabel(ylabel); ax.grid(True, axis="y"); panel(ax, letter)
        for ax, metric, ylabel, letter in (
            (axes[1,0], "E_co_time_s", r"$E_{\mathrm{co\mathrm{-}time}}$ (s)", "(c)"),
            (axes[1,1], "E_miss_m", r"$E_{\mathrm{miss}}$ (m)", "(d)")):
            boxplot(ax, [data[data.variant == v][metric] for v in variants], variants, ylabel); panel(ax, letter)
        stem=f"ablation_monte_carlo_{case}_v10"; save(fig,outdir,stem); stems.append(stem)

        fig, axes = plt.subplots(2,2,figsize=(DOUBLE,4.35),constrained_layout=True)
        for ax, metric, ylabel, letter in zip(axes.flat,
            ("E_co_time_s","E_n_g","E_miss_m","E_t_s"),
            (r"$E_{\mathrm{co\mathrm{-}time}}$ (s)",r"$E_n$ (g)",r"$E_{\mathrm{miss}}$ (m)",r"$E_t$ (s)"),
            ("(a)","(b)","(c)","(d)")):
            boxplot(ax,[data[data.variant==v][metric] for v in variants],variants,ylabel); panel(ax,letter)
        stem=f"ablation_terminal_metrics_{case}_v10"; save(fig,outdir,stem); stems.append(stem)
    return stems


def main():
    p=argparse.ArgumentParser(); p.add_argument("--training_root",type=Path,required=True)
    p.add_argument("--episodes_csv",type=Path); p.add_argument("--outdir",type=Path,required=True)
    a=p.parse_args(); style(); generated=plot_training(a.training_root,a.outdir)
    if a.episodes_csv: generated += plot_mc(a.episodes_csv,a.outdir)
    manifest={"style":"IEEE TASE V10","double_column_width_in":DOUBLE,
              "single_column_width_in":SINGLE,"png_dpi":600,
              "training_visual_smoothing_window_updates":TRAIN_WINDOW,
              "monte_carlo_values_smoothed":False,"missing_values_imputed":False,
              "generated_stems":generated}
    a.outdir.mkdir(parents=True,exist_ok=True)
    (a.outdir/"plot_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))


if __name__ == "__main__": main()
