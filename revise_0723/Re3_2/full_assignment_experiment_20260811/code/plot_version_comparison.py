#!/usr/bin/env python3
"""Plot and tabulate the paired V1/V2 comparison."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args(); out = args.root / "comparison"
    quality = read(out / "data" / "paired_quality_raw.csv")
    qsum = read(out / "data" / "paired_quality_summary.csv")
    blue, orange = "#0072B2", "#D55E00"
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8.2,
                         "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "axes.linewidth": 0.8, "lines.linewidth": 1.5,
                         "lines.markersize": 4.5})
    fig, ax = plt.subplots(2, 2, figsize=(7.15, 5.05), constrained_layout=True)

    for version, color, marker in [("V1", blue, "o"), ("V2", orange, "s")]:
        rt = [r for r in read(args.root / version / "data" / "runtime_summary.csv")
              if r["sweep"] == "problem_size"]
        ax[0, 0].semilogy([float(r["M"]) for r in rt],
                          [float(r["runtime_s_mean"]) for r in rt],
                          marker + "-", color=color, label=version)
    ax[0, 0].set(xlabel="Number of defenders $M$ ($M/N=2.5$)",
                 ylabel="IDBO runtime (s)")
    ax[0, 0].axvline(20, color="0.4", linestyle=":", linewidth=1.0)
    ax[0, 0].text(20, ax[0, 0].get_ylim()[1], " nominal $20\\times8$",
                  rotation=90, va="top", ha="left", fontsize=7)
    ax[0, 0].legend(frameon=False)

    costs = [[float(r["assignment_cost"]) for r in quality if r["version"] == v]
             for v in ["V1", "V2"]]
    bp = ax[0, 1].boxplot(costs, labels=["V1", "V2"], widths=0.5,
                          patch_artist=True, showmeans=True,
                          meanprops={"marker": "D", "markerfacecolor": "white",
                                     "markeredgecolor": "black", "markersize": 4})
    for patch, color in zip(bp["boxes"], [blue, orange]):
        patch.set_facecolor(color); patch.set_alpha(0.55)
    ax[0, 1].set(ylabel="Static assignment cost (lower is better)")

    metrics = ["mean_interception_probability_mean", "mean_adversarial_advantage_mean",
               "covered_target_fraction_mean"]
    labels = ["Interception\nprobability", "Adversarial\nadvantage", "Target\ncoverage"]
    x = np.arange(3); width = 0.34
    for idx, (version, color) in enumerate([("V1", blue), ("V2", orange)]):
        row = next(r for r in qsum if r["version"] == version)
        values = [float(row[m]) for m in metrics]
        ax[1, 0].bar(x + (idx - .5) * width, values, width, color=color,
                     alpha=.75, edgecolor="black", linewidth=.5, label=version)
    ax[1, 0].set_xticks(x, labels)
    ax[1, 0].set(ylabel="Paired-scene mean", ylim=(0, 1.08))
    ax[1, 0].legend(frameon=False)

    for version, color, marker in [("V1", blue, "o"), ("V2", orange, "s")]:
        dy = read(args.root / version / "data" / "dynamic_summary.csv")
        ax[1, 1].errorbar([float(r["delay_ms"]) for r in dy],
                          [float(r["winner_jaccard_mean"]) for r in dy],
                          yerr=[float(r["winner_jaccard_ci95"]) for r in dy],
                          fmt=marker + "-", capsize=2.5, color=color, label=version)
    ax[1, 1].set(xlabel="Additional per-hop delay (ms)",
                 ylabel="Dynamic winner-set agreement", ylim=(0.45, 1.02))
    ax[1, 1].legend(frameon=False)

    for label, a in zip(["(a)", "(b)", "(c)", "(d)"], ax.flat):
        a.grid(True, alpha=.22, linewidth=.45)
        a.tick_params(direction="in", top=True, right=True)
        a.text(0, 1.02, label, transform=a.transAxes, va="bottom", fontweight="bold")
    figdir = out / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    for ext, opts in [("pdf", {}), ("svg", {}), ("png", {"dpi": 600})]:
        fig.savefig(figdir / f"idbo_v1_v2_comparison.{ext}", bbox_inches="tight", **opts)
    plt.close(fig)

    combined = read(out / "data" / "version_comparison_summary.csv")
    lines = []
    for row in combined:
        lines.append(
            f"{row['version']} & {float(row['runtime_20x8_s']):.3f} & "
            f"{float(row['runtime_160x64_s']):.3f} & {float(row['static_assignment_cost']):.3f} & "
            f"{float(row['dynamic_change_fraction']):.3f} & {float(row['agreement_at_100ms']):.3f} \\\\")
    tex = r"""\begin{table}[!t]
\centering
\footnotesize
\setlength{\tabcolsep}{2.0pt}
\caption{Paired comparison of the V1 and V2 IDBO implementations.}
\label{tab:idbo_v1_v2}
\begin{tabular}{@{}cccccc@{}}
\hline\hline
Version & $t_{20\times8}$ (s) & $t_{160\times64}$ (s) & Cost $\downarrow$ & Change & Agree. (100 ms) \\
\hline
""" + "\n".join(lines) + r"""
\hline
\end{tabular}
\end{table}
"""
    tabledir = out / "tables"; tabledir.mkdir(parents=True, exist_ok=True)
    (tabledir / "table_v1_v2_comparison.tex").write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    main()
