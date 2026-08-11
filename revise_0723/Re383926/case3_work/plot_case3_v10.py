#!/usr/bin/env python3
"""IEEE/V10-style Case-3 summary figure from the formal raw CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]


def wilson_half_width(successes, n, z=1.959963984540054):
    if n <= 0:
        return np.nan
    p = successes / n
    den = 1.0 + z * z / n
    return z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den


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
    success = np.asarray([r["mission_success"].lower() == "true" for r in rows])
    successful_rows = [r for r, ok in zip(rows, success) if ok]

    plt.rcParams.update(
        {
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
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.55), constrained_layout=True)

    rates = [
        np.mean(as_float(rows, "interceptor_hit_count") / 20.0),
        np.mean(as_float(rows, "target_coverage_count") / 8.0),
        np.mean([r["all_targets_hit"].lower() == "true" for r in rows]),
        np.mean(success),
    ]
    labels = ["Individual\nhit", "Target\ncoverage", "All-target\nhit", "Coop.\nmission"]
    counts = [
        int(round(rates[0] * len(rows) * 20)),
        int(round(rates[1] * len(rows) * 8)),
        int(round(rates[2] * len(rows))),
        int(round(rates[3] * len(rows))),
    ]
    denoms = [len(rows) * 20, len(rows) * 8, len(rows), len(rows)]
    errors = [wilson_half_width(k, n) for k, n in zip(counts, denoms)]
    ax = axes[0, 0]
    x = np.arange(4)
    ax.bar(x, 100 * np.asarray(rates), color=COLORS[:4], width=0.68, edgecolor="black", linewidth=0.55)
    ax.errorbar(x, 100 * np.asarray(rates), yerr=100 * np.asarray(errors), fmt="none", ecolor="black", capsize=2.2, lw=0.8)
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelsize=5.8)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color="0.88", lw=0.55)
    ax.text(-0.16, 1.03, "(a)", transform=ax.transAxes, fontweight="bold")

    metrics = [
        ("E_co_time_s", r"$E_{\mathrm{co\mathrm{-}time}}$ (s)", "(b)"),
        ("E_n_g", r"$E_n$ (g)", "(c)"),
        ("E_miss_m", r"$E_{\mathrm{miss}}$ (m)", "(d)"),
        ("E_t_s", r"$E_t$ (s)", "(e)"),
    ]
    for ax, (key, ylabel, panel) in zip(axes.flat[1:5], metrics):
        data = as_float(successful_rows, key)
        data = data[np.isfinite(data)]
        if data.size == 0:
            ax.text(0.5, 0.5, "No successful trials", ha="center", va="center", transform=ax.transAxes)
        else:
            bp = ax.boxplot(
                [data],
                widths=0.38,
                patch_artist=True,
                showfliers=True,
                medianprops={"color": "black", "linewidth": 1.2},
                whiskerprops={"linewidth": 0.8},
                capprops={"linewidth": 0.8},
                boxprops={"linewidth": 0.8},
                flierprops={"markersize": 2.3, "markerfacecolor": "none", "markeredgecolor": "0.35"},
            )
            bp["boxes"][0].set_facecolor(COLORS[metrics.index((key, ylabel, panel))])
            bp["boxes"][0].set_alpha(0.72)
            ax.scatter([1], [np.mean(data)], marker="D", s=16, color="white", edgecolor="black", linewidth=0.6, zorder=4, label="Mean")
        ax.set_xticks([1], ["Case 3"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="0.88", lw=0.55)
        ax.text(-0.16, 1.03, panel, transform=ax.transAxes, fontweight="bold")

    ax = axes[1, 2]
    runtime = as_float(rows, "idbo_runtime_ms")
    ax.hist(runtime, bins=min(12, max(5, len(rows) // 8)), color=COLORS[5], alpha=0.78, edgecolor="black", linewidth=0.5)
    ax.axvline(np.mean(runtime), color="#D55E00", ls="--", lw=1.1, label=f"Mean = {np.mean(runtime):.1f} ms")
    ax.set_xlabel("IDBO runtime (ms)")
    ax.set_ylabel("Number of trials")
    ax.grid(axis="y", color="0.88", lw=0.55)
    ax.legend(frameon=False, loc="best")
    ax.text(-0.16, 1.03, "(f)", transform=ax.transAxes, fontweight="bold")

    for ax in axes.flat:
        ax.tick_params(direction="in", length=3.0, width=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    stem = args.outdir / "case3_end_to_end_v10"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    summary_rows = [
        ["Individual interception", f"{100*rates[0]:.6f}", f"{100*errors[0]:.6f}", len(rows)],
        ["Target coverage", f"{100*rates[1]:.6f}", f"{100*errors[1]:.6f}", len(rows)],
        ["All-target interception", f"{100*rates[2]:.6f}", f"{100*errors[2]:.6f}", len(rows)],
        ["Cooperative mission", f"{100*rates[3]:.6f}", f"{100*errors[3]:.6f}", len(rows)],
    ]
    with (args.outdir / "case3_rate_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Rate_percent", "Wilson95_half_width_percent", "Episodes"])
        w.writerows(summary_rows)
    (args.outdir / "case3_plot_summary.json").write_text(
        json.dumps(
            {
                "episodes": len(rows),
                "successful_trials": int(np.sum(success)),
                "metrics_use_successful_trials_only": True,
                "rates": dict(zip(labels, rates)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
