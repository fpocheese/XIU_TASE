#!/usr/bin/env python3
"""Audited failure-mode analysis for reviewer comments 3.8/3.9."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ABLATION_ROOT = Path("/home/a2rl/reviewer_xiu_ablation_20260729")
CASE3_ROOT = Path(
    "/home/a2rl/reviewer_xiu_case3_20260729/results/"
    "formal_case3_idbo_artmappo_n100_seed74001"
)
OUT = ABLATION_ROOT / "failure_analysis"


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    episodes = pd.read_csv(
        ABLATION_ROOT / "formal_evaluation_n100/combined/episodes.csv"
    )
    targets = pd.read_csv(
        ABLATION_ROOT / "formal_evaluation_n100/combined/targets.csv"
    )
    c2 = episodes[(episodes["variant"] == "full") & (episodes["case"] == "case2")]
    c2t = targets[(targets["variant"] == "full") & (targets["case"] == "case2")]
    c2_fail = c2[c2["target_coverage_success"] == 0].copy()
    c2_uncovered = c2t[c2t["target_covered"] == 0].copy()

    c2_fail.to_csv(OUT / "case2_full_unsuccessful_episodes.csv", index=False)
    c2_uncovered.to_csv(OUT / "case2_full_uncovered_target_groups.csv", index=False)
    c2_freq = (
        c2_uncovered.groupby("target_index")
        .agg(
            uncovered_episode_count=("episode", "nunique"),
            mean_closest_approach_m=("closest_approach_m", "mean"),
            median_closest_approach_m=("closest_approach_m", "median"),
            max_closest_approach_m=("closest_approach_m", "max"),
        )
        .reset_index()
    )
    c2_freq.to_csv(OUT / "case2_full_uncovered_target_frequency.csv", index=False)

    c3e = pd.read_csv(CASE3_ROOT / "case3_episode_metrics.csv")
    c3a = pd.read_csv(CASE3_ROOT / "case3_assignment_and_arrivals.csv")
    delayed = c3e[(c3e["all_targets_hit"]) & (~c3e["all_targets_coordinated"])].copy()
    delayed.to_csv(OUT / "case3_delayed_cooperation_episodes.csv", index=False)

    group = (
        c3a.groupby(["episode", "seed", "target_index"])
        .agg(
            group_size=("defender_id", "size"),
            hit_count=("hit_time_s", "count"),
            first_hit_s=("hit_time_s", "min"),
            last_hit_s=("hit_time_s", "max"),
        )
        .reset_index()
    )
    group["arrival_spread_s"] = group["last_hit_s"] - group["first_hit_s"]
    group["delayed_group"] = group["arrival_spread_s"] > 0.5 + 1e-9
    delayed_group = group[group["delayed_group"]].copy()
    delayed_group.to_csv(OUT / "case3_delayed_target_groups.csv", index=False)
    c3_freq = (
        delayed_group.groupby("target_index")
        .agg(
            delayed_episode_count=("episode", "nunique"),
            mean_arrival_spread_s=("arrival_spread_s", "mean"),
            median_arrival_spread_s=("arrival_spread_s", "median"),
            max_arrival_spread_s=("arrival_spread_s", "max"),
        )
        .reset_index()
    )
    c3_freq.to_csv(OUT / "case3_delayed_target_frequency.csv", index=False)

    c2_distribution = (
        c2_fail["targets_covered"].value_counts().sort_index().rename_axis(
            "targets_covered"
        ).reset_index(name="episode_count")
    )
    c2_distribution.to_csv(
        OUT / "case2_full_failed_targets_covered_distribution.csv", index=False
    )

    summary = {
        "case2_full": {
            "episodes": int(len(c2)),
            "target_coverage_failures": int(len(c2_fail)),
            "target_coverage_failure_rate": float(len(c2_fail) / len(c2)),
            "mean_targets_covered_in_failed_episodes": float(
                c2_fail["targets_covered"].mean()
            ),
            "median_targets_covered_in_failed_episodes": float(
                c2_fail["targets_covered"].median()
            ),
            "mean_worst_closest_approach_m_failed": float(
                c2_fail["worst_closest_approach_m"].mean()
            ),
            "median_worst_closest_approach_m_failed": float(
                c2_fail["worst_closest_approach_m"].median()
            ),
            "uncovered_target_group_rows": int(len(c2_uncovered)),
        },
        "case3": {
            "episodes": int(len(c3e)),
            "all_target_interception_successes": int(c3e["all_targets_hit"].sum()),
            "strict_cooperative_successes": int(c3e["mission_success"].sum()),
            "delayed_cooperation_failures": int(len(delayed)),
            "delayed_target_group_rows": int(len(delayed_group)),
            "mean_worst_group_spread_s_delayed": float(
                delayed["max_group_spread_s"].mean()
            ),
            "median_worst_group_spread_s_delayed": float(
                delayed["max_group_spread_s"].median()
            ),
            "p95_worst_group_spread_s_delayed": float(
                delayed["max_group_spread_s"].quantile(0.95)
            ),
        },
    }
    (OUT / "failure_analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "legend.fontsize": 7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    blue, orange = "#0072B2", "#D55E00"
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55))
    ax = axes[0]
    ax.bar(
        c2_distribution["targets_covered"],
        c2_distribution["episode_count"],
        color=blue,
        edgecolor="black",
        linewidth=0.6,
        width=0.65,
    )
    ax.set_xlabel("Targets intercepted in failed Case-2 episodes")
    ax.set_ylabel("Episode count")
    ax.set_xticks(sorted(c2_distribution["targets_covered"].unique()))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.text(0.02, 0.96, "(a)", transform=ax.transAxes, va="top", fontweight="bold")

    ax = axes[1]
    values = np.sort(delayed["max_group_spread_s"].to_numpy(float))
    ax.plot(
        np.arange(1, len(values) + 1),
        values,
        marker="o",
        markersize=3.0,
        linewidth=1.2,
        color=orange,
        label="Delayed Case-3 episodes",
    )
    ax.axhline(
        0.5,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="0.5-s requirement",
    )
    ax.set_xlabel("Delayed episode (sorted)")
    ax.set_ylabel("Worst-group arrival spread (s)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(loc="upper left", frameon=True)
    ax.text(0.02, 0.96, "(b)", transform=ax.transAxes, va="top", fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.20, top=0.97, wspace=0.30)
    save_figure(fig, OUT / "failure_case_analysis_v10")
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
