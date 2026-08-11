#!/usr/bin/env python3
"""Plot Case 2 asset-safe interception successes with E_t <= 45 s.

Case 1 retains the complete asset-safe interception-success population. Case 2
uses the locally audited asset-safe interception successes whose recorded
coordinated interception duration does not exceed the requested threshold.
The filtering rule is exported with the data and is intentionally not printed
inside the figure.
"""

from __future__ import annotations

import argparse
import csv
import json
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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]], case: str, rule: str) -> list[dict]:
    output = []
    for key, _ in METRICS:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        output.append(
            {
                "case": case,
                "selection_rule": rule,
                "n": int(values.size),
                "metric": key,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "median": float(np.median(values)),
                "q1": float(np.quantile(values, 0.25)),
                "q3": float(np.quantile(values, 0.75)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case1-subset", type=Path, required=True)
    parser.add_argument("--case2-episodes", type=Path, required=True)
    parser.add_argument("--asset-audit", type=Path, required=True)
    parser.add_argument("--max-et", type=float, default=45.0)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    case1 = read_rows(args.case1_subset)
    case2_all = read_rows(args.case2_episodes)
    audit = [
        row
        for row in read_rows(args.asset_audit)
        if row["case"] == "case2"
    ]
    audit_by_episode = {int(row["episode"]): row for row in audit}
    case2 = [
        row
        for row in case2_all
        if int(
            audit_by_episode[int(row["episode"])][
                "asset_safe_interception_success"
            ]
        )
        and np.isfinite(float(row["E_t_s"]))
        and float(row["E_t_s"]) <= args.max_et
    ]
    if not case1 or not case2:
        raise RuntimeError("one of the selected populations is empty")

    write_rows(args.outdir / "case1_asset_safe_success_subset.csv", case1)
    write_rows(
        args.outdir / "case2_asset_safe_success_et_le_45s_subset.csv", case2
    )
    stats = summarize(
        case1, "case1", "asset_safe_interception_success == 1"
    ) + summarize(
        case2,
        "case2",
        "asset_safe_interception_success == 1 and E_t_s <= 45.0",
    )
    write_rows(args.outdir / "case2_et45_subset_statistics.csv", stats)

    strict_count = sum(
        int(row["cooperative_success"]) == 1 for row in case2
    )
    manifest = {
        "case1_input": str(args.case1_subset),
        "case1_selection": "asset_safe_interception_success == 1",
        "case1_selected_episodes": len(case1),
        "case2_input": str(args.case2_episodes),
        "case2_selection": (
            "asset_safe_interception_success == 1 and E_t_s <= 45.0"
        ),
        "case2_selected_episodes": len(case2),
        "case2_selected_strict_cooperative_success_count": strict_count,
        "E_t_threshold_s": args.max_et,
        "threshold_inclusive": True,
        "metric_values_modified": False,
        "figure_labels": "Case 1 and Case 2 only, as requested",
        "reporting_note": (
            "This is a conditionally selected Case 2 subset and must not be "
            "reported as the unconditional 100-trial distribution or ISR."
        ),
    }
    (args.outdir / "case2_et45_subset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

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
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.25))
    for panel, (axis, (key, ylabel)) in enumerate(zip(axes.flat, METRICS)):
        values = [
            np.asarray([float(row[key]) for row in case1], dtype=float),
            np.asarray([float(row[key]) for row in case2], dtype=float),
        ]
        boxes = axis.boxplot(
            values,
            labels=["Case 1", "Case 2"],
            widths=0.48,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 3.2,
            },
            medianprops={"color": "black", "linewidth": 1.15},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
            flierprops={
                "marker": "o",
                "markersize": 2.2,
                "markerfacecolor": "none",
                "markeredgecolor": "#666666",
                "alpha": 0.55,
            },
        )
        for patch, color in zip(boxes["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
            patch.set_alpha(0.65)
            patch.set_linewidth(0.8)
        axis.set_ylabel(ylabel)
        axis.grid(
            True,
            axis="y",
            linestyle="--",
            linewidth=0.35,
            color="#B8B8B8",
            alpha=0.65,
        )
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
    fig.subplots_adjust(
        left=0.075, right=0.995, bottom=0.22, top=0.94, wspace=0.52
    )
    stem = args.outdir / "two_defender_failure_mc_boxplots_case2_et_le_45s"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
