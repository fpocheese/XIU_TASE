#!/usr/bin/env python3
"""Generate a visibly degraded mixed-severity synthetic failure reference.

This is an experiment-planning artifact, not simulation evidence.  It uses the
paper's nominal Monte Carlo rows as a distributional basis and injects a
documented mixture of mild, moderate, and severe failure effects.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import generate_mc_based_partial_failure_reference as base


SEVERITY_NAMES = np.asarray(["mild", "moderate", "severe"])
MIXTURE = {
    "case1": {
        "probabilities": [0.65, 0.28, 0.07],
        "base_mean_fraction": [0.15, 0.16, 0.10, 0.08],
        "moderate_increment": [0.16, 0.08, 0.08, 0.05],
        "severe_increment": [0.38, 0.18, 0.18, 0.12],
        "spread_scale": [1.20, 1.18, 1.18, 1.15],
    },
    "case2": {
        "probabilities": [0.45, 0.40, 0.15],
        "base_mean_fraction": [0.20, 0.22, 0.12, 0.10],
        "moderate_increment": [0.22, 0.10, 0.10, 0.06],
        "severe_increment": [0.55, 0.25, 0.22, 0.15],
        "spread_scale": [1.35, 1.28, 1.25, 1.22],
    },
}
SEVERITY_LOGNORMAL_SIGMA = 0.18
COMMON_JITTER_STD_FRACTION = 0.10
INDEPENDENT_JITTER_STD_FRACTION = 0.08


def transform_mixture(
    nominal: np.ndarray, case: str, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    cfg = MIXTURE[case]
    categories = rng.choice(
        3, size=nominal.shape[0], p=cfg["probabilities"]
    )
    increments = np.zeros_like(nominal)
    increments[categories == 1] = np.asarray(cfg["moderate_increment"])
    increments[categories == 2] = np.asarray(cfg["severe_increment"])
    severity_multiplier = rng.lognormal(
        mean=-0.5 * SEVERITY_LOGNORMAL_SIGMA**2,
        sigma=SEVERITY_LOGNORMAL_SIGMA,
        size=(nominal.shape[0], 1),
    )
    mean = np.mean(nominal, axis=0)
    std = np.std(nominal, axis=0, ddof=1)
    common_jitter = rng.normal(size=(nominal.shape[0], 1))
    independent_jitter = rng.normal(size=nominal.shape)
    reference = (
        mean
        * (
            1.0
            + np.asarray(cfg["base_mean_fraction"])
            + increments * severity_multiplier
        )
        + np.asarray(cfg["spread_scale"]) * (nominal - mean)
        + COMMON_JITTER_STD_FRACTION * std * common_jitter
        + INDEPENDENT_JITTER_STD_FRACTION * std * independent_jitter
    )
    reference[:, :3] = np.maximum(reference[:, :3], 1.0e-8)
    if case == "case2":
        reference[:, 3] = base.soft_upper_bound(
            reference[:, 3], upper=45.0, tau=0.35
        )
    reference[:, 3] = np.round(reference[:, 3] / 0.05) * 0.05
    if case == "case2":
        reference[:, 3] = np.minimum(reference[:, 3], 44.95)
    return reference, categories


def write_reference(
    path: Path,
    nominal: np.ndarray,
    reference: np.ndarray,
    categories: np.ndarray,
) -> None:
    fields = [
        "reference_sample_id",
        "data_status",
        "reference_severity_class",
        *[f"source_{key}" for key, _ in base.METRICS],
        *[key for key, _ in base.METRICS],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (source, transformed, category) in enumerate(
            zip(nominal, reference, categories), start=1
        ):
            row: dict[str, str | int | float] = {
                "reference_sample_id": index,
                "data_status": "synthetic_planning_reference_v2",
                "reference_severity_class": SEVERITY_NAMES[int(category)],
            }
            for (key, _), value in zip(base.METRICS, source):
                row[f"source_{key}"] = float(value)
            for (key, _), value in zip(base.METRICS, transformed):
                row[key] = float(value)
            writer.writerow(row)


def apply_style() -> None:
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


def format_axis(axis: plt.Axes, panel: int, ylabel: str) -> None:
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


def draw_boxplot(axis: plt.Axes, values: list[np.ndarray], labels: list[str], colors: list[str]) -> None:
    boxes = axis.boxplot(
        values,
        labels=labels,
        widths=0.55 if len(values) == 2 else 0.62,
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
            "markersize": 2.0,
            "markerfacecolor": "none",
            "markeredgecolor": "#666666",
            "alpha": 0.38,
        },
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_alpha(0.70)
        patch.set_linewidth(0.8)


def plot_failure_only(case1: np.ndarray, case2: np.ndarray, outdir: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.38))
    for panel, (axis, (_, ylabel)) in enumerate(zip(axes.flat, base.METRICS)):
        draw_boxplot(
            axis,
            [case1[:, panel], case2[:, panel]],
            ["Case 1", "Case 2"],
            ["#0072B2", "#D55E00"],
        )
        format_axis(axis, panel, ylabel)
    fig.text(
        0.5,
        0.025,
        "Synthetic mixed-severity planning reference; not independent simulations.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.075, right=0.995, bottom=0.27, top=0.94, wspace=0.52
    )
    stem = outdir / "partial_interceptor_failure_reference_v2_boxplots"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_direct_comparison(
    nominal1: np.ndarray,
    failure1: np.ndarray,
    nominal2: np.ndarray,
    failure2: np.ndarray,
    outdir: Path,
) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.65))
    labels = ["C1\nNom.", "C1\nFail.", "C2\nNom.", "C2\nFail."]
    colors = ["#9ECAE1", "#0072B2", "#FDD0A2", "#D55E00"]
    for panel, (axis, (_, ylabel)) in enumerate(zip(axes.flat, base.METRICS)):
        draw_boxplot(
            axis,
            [
                nominal1[:, panel],
                failure1[:, panel],
                nominal2[:, panel],
                failure2[:, panel],
            ],
            labels,
            colors,
        )
        format_axis(axis, panel, ylabel)
    fig.text(
        0.5,
        0.018,
        "Failure distributions are synthetic planning references derived from nominal Monte Carlo data.",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.075, right=0.995, bottom=0.27, top=0.94, wspace=0.52
    )
    stem = outdir / "partial_interceptor_failure_reference_v2_nominal_comparison"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def effect_rows(
    nominal: np.ndarray, reference: np.ndarray, case: str
) -> list[dict]:
    rows = []
    for index, (key, _) in enumerate(base.METRICS):
        nominal_mean = float(np.mean(nominal[:, index]))
        reference_mean = float(np.mean(reference[:, index]))
        rows.append(
            {
                "case": case,
                "metric": key,
                "nominal_mean": nominal_mean,
                "reference_failure_mean": reference_mean,
                "mean_change_percent": 100.0
                * (reference_mean - nominal_mean)
                / nominal_mean,
                "nominal_std": float(np.std(nominal[:, index], ddof=1)),
                "reference_failure_std": float(
                    np.std(reference[:, index], ddof=1)
                ),
                "std_ratio": float(
                    np.std(reference[:, index], ddof=1)
                    / np.std(nominal[:, index], ddof=1)
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case1", type=Path, required=True)
    parser.add_argument("--case2", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    nominal1 = base.load_nominal(args.case1, args.samples)
    nominal2 = base.load_nominal(args.case2, args.samples)
    failure1, categories1 = transform_mixture(
        nominal1, "case1", np.random.default_rng(args.seed)
    )
    failure2, categories2 = transform_mixture(
        nominal2, "case2", np.random.default_rng(args.seed + 1)
    )
    if not np.isfinite(failure1).all() or not np.isfinite(failure2).all():
        raise RuntimeError("V2 reference contains NaN or Inf")

    write_reference(
        args.outdir / "case1_partial_failure_reference_v2_n1000.csv",
        nominal1,
        failure1,
        categories1,
    )
    write_reference(
        args.outdir / "case2_partial_failure_reference_v2_n1000.csv",
        nominal2,
        failure2,
        categories2,
    )
    statistics = (
        base.summarize(nominal1, "case1", "paper_nominal_mc_source")
        + base.summarize(failure1, "case1", "synthetic_failure_reference_v2")
        + base.summarize(nominal2, "case2", "paper_nominal_mc_source")
        + base.summarize(failure2, "case2", "synthetic_failure_reference_v2")
    )
    base.write_table(args.outdir / "reference_statistics_v2.csv", statistics)
    base.write_table(
        args.outdir / "reference_effect_summary_v2.csv",
        effect_rows(nominal1, failure1, "case1")
        + effect_rows(nominal2, failure2, "case2"),
    )
    plot_failure_only(failure1, failure2, args.outdir)
    plot_direct_comparison(nominal1, failure1, nominal2, failure2, args.outdir)

    counts1 = dict(
        zip(SEVERITY_NAMES.tolist(), np.bincount(categories1, minlength=3).tolist())
    )
    counts2 = dict(
        zip(SEVERITY_NAMES.tolist(), np.bincount(categories2, minlength=3).tolist())
    )
    manifest = {
        "purpose": "more differentiated planning reference for future failure tests",
        "data_status": "synthetic planning reference; not experimental data",
        "source_case1": str(args.case1),
        "source_case2": str(args.case2),
        "samples_per_case": args.samples,
        "seed": args.seed,
        "mixture_configuration": MIXTURE,
        "realized_severity_counts": {"case1": counts1, "case2": counts2},
        "severity_lognormal_sigma": SEVERITY_LOGNORMAL_SIGMA,
        "common_jitter_std_fraction": COMMON_JITTER_STD_FRACTION,
        "independent_jitter_std_fraction": INDEPENDENT_JITTER_STD_FRACTION,
        "case2_E_t_upper_bound_s": 45.0,
        "simulator_executed": False,
        "policy_evaluation_executed": False,
        "reporting_restriction": (
            "Do not report these values as Monte Carlo trials. Replace them "
            "with independently simulated failure episodes."
        ),
    }
    (args.outdir / "manifest_v2.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.outdir / "README_v2.md").write_text(
        "# 部分拦截器失效参考分布 V2\n\n"
        "该版本用于规划真实失效实验，不是实验结果。它在论文名义蒙特卡洛分布上加入轻度、"
        "中度和重度三类随机失效影响，使 Case 1 呈现温和退化、Case 2 呈现更明显的均值上升、"
        "箱体变宽和右侧长尾。`reference_effect_summary_v2.csv` 给出相对名义分布的均值变化和"
        "标准差比值。\n\n"
        "推荐优先查看 `partial_interceptor_failure_reference_v2_nominal_comparison.png`，因为"
        "它直接并列名义与失效参考分布。所有合成数据必须由后续独立仿真结果替换。\n",
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__), args.outdir / Path(__file__).name)

    products = sorted(
        path
        for path in args.outdir.iterdir()
        if path.is_file() and path.name != "checksums_v2.sha256"
    )
    (args.outdir / "checksums_v2.sha256").write_text(
        "\n".join(f"{base.sha256(path)}  {path.name}" for path in products)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
