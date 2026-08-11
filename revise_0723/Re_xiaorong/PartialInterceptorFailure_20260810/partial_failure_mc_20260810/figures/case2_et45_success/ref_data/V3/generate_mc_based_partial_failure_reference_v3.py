#!/usr/bin/env python3
"""Generate a heterogeneous V3 planning reference for failure experiments.

The output is synthetic guidance for future simulator runs, not experimental
evidence.  Unlike V2, V3 combines source-episode difficulty, four failure
severity levels, a load-redistribution shock, cross-metric correlation, and
severity-dependent variance.  This avoids representing partial failure as an
approximately constant offset from the nominal distribution.
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
import generate_mc_based_partial_failure_reference_v2 as v2


SEVERITY_NAMES = np.asarray(["low", "mild", "moderate", "severe"])
CONFIG = {
    "case1": {
        "probabilities": [0.38, 0.38, 0.19, 0.05],
        "severity_centers": [0.12, 0.72, 1.55, 2.75],
        "fraction_coefficients": [0.20, 0.45, 0.13, 0.09],
        "source_spread": [1.25, 1.45, 1.35, 1.30],
    },
    "case2": {
        "probabilities": [0.23, 0.36, 0.29, 0.12],
        "severity_centers": [0.10, 0.78, 1.70, 3.00],
        "fraction_coefficients": [0.28, 0.40, 0.17, 0.12],
        "source_spread": [1.40, 1.55, 1.55, 1.30],
    },
}
SEVERITY_LOGNORMAL_SIGMA = 0.35
LOAD_REDISTRIBUTION_SIGMA = 0.38


def smooth_tail(values: np.ndarray, start: float, end: float, scale: float) -> np.ndarray:
    """Smoothly compress only the far upper tail without clipping a plateau."""
    result = values.copy()
    mask = result > start
    result[mask] = start + (end - start) * np.tanh((result[mask] - start) / scale)
    return result


def transform_distribution(
    nominal: np.ndarray, case: str, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = CONFIG[case]
    mean = np.mean(nominal, axis=0)
    std = np.std(nominal, axis=0, ddof=1)
    standardized = (nominal - mean) / std

    # A continuous source-episode difficulty score couples the failure impact
    # to the original trajectory rather than adding a row-independent offset.
    common_difficulty_noise = rng.normal(size=nominal.shape[0])
    difficulty_logit = (
        0.50 * standardized[:, 0]
        + 0.35 * standardized[:, 1]
        + 0.20 * standardized[:, 2]
        + 0.20 * standardized[:, 3]
        + 0.55 * common_difficulty_noise
    )
    difficulty = 1.0 / (1.0 + np.exp(-difficulty_logit))

    categories = rng.choice(
        4, size=nominal.shape[0], p=np.asarray(cfg["probabilities"])
    )
    centers = np.asarray(cfg["severity_centers"])
    severity = centers[categories] * rng.lognormal(
        mean=-0.5 * SEVERITY_LOGNORMAL_SIGMA**2,
        sigma=SEVERITY_LOGNORMAL_SIGMA,
        size=nominal.shape[0],
    )
    effective_severity = severity * (0.60 + 0.90 * difficulty)

    # Loss of two defenders can redistribute terminal control demand unevenly.
    # The dedicated shock therefore acts most strongly on E_n.
    load_redistribution = rng.lognormal(
        mean=-0.5 * LOAD_REDISTRIBUTION_SIGMA**2,
        sigma=LOAD_REDISTRIBUTION_SIGMA,
        size=nominal.shape[0],
    )
    metric_severity = np.column_stack(
        [
            effective_severity,
            effective_severity * load_redistribution,
            effective_severity * (0.75 + 0.50 * rng.random(nominal.shape[0])),
            effective_severity * (0.80 + 0.40 * rng.random(nominal.shape[0])),
        ]
    )

    fraction = metric_severity * np.asarray(cfg["fraction_coefficients"])
    heteroscedastic_noise = (
        (0.12 + 0.16 * effective_severity[:, None])
        * std
        * rng.normal(size=nominal.shape)
    )
    centered_right_skew = (
        (rng.gamma(shape=2.0, scale=1.0, size=nominal.shape) - 2.0)
        * std
        * (0.06 + 0.08 * effective_severity[:, None])
    )
    reference = (
        mean * (1.0 + fraction)
        + np.asarray(cfg["source_spread"]) * (nominal - mean)
        + heteroscedastic_noise
        + centered_right_skew
    )
    reference[:, :3] = np.maximum(reference[:, :3], 1.0e-8)

    # Preserve the previously requested successful-engagement time range while
    # avoiding a hard-clipped pile-up at the upper boundary.
    if case == "case1":
        reference[:, 2] = smooth_tail(reference[:, 2], 5.0, 5.8, 1.2)
        reference[:, 3] = smooth_tail(reference[:, 3], 39.0, 42.0, 3.0)
    else:
        reference[:, 2] = smooth_tail(reference[:, 2], 6.0, 7.0, 1.5)
        reference[:, 3] = smooth_tail(reference[:, 3], 42.5, 45.0, 3.0)
    reference[:, 3] = np.round(reference[:, 3] / 0.05) * 0.05
    return reference, categories, effective_severity, difficulty


def write_reference(
    path: Path,
    nominal: np.ndarray,
    reference: np.ndarray,
    categories: np.ndarray,
    effective_severity: np.ndarray,
    difficulty: np.ndarray,
) -> None:
    fields = [
        "reference_sample_id",
        "data_status",
        "reference_severity_class",
        "source_difficulty_score",
        "effective_failure_severity",
        *[f"source_{key}" for key, _ in base.METRICS],
        *[key for key, _ in base.METRICS],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, values in enumerate(zip(nominal, reference), start=1):
            source, transformed = values
            row: dict[str, str | int | float] = {
                "reference_sample_id": index,
                "data_status": "synthetic_planning_reference_v3",
                "reference_severity_class": SEVERITY_NAMES[int(categories[index - 1])],
                "source_difficulty_score": float(difficulty[index - 1]),
                "effective_failure_severity": float(effective_severity[index - 1]),
            }
            for (key, _), value in zip(base.METRICS, source):
                row[f"source_{key}"] = float(value)
            for (key, _), value in zip(base.METRICS, transformed):
                row[key] = float(value)
            writer.writerow(row)


def apply_case4_style() -> None:
    """Match the font and line sizing of the fig:case4_metrics source."""
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
        }
    )


def draw_boxplot(axis, values, labels, colors, width=0.42) -> None:
    result = axis.boxplot(
        values,
        widths=width,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
        boxprops={"linewidth": 0.8},
        flierprops={
            "marker": "o",
            "markersize": 2.3,
            "markerfacecolor": "none",
            "markeredgecolor": "0.35",
            "alpha": 0.55,
        },
    )
    for patch, color in zip(result["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    for position, data in enumerate(values, start=1):
        axis.scatter(
            [position],
            [np.mean(data)],
            marker="D",
            s=16,
            color="white",
            edgecolor="black",
            linewidth=0.6,
            zorder=4,
        )
    axis.set_xticks(np.arange(1, len(labels) + 1), labels)


def format_axis(axis, ylabel: str) -> None:
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="0.88", lw=0.55)
    axis.tick_params(direction="in", length=3.0, width=0.7, top=True, right=True)
    axis.spines["top"].set_visible(True)
    axis.spines["right"].set_visible(True)


def save_figure(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_failure_only(case1: np.ndarray, case2: np.ndarray, outdir: Path) -> None:
    apply_case4_style()
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.05), constrained_layout=True)
    for index, (axis, (_, ylabel)) in enumerate(zip(axes, base.METRICS)):
        draw_boxplot(
            axis,
            [case1[:, index], case2[:, index]],
            ["Case 1", "Case 2"],
            ["#0072B2", "#D55E00"],
            width=0.42,
        )
        format_axis(axis, ylabel)
    save_figure(fig, outdir / "partial_interceptor_failure_reference_v3_boxplots")


def plot_nominal_comparison(
    nominal1: np.ndarray,
    failure1: np.ndarray,
    nominal2: np.ndarray,
    failure2: np.ndarray,
    outdir: Path,
) -> None:
    apply_case4_style()
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.15), constrained_layout=True)
    labels = ["C1\nNom.", "C1\nFail.", "C2\nNom.", "C2\nFail."]
    colors = ["#9ECAE1", "#0072B2", "#FDD0A2", "#D55E00"]
    for index, (axis, (_, ylabel)) in enumerate(zip(axes, base.METRICS)):
        draw_boxplot(
            axis,
            [
                nominal1[:, index],
                failure1[:, index],
                nominal2[:, index],
                failure2[:, index],
            ],
            labels,
            colors,
            width=0.50,
        )
        format_axis(axis, ylabel)
    save_figure(
        fig, outdir / "partial_interceptor_failure_reference_v3_nominal_comparison"
    )


def distribution_rows(data: np.ndarray, case: str, dataset_kind: str) -> list[dict]:
    rows = []
    for index, (key, _) in enumerate(base.METRICS):
        values = data[:, index]
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        centered = values - mean
        skewness = float(np.mean(centered**3) / max(np.mean(centered**2) ** 1.5, 1e-16))
        q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        rows.append(
            {
                "case": case,
                "dataset_kind": dataset_kind,
                "metric": key,
                "n": len(values),
                "mean": mean,
                "std": std,
                "cv": std / mean,
                "q1": float(q1),
                "median": float(median),
                "q3": float(q3),
                "iqr": float(q3 - q1),
                "skewness": skewness,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return rows


def effect_rows(nominal: np.ndarray, reference: np.ndarray, case: str) -> list[dict]:
    rows = []
    for index, (key, _) in enumerate(base.METRICS):
        nominal_mean = float(np.mean(nominal[:, index]))
        reference_mean = float(np.mean(reference[:, index]))
        nominal_std = float(np.std(nominal[:, index], ddof=1))
        reference_std = float(np.std(reference[:, index], ddof=1))
        rows.append(
            {
                "case": case,
                "metric": key,
                "nominal_mean": nominal_mean,
                "reference_failure_mean": reference_mean,
                "mean_change_percent": 100.0 * (reference_mean / nominal_mean - 1.0),
                "nominal_std": nominal_std,
                "reference_failure_std": reference_std,
                "std_ratio": reference_std / nominal_std,
            }
        )
    return rows


def v2_v3_rows(v2_data: np.ndarray, v3_data: np.ndarray, case: str) -> list[dict]:
    rows = []
    for index, (key, _) in enumerate(base.METRICS):
        v2_q1, v2_med, v2_q3 = np.quantile(v2_data[:, index], [0.25, 0.5, 0.75])
        v3_q1, v3_med, v3_q3 = np.quantile(v3_data[:, index], [0.25, 0.5, 0.75])
        rows.append(
            {
                "case": case,
                "metric": key,
                "v2_mean": float(np.mean(v2_data[:, index])),
                "v3_mean": float(np.mean(v3_data[:, index])),
                "v3_vs_v2_mean_change_percent": 100.0
                * (np.mean(v3_data[:, index]) / np.mean(v2_data[:, index]) - 1.0),
                "v2_std": float(np.std(v2_data[:, index], ddof=1)),
                "v3_std": float(np.std(v3_data[:, index], ddof=1)),
                "v3_vs_v2_std_ratio": float(
                    np.std(v3_data[:, index], ddof=1)
                    / np.std(v2_data[:, index], ddof=1)
                ),
                "v2_q1": float(v2_q1),
                "v2_median": float(v2_med),
                "v2_q3": float(v2_q3),
                "v3_q1": float(v3_q1),
                "v3_median": float(v3_med),
                "v3_q3": float(v3_q3),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case1", type=Path, required=True)
    parser.add_argument("--case2", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    nominal1 = base.load_nominal(args.case1, args.samples)
    nominal2 = base.load_nominal(args.case2, args.samples)
    failure1, categories1, severity1, difficulty1 = transform_distribution(
        nominal1, "case1", np.random.default_rng(args.seed)
    )
    failure2, categories2, severity2, difficulty2 = transform_distribution(
        nominal2, "case2", np.random.default_rng(args.seed + 1)
    )
    if not np.isfinite(failure1).all() or not np.isfinite(failure2).all():
        raise RuntimeError("V3 reference contains NaN or Inf")
    if np.max(failure2[:, 3]) > 45.0 + 1e-12:
        raise RuntimeError("Case 2 E_t exceeds the retained 45-s reference range")

    write_reference(
        args.outdir / "case1_partial_failure_reference_v3_n1000.csv",
        nominal1,
        failure1,
        categories1,
        severity1,
        difficulty1,
    )
    write_reference(
        args.outdir / "case2_partial_failure_reference_v3_n1000.csv",
        nominal2,
        failure2,
        categories2,
        severity2,
        difficulty2,
    )
    distribution = (
        distribution_rows(nominal1, "case1", "paper_nominal_mc_source")
        + distribution_rows(failure1, "case1", "synthetic_failure_reference_v3")
        + distribution_rows(nominal2, "case2", "paper_nominal_mc_source")
        + distribution_rows(failure2, "case2", "synthetic_failure_reference_v3")
    )
    base.write_table(args.outdir / "reference_statistics_v3.csv", distribution)
    base.write_table(
        args.outdir / "reference_effect_summary_v3.csv",
        effect_rows(nominal1, failure1, "case1")
        + effect_rows(nominal2, failure2, "case2"),
    )

    # Recompute V2 deterministically to quantify how V3 changes shape and En.
    failure1_v2, _ = v2.transform_mixture(
        nominal1, "case1", np.random.default_rng(20260820)
    )
    failure2_v2, _ = v2.transform_mixture(
        nominal2, "case2", np.random.default_rng(20260821)
    )
    base.write_table(
        args.outdir / "v2_v3_distribution_comparison.csv",
        v2_v3_rows(failure1_v2, failure1, "case1")
        + v2_v3_rows(failure2_v2, failure2, "case2"),
    )

    plot_failure_only(failure1, failure2, args.outdir)
    plot_nominal_comparison(nominal1, failure1, nominal2, failure2, args.outdir)

    counts1 = dict(
        zip(SEVERITY_NAMES.tolist(), np.bincount(categories1, minlength=4).tolist())
    )
    counts2 = dict(
        zip(SEVERITY_NAMES.tolist(), np.bincount(categories2, minlength=4).tolist())
    )
    manifest = {
        "purpose": "distribution-shape planning target for future independent failure simulations",
        "data_status": "synthetic planning reference; not experimental data",
        "source_case1": str(args.case1),
        "source_case2": str(args.case2),
        "samples_per_case": args.samples,
        "seed": args.seed,
        "configuration": CONFIG,
        "realized_severity_counts": {"case1": counts1, "case2": counts2},
        "mechanisms": [
            "source-dependent continuous difficulty",
            "four-level random failure severity",
            "load-redistribution shock for E_n",
            "cross-metric latent severity",
            "severity-dependent heteroscedastic noise",
            "centered right-skew perturbation",
        ],
        "plot_style_source": "fig:case4_metrics / Re3_9/plot_case3_box3.py",
        "plot_style_changes_requested": [
            "top and right spines retained",
            "panel labels removed",
        ],
        "simulator_executed": False,
        "policy_evaluation_executed": False,
        "reporting_restriction": (
            "Do not report these values as Monte Carlo trials. Replace them "
            "with independently simulated failure episodes."
        ),
    }
    (args.outdir / "manifest_v3.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.outdir / "README_v3.md").write_text(
        "# 部分拦截器失效参考分布 V3\n\n"
        "该目录仅用于规划后续真实仿真实验，不是实验结果。V3不再把失效影响近似为固定偏差，"
        "而是加入原始回合难度、四级失效强度、末端过载重分配冲击、指标间相关性、异方差和右偏尾部。"
        "因此中位数、IQR、标准差、须线和离群点都会发生不同程度的变化。\n\n"
        "绘图字体和线宽取自 `fig:case4_metrics` 的脚本；按照本次要求保留上/右边框，并删除"
        " `(a)--(d)` 标记。`v2_v3_distribution_comparison.csv` 用于核对V3相对V2的变化。"
        "所有CSV和图必须由后续独立仿真结果替换后才能用于论文或审稿回复。\n",
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__), args.outdir / Path(__file__).name)
    products = sorted(
        path
        for path in args.outdir.iterdir()
        if path.is_file() and path.name != "checksums_v3.sha256"
    )
    (args.outdir / "checksums_v3.sha256").write_text(
        "\n".join(f"{base.sha256(path)}  {path.name}" for path in products) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
