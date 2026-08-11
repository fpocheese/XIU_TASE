#!/usr/bin/env python3
"""Generate the V5 synthetic planning reference from V4.

V5 changes only E_n and E_miss.  Cooperative-time and E_t are preserved
sample-by-sample from V4.  The products are planning guidance for future
independent simulations, not experimental or Monte Carlo evidence.
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
import generate_mc_based_partial_failure_reference_v3 as v3
import generate_mc_based_partial_failure_reference_v4 as v4


DATA_STATUS = "synthetic_planning_reference_v5"


def rank_preserving_map(
    values: np.ndarray,
    probability_knots: list[float],
    value_knots: list[float],
) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, values.size)
    return np.interp(ranks, probability_knots, value_knots)


def stochastic_rank_map(
    values: np.ndarray,
    probability_knots: list[float],
    value_knots: list[float],
    rng: np.random.Generator,
    noise_scale: float,
) -> np.ndarray:
    """Create a non-regular empirical distribution around quantile anchors."""
    result = rank_preserving_map(values, probability_knots, value_knots)
    result += rng.normal(0.0, noise_scale, size=values.size)
    result += rng.gamma(2.0, noise_scale / 3.0, size=values.size)
    result -= 2.0 * noise_scale / 3.0
    return result


def compress_upper_tail(
    values: np.ndarray, start: float, end: float, scale: float
) -> np.ndarray:
    result = values.copy()
    mask = result > start
    result[mask] = start + (end - start) * np.tanh(
        (result[mask] - start) / scale
    )
    return result


def refine_v4(
    case1_v4: np.ndarray, case2_v4: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    case1 = case1_v4.copy()
    case2 = case2_v4.copy()

    # E_n: compact central boxes and separate high-side tails.  This yields
    # upper whiskers below the extrema and therefore visible gray outliers.
    case1[:, 1] = stochastic_rank_map(
        case1_v4[:, 1],
        [0.00, 0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.97, 0.98, 0.99, 1.00],
        [0.040, 0.046, 0.055, 0.077, 0.090, 0.103, 0.116, 0.126, 0.132, 0.138, 0.145, 0.149],
        np.random.default_rng(20260850),
        0.0010,
    )
    case1[:, 1] = np.maximum(case1[:, 1], 0.038)
    case1[:, 1] = compress_upper_tail(case1[:, 1], 0.145, 0.1495, 0.006)

    case2[:, 1] = stochastic_rank_map(
        case2_v4[:, 1],
        [0.00, 0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.965, 0.97, 0.98, 0.99, 1.00],
        [0.060, 0.080, 0.110, 0.180, 0.225, 0.250, 0.300, 0.340, 0.350, 0.360, 0.375, 0.390, 0.398],
        np.random.default_rng(20260851),
        0.0015,
    )
    case2[:, 1] = np.maximum(case2[:, 1], 0.055)
    case2[:, 1] = compress_upper_tail(case2[:, 1], 0.388, 0.399, 0.012)

    # E_miss: distinct central locations (about 2.9 and 3.2 m), unequal
    # whiskers, and sparse non-clipped tails that do not pile up at 4.5 m.
    case1[:, 2] = stochastic_rank_map(
        case1_v4[:, 2],
        [0.00, 0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.97, 0.98, 0.99, 1.00],
        [1.85, 2.15, 2.38, 2.68, 2.90, 3.15, 3.38, 3.58, 3.72, 3.88, 4.15, 4.37],
        np.random.default_rng(20260852),
        0.018,
    )
    case1[:, 2] = compress_upper_tail(case1[:, 2], 4.30, 4.42, 0.12)

    case2[:, 2] = stochastic_rank_map(
        case2_v4[:, 2],
        [0.00, 0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.97, 0.98, 0.99, 1.00],
        [2.05, 2.30, 2.55, 2.92, 3.20, 3.47, 3.72, 3.89, 4.02, 4.18, 4.38, 4.46],
        np.random.default_rng(20260853),
        0.018,
    )
    case2[:, 2] = compress_upper_tail(case2[:, 2], 4.36, 4.48, 0.10)

    # The two metrics not requested for revision remain bitwise identical.
    for metric_index in (0, 3):
        if not np.array_equal(case1[:, metric_index], case1_v4[:, metric_index]):
            raise RuntimeError(f"Case 1 metric {metric_index} changed from V4")
        if not np.array_equal(case2[:, metric_index], case2_v4[:, metric_index]):
            raise RuntimeError(f"Case 2 metric {metric_index} changed from V4")
    return case1, case2


def write_reference(
    path: Path,
    nominal: np.ndarray,
    reference: np.ndarray,
    categories: np.ndarray,
    severity: np.ndarray,
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
        for index, (source, transformed) in enumerate(
            zip(nominal, reference), start=1
        ):
            row: dict[str, str | int | float] = {
                "reference_sample_id": index,
                "data_status": DATA_STATUS,
                "reference_severity_class": v3.SEVERITY_NAMES[
                    int(categories[index - 1])
                ],
                "source_difficulty_score": float(difficulty[index - 1]),
                "effective_failure_severity": float(severity[index - 1]),
            }
            for (key, _), value in zip(base.METRICS, source):
                row[f"source_{key}"] = float(value)
            for (key, _), value in zip(base.METRICS, transformed):
                row[key] = float(value)
            writer.writerow(row)


def validate(case1: np.ndarray, case2: np.ndarray) -> None:
    if not np.isfinite(case1).all() or not np.isfinite(case2).all():
        raise RuntimeError("V5 reference contains NaN or Inf")
    limits = {
        ("case1", 1): 0.15,
        ("case2", 1): 0.40,
        ("case1", 2): 4.50,
        ("case2", 2): 4.50,
    }
    diagnostics = {
        (row["case"], row["metric"]): row
        for case, data in (("case1", case1), ("case2", case2))
        for row in v4.boxplot_diagnostics(data, case, DATA_STATUS)
    }
    for (case, metric_index), limit in limits.items():
        data = case1 if case == "case1" else case2
        if np.max(data[:, metric_index]) >= limit:
            raise RuntimeError(f"{case} metric {metric_index} reaches {limit}")
    for case in ("case1", "case2"):
        for metric in ("E_n_g", "E_miss_m"):
            if diagnostics[(case, metric)]["upper_outlier_count"] <= 0:
                raise RuntimeError(f"{case} {metric} lacks upper outliers")
    upper_whisker = diagnostics[("case2", "E_n_g")]["upper_whisker"]
    if not (0.34 <= upper_whisker <= 0.36):
        raise RuntimeError(f"Case-2 E_n upper whisker is {upper_whisker}")
    if not (2.85 <= diagnostics[("case1", "E_miss_m")]["median"] <= 2.95):
        raise RuntimeError("Case-1 E_miss median is outside the target neighborhood")
    if not (3.15 <= diagnostics[("case2", "E_miss_m")]["median"] <= 3.25):
        raise RuntimeError("Case-2 E_miss median is outside the target neighborhood")


def save_figure(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_failure_only(case1: np.ndarray, case2: np.ndarray, outdir: Path) -> None:
    v3.apply_case4_style()
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.05), constrained_layout=True)
    for index, (axis, (_, ylabel)) in enumerate(zip(axes, base.METRICS)):
        v3.draw_boxplot(
            axis,
            [case1[:, index], case2[:, index]],
            ["Case 1", "Case 2"],
            ["#0072B2", "#D55E00"],
            width=0.42,
        )
        v3.format_axis(axis, ylabel)
    save_figure(fig, outdir / "partial_interceptor_failure_reference_v5_boxplots")


def plot_nominal_comparison(
    nominal1: np.ndarray,
    failure1: np.ndarray,
    nominal2: np.ndarray,
    failure2: np.ndarray,
    outdir: Path,
) -> None:
    v3.apply_case4_style()
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.15), constrained_layout=True)
    labels = ["C1\nNom.", "C1\nFail.", "C2\nNom.", "C2\nFail."]
    colors = ["#9ECAE1", "#0072B2", "#FDD0A2", "#D55E00"]
    for index, (axis, (_, ylabel)) in enumerate(zip(axes, base.METRICS)):
        v3.draw_boxplot(
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
        v3.format_axis(axis, ylabel)
    save_figure(
        fig, outdir / "partial_interceptor_failure_reference_v5_nominal_comparison"
    )


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
    case1_v3, categories1, severity1, difficulty1 = v3.transform_distribution(
        nominal1, "case1", np.random.default_rng(args.seed)
    )
    case2_v3, categories2, severity2, difficulty2 = v3.transform_distribution(
        nominal2, "case2", np.random.default_rng(args.seed + 1)
    )
    case1_v4, case2_v4 = v4.refine_v3(case1_v3, case2_v3)
    case1_v5, case2_v5 = refine_v4(case1_v4, case2_v4)
    validate(case1_v5, case2_v5)

    write_reference(
        args.outdir / "case1_partial_failure_reference_v5_n1000.csv",
        nominal1,
        case1_v5,
        categories1,
        severity1,
        difficulty1,
    )
    write_reference(
        args.outdir / "case2_partial_failure_reference_v5_n1000.csv",
        nominal2,
        case2_v5,
        categories2,
        severity2,
        difficulty2,
    )
    diagnostics = (
        v4.boxplot_diagnostics(case1_v4, "case1", "synthetic_reference_v4")
        + v4.boxplot_diagnostics(case1_v5, "case1", DATA_STATUS)
        + v4.boxplot_diagnostics(case2_v4, "case2", "synthetic_reference_v4")
        + v4.boxplot_diagnostics(case2_v5, "case2", DATA_STATUS)
    )
    base.write_table(args.outdir / "v4_v5_boxplot_diagnostics.csv", diagnostics)
    base.write_table(
        args.outdir / "reference_effect_summary_v5.csv",
        v3.effect_rows(nominal1, case1_v5, "case1")
        + v3.effect_rows(nominal2, case2_v5, "case2"),
    )
    plot_failure_only(case1_v5, case2_v5, args.outdir)
    plot_nominal_comparison(nominal1, case1_v5, nominal2, case2_v5, args.outdir)

    manifest = {
        "purpose": "planning target for future independent partial-failure simulations",
        "data_status": "synthetic planning reference; not experimental data",
        "derived_from": "V4 deterministic construction",
        "source_case1": str(args.case1),
        "source_case2": str(args.case2),
        "samples_per_case": args.samples,
        "seed": args.seed,
        "unchanged_from_v4": ["E_co_time_s", "E_t_s"],
        "revised_metrics": ["E_n_g", "E_miss_m"],
        "requested_shape": {
            "E_n_g": "upper whisker below the tail with visible gray outliers",
            "case2_E_n_upper_whisker": "approximately 0.35 g",
            "case1_E_miss_median": "approximately 2.9 m",
            "case2_E_miss_median": "approximately 3.2 m",
            "E_miss_tail": "different non-clipped tails below 4.5 m",
        },
        "distribution_method": "rank-preserving quantile envelope plus seeded heterogeneous jitter and continuous tail compression",
        "simulator_executed": False,
        "policy_evaluation_executed": False,
        "reporting_restriction": (
            "Do not report these values as Monte Carlo trials. Replace them "
            "with independently simulated failure episodes."
        ),
    }
    (args.outdir / "manifest_v5.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.outdir / "README_v5.md").write_text(
        "# 部分拦截器失效参考分布 V5\n\n"
        "本目录仅用于规划后续独立仿真，不是真实蒙特卡洛实验结果。V5 仅调整 "
        "$E_n$ 与 $E_{miss}$；两个工况的 $E_{co-time}$ 和 $E_t$ 与 V4 逐项一致。\n\n"
        "$E_n$ 的中央分布与高侧尾部被分开，使两个工况均出现灰色离群点；Case 2 "
        "上须约为 0.35 g，少量离群点延伸至约 0.40 g。$E_{miss}$ 的中位数分别约为 "
        "2.9 m 和 3.2 m，并采用不同的随机扰动与尾部范围，避免两个工况同时贴近 "
        "4.5 m。`v4_v5_boxplot_diagnostics.csv` 记录所有须线、分位数和离群点数量。\n\n"
        "所有 CSV 和图在用于论文或审稿回复前，必须由独立仿真数据替换。\n",
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__), args.outdir / Path(__file__).name)
    products = sorted(
        path
        for path in args.outdir.iterdir()
        if path.is_file() and path.name != "checksums_v5.sha256"
    )
    (args.outdir / "checksums_v5.sha256").write_text(
        "\n".join(f"{base.sha256(path)}  {path.name}" for path in products) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
