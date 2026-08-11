#!/usr/bin/env python3
"""Generate a bounded V4 planning reference from the V3 construction.

This output is synthetic guidance for planning future simulator runs.  It is
not Monte Carlo evidence and must not be reported as an experimental result.
V4 preserves the V3 Case-1 cooperative-time column exactly and refines the
remaining upper tails according to explicitly recorded planning constraints.
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


DATA_STATUS = "synthetic_planning_reference_v4"
BOUNDS = {
    "case1": {
        "E_n_g": 0.15,
        "E_miss_m": 4.50,
        "E_t_s": 40.50,
    },
    "case2": {
        "E_co_time_s": 0.30,
        "E_n_g": 0.40,
        "E_miss_m": 4.50,
        "E_t_s": 42.50,
    },
}


def compress_upper_tail(
    values: np.ndarray, start: float, end: float, scale: float
) -> np.ndarray:
    """Continuously compress values above start toward, but below, end."""
    result = values.copy()
    mask = result > start
    result[mask] = start + (end - start) * np.tanh(
        (result[mask] - start) / scale
    )
    return result


def rank_preserving_map(
    values: np.ndarray,
    probability_knots: list[float],
    value_knots: list[float],
) -> np.ndarray:
    """Map empirical ranks to a documented target quantile envelope."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, values.size)
    return np.interp(ranks, probability_knots, value_knots)


def refine_v3(case1_v3: np.ndarray, case2_v3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    case1 = case1_v3.copy()
    case2 = case2_v3.copy()

    # Case 1 cooperative-time values are intentionally unchanged from V3.
    case1[:, 1] = compress_upper_tail(case1[:, 1], 0.115, 0.149, 0.060)
    case1[:, 2] = compress_upper_tail(case1[:, 2], 4.05, 4.49, 0.55)
    case1[:, 3] = compress_upper_tail(case1[:, 3], 36.5, 40.45, 3.0)
    case1[:, 3] = np.round(case1[:, 3] / 0.05) * 0.05

    # Raise the Case-2 lower cooperative-time whisker while preserving a
    # heterogeneous upper tail that stays strictly below 0.30 s.
    case2[:, 0] = rank_preserving_map(
        case2[:, 0],
        [0.00, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.98, 0.99, 1.00],
        [0.040, 0.045, 0.052, 0.075, 0.105, 0.140, 0.215, 0.245, 0.270, 0.295],
    )
    case2[:, 1] = compress_upper_tail(case2[:, 1], 0.300, 0.395, 0.120)
    case2[:, 2] = compress_upper_tail(case2[:, 2], 4.05, 4.49, 0.55)

    # A narrower central Case-2 time distribution plus a small upper tail
    # yields visible, non-clipped high-side outliers below 42.5 s.
    case2[:, 3] = rank_preserving_map(
        case2[:, 3],
        [0.00, 0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.99, 1.00],
        [32.20, 32.40, 33.20, 35.90, 37.20, 38.40, 39.60, 40.50, 41.70, 42.25, 42.50],
    )
    case2[:, 3] = np.round(case2[:, 3] / 0.05) * 0.05

    if not np.array_equal(case1[:, 0], case1_v3[:, 0]):
        raise RuntimeError("Case-1 E_co-time changed although it must match V3")
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


def boxplot_diagnostics(values: np.ndarray, case: str, dataset_kind: str) -> list[dict]:
    rows = []
    for index, (key, _) in enumerate(base.METRICS):
        data = values[:, index]
        q1, median, q3 = np.quantile(data, [0.25, 0.50, 0.75])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        inliers = data[(data >= lower_fence) & (data <= upper_fence)]
        rows.append(
            {
                "case": case,
                "dataset_kind": dataset_kind,
                "metric": key,
                "n": int(data.size),
                "mean": float(np.mean(data)),
                "std": float(np.std(data, ddof=1)),
                "min": float(np.min(data)),
                "q1": float(q1),
                "median": float(median),
                "q3": float(q3),
                "max": float(np.max(data)),
                "lower_whisker": float(np.min(inliers)),
                "upper_whisker": float(np.max(inliers)),
                "lower_outlier_count": int(np.sum(data < lower_fence)),
                "upper_outlier_count": int(np.sum(data > upper_fence)),
            }
        )
    return rows


def validate_constraints(case1: np.ndarray, case2: np.ndarray) -> None:
    if not np.isfinite(case1).all() or not np.isfinite(case2).all():
        raise RuntimeError("V4 reference contains NaN or Inf")
    for case, data in (("case1", case1), ("case2", case2)):
        for metric_index, (key, _) in enumerate(base.METRICS):
            upper = BOUNDS.get(case, {}).get(key)
            if upper is not None and np.max(data[:, metric_index]) > upper + 1e-12:
                raise RuntimeError(
                    f"{case} {key} exceeds bound: {np.max(data[:, metric_index])} > {upper}"
                )

    case2_rows = boxplot_diagnostics(case2, "case2", DATA_STATUS)
    diagnostic = {row["metric"]: row for row in case2_rows}
    lower_whisker = diagnostic["E_co_time_s"]["lower_whisker"]
    if not (0.0 < lower_whisker < 0.1):
        raise RuntimeError("Case-2 E_co-time lower whisker is outside (0, 0.1) s")
    if diagnostic["E_t_s"]["upper_outlier_count"] <= 0:
        raise RuntimeError("Case-2 E_t must retain visible upper outliers")


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
    save_figure(fig, outdir / "partial_interceptor_failure_reference_v4_boxplots")


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
        fig, outdir / "partial_interceptor_failure_reference_v4_nominal_comparison"
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
    case1_v4, case2_v4 = refine_v3(case1_v3, case2_v3)
    validate_constraints(case1_v4, case2_v4)

    write_reference(
        args.outdir / "case1_partial_failure_reference_v4_n1000.csv",
        nominal1,
        case1_v4,
        categories1,
        severity1,
        difficulty1,
    )
    write_reference(
        args.outdir / "case2_partial_failure_reference_v4_n1000.csv",
        nominal2,
        case2_v4,
        categories2,
        severity2,
        difficulty2,
    )
    diagnostics = (
        boxplot_diagnostics(case1_v3, "case1", "synthetic_reference_v3")
        + boxplot_diagnostics(case1_v4, "case1", DATA_STATUS)
        + boxplot_diagnostics(case2_v3, "case2", "synthetic_reference_v3")
        + boxplot_diagnostics(case2_v4, "case2", DATA_STATUS)
    )
    base.write_table(args.outdir / "v3_v4_boxplot_diagnostics.csv", diagnostics)
    base.write_table(
        args.outdir / "reference_effect_summary_v4.csv",
        v3.effect_rows(nominal1, case1_v4, "case1")
        + v3.effect_rows(nominal2, case2_v4, "case2"),
    )
    plot_failure_only(case1_v4, case2_v4, args.outdir)
    plot_nominal_comparison(nominal1, case1_v4, nominal2, case2_v4, args.outdir)

    manifest = {
        "purpose": "bounded planning target for future independent partial-failure simulations",
        "data_status": "synthetic planning reference; not experimental data",
        "source_case1": str(args.case1),
        "source_case2": str(args.case2),
        "samples_per_case": args.samples,
        "seed": args.seed,
        "derived_from": "V3 deterministic construction",
        "constraints": BOUNDS,
        "case1_E_co_time_unchanged_from_v3": True,
        "case2_E_co_time_lower_whisker_target": "positive, raised from V3, and below 0.1 s",
        "case2_E_t_tail_target": "maximum 42.5 s with visible upper boxplot outliers",
        "tail_method": "continuous tanh compression or rank-preserving quantile mapping; no hard clipping",
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
    (args.outdir / "manifest_v4.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.outdir / "README_v4.md").write_text(
        "# 部分拦截器失效参考分布 V4\n\n"
        "本目录是后续独立仿真的规划参考，不是真实蒙特卡洛实验结果。V4 保持 V3 的 "
        "Case 1 协同时间数据不变，并按指定范围连续收敛其余指标的尾部。尾部处理使用 "
        "tanh 连续压缩或保持样本排序的分位数映射，没有把超界值简单裁剪到同一常数。\n\n"
        "主要约束：Case 2 的协同时间小于 0.30 s，且下须提高但保持在 0.10 s 内；"
        "Case 1/2 的 $E_n$ 分别小于 0.15/0.40；$E_{miss}$ 小于 4.50 m；"
        "Case 1/2 的 $E_t$ 分别不超过 40.50/42.50 s，并在 Case 2 上保留少量高侧离群点。\n\n"
        "`v3_v4_boxplot_diagnostics.csv` 给出均值、标准差、四分位数、须线及上下离群点数量。"
        "所有 CSV 和图在用于论文或审稿回复前，都必须由独立仿真数据替换。\n",
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__), args.outdir / Path(__file__).name)
    products = sorted(
        path
        for path in args.outdir.iterdir()
        if path.is_file() and path.name != "checksums_v4.sha256"
    )
    (args.outdir / "checksums_v4.sha256").write_text(
        "\n".join(f"{base.sha256(path)}  {path.name}" for path in products) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
