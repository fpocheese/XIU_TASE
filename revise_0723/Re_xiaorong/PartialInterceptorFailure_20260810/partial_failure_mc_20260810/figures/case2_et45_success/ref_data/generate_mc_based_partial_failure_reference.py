#!/usr/bin/env python3
"""Create a biased synthetic reference from the paper's nominal MC data.

The output is intended only as a planning target for future independent
partial-interceptor-failure simulations.  Every product is explicitly marked
as synthetic and must not be reported as experimental evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
BIAS = {
    "case1": {
        "mean_fraction": [0.12, 0.10, 0.08, 0.06],
        "spread_scale": [1.10, 1.10, 1.10, 1.08],
    },
    "case2": {
        "mean_fraction": [0.15, 0.12, 0.10, 0.07],
        "spread_scale": [1.15, 1.12, 1.12, 1.10],
    },
}
JITTER_STD_FRACTION = 0.05
SELECTION_COLUMNS = [0, 1, 2, 4]


def load_nominal(path: Path, n: int) -> np.ndarray:
    values = np.genfromtxt(path, delimiter=",")
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[0] < n or values.shape[1] <= max(SELECTION_COLUMNS):
        raise ValueError(f"insufficient source data in {path}: {values.shape}")
    values = np.asarray(values[:n, SELECTION_COLUMNS], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"NaN or Inf found in {path}")
    return values


def soft_upper_bound(values: np.ndarray, upper: float, tau: float) -> np.ndarray:
    """Smoothly constrain values below upper without a hard clipping pile-up."""
    return upper - tau * np.logaddexp(0.0, (upper - values) / tau)


def transform(
    nominal: np.ndarray, case: str, rng: np.random.Generator
) -> np.ndarray:
    cfg = BIAS[case]
    result = np.empty_like(nominal)
    for metric_index in range(nominal.shape[1]):
        source = nominal[:, metric_index]
        mean = float(np.mean(source))
        std = float(np.std(source, ddof=1))
        result[:, metric_index] = (
            mean * (1.0 + cfg["mean_fraction"][metric_index])
            + cfg["spread_scale"][metric_index] * (source - mean)
            + rng.normal(
                0.0,
                JITTER_STD_FRACTION * std,
                size=source.size,
            )
        )
    result[:, :3] = np.maximum(result[:, :3], 1.0e-8)
    if case == "case2":
        result[:, 3] = soft_upper_bound(result[:, 3], upper=45.0, tau=0.35)
    result[:, 3] = np.round(result[:, 3] / 0.05) * 0.05
    if case == "case2":
        result[:, 3] = np.minimum(result[:, 3], 44.95)
    return result


def write_reference_csv(path: Path, nominal: np.ndarray, reference: np.ndarray) -> None:
    fields = [
        "reference_sample_id",
        "data_status",
        *[f"source_{key}" for key, _ in METRICS],
        *[key for key, _ in METRICS],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (source, transformed) in enumerate(
            zip(nominal, reference), start=1
        ):
            row: dict[str, str | int | float] = {
                "reference_sample_id": index,
                "data_status": "synthetic_planning_reference",
            }
            for (key, _), value in zip(METRICS, source):
                row[f"source_{key}"] = float(value)
            for (key, _), value in zip(METRICS, transformed):
                row[key] = float(value)
            writer.writerow(row)


def summarize(values: np.ndarray, case: str, kind: str) -> list[dict]:
    rows = []
    for metric_index, (key, _) in enumerate(METRICS):
        column = values[:, metric_index]
        rows.append(
            {
                "case": case,
                "dataset_kind": kind,
                "n": int(column.size),
                "metric": key,
                "mean": float(np.mean(column)),
                "std": float(np.std(column, ddof=1)),
                "median": float(np.median(column)),
                "q1": float(np.quantile(column, 0.25)),
                "q3": float(np.quantile(column, 0.75)),
                "min": float(np.min(column)),
                "max": float(np.max(column)),
            }
        )
    return rows


def write_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_reference(case1: np.ndarray, case2: np.ndarray, outdir: Path) -> None:
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
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.38))
    for panel, (axis, (_, ylabel)) in enumerate(zip(axes.flat, METRICS)):
        boxes = axis.boxplot(
            [case1[:, panel], case2[:, panel]],
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
                "alpha": 0.45,
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
    fig.text(
        0.5,
        0.025,
        "Synthetic planning reference derived from nominal Monte Carlo distributions.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.075, right=0.995, bottom=0.27, top=0.94, wspace=0.52
    )
    stem = outdir / "partial_interceptor_failure_mc_based_reference_boxplots"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case1", type=Path, required=True)
    parser.add_argument("--case2", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    nominal_case1 = load_nominal(args.case1, args.samples)
    nominal_case2 = load_nominal(args.case2, args.samples)
    reference_case1 = transform(
        nominal_case1, "case1", np.random.default_rng(args.seed + 1)
    )
    reference_case2 = transform(
        nominal_case2, "case2", np.random.default_rng(args.seed + 2)
    )
    if not np.isfinite(reference_case1).all() or not np.isfinite(reference_case2).all():
        raise RuntimeError("generated reference contains NaN or Inf")

    case1_csv = args.outdir / "case1_partial_failure_reference_n1000.csv"
    case2_csv = args.outdir / "case2_partial_failure_reference_n1000.csv"
    write_reference_csv(case1_csv, nominal_case1, reference_case1)
    write_reference_csv(case2_csv, nominal_case2, reference_case2)

    statistics = (
        summarize(nominal_case1, "case1", "paper_nominal_mc_source")
        + summarize(reference_case1, "case1", "synthetic_failure_reference")
        + summarize(nominal_case2, "case2", "paper_nominal_mc_source")
        + summarize(reference_case2, "case2", "synthetic_failure_reference")
    )
    write_table(args.outdir / "reference_statistics.csv", statistics)
    plot_reference(reference_case1, reference_case2, args.outdir)

    manifest = {
        "purpose": "reference target for future independent failure simulations",
        "data_status": "synthetic planning reference; not experimental data",
        "source_case1": str(args.case1),
        "source_case2": str(args.case2),
        "source_relation": "ART-MAPPO data used by Fig. mc_boxplots",
        "samples_per_case": args.samples,
        "seed": args.seed,
        "transformation": {
            "formula": (
                "reference = source_mean*(1+mean_fraction) + "
                "spread_scale*(source-source_mean) + Gaussian jitter"
            ),
            "case1": BIAS["case1"],
            "case2": BIAS["case2"],
            "jitter_std_fraction_of_source_std": JITTER_STD_FRACTION,
            "case2_E_t_upper_bound_s": 45.0,
            "case2_E_t_upper_bound_method": "smooth saturation, then 0.05-s quantization",
        },
        "simulator_executed": False,
        "policy_evaluation_executed": False,
        "reporting_restriction": (
            "Do not report the generated values as Monte Carlo trials or use "
            "them as evidence. Replace them with independent simulation data."
        ),
    }
    (args.outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    readme = """# 部分拦截器失效参考数据（基于论文蒙特卡洛分布）

本目录只用于规划后续真实实验，不是新增仿真结果。

- 基准：论文 Fig. `mc_boxplots` 所用 ART-MAPPO Case 1/Case 2 各1000条数据；
- 方法：保留原始分布形状和跨指标对应关系，加入预先记录的均值偏差、离散度放大和小幅随机扰动；
- Case 2 的参考打击时间通过平滑上界限制在45 s以内；
- 图中的空心灰点仍按标准1.5-IQR规则表示离群点；
- 生成值不得作为1000次独立蒙特卡洛试验或论文实验依据。

真实两机失效实验完成后，应使用独立仿真数据替换本目录全部参考CSV、统计表和图片。
"""
    (args.outdir / "README.md").write_text(readme, encoding="utf-8")

    products = sorted(
        path
        for path in args.outdir.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    (args.outdir / "checksums.sha256").write_text(
        "\n".join(f"{sha256(path)}  {path.name}" for path in products) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
