#!/usr/bin/env python3
"""Generate a clearly labelled n=1000 synthetic reference by balanced resampling.

This utility is for experiment planning only.  It resamples complete metric rows
from the already selected Case-1 and Case-2 populations, thereby retaining the
observed cross-metric dependence and physical support.  It does not run the
simulator and its outputs must not be reported as independent Monte Carlo data.
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
ORIGIN = "synthetic_balanced_empirical_resample"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
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


def balanced_resample(
    rows: list[dict[str, str]], n: int, rng: np.random.Generator
) -> list[dict]:
    """Repeat each empirical row nearly equally, then shuffle complete rows."""
    if not rows:
        raise ValueError("source population is empty")
    quotient, remainder = divmod(n, len(rows))
    indices = np.repeat(np.arange(len(rows), dtype=int), quotient)
    if remainder:
        indices = np.concatenate(
            [indices, rng.choice(len(rows), size=remainder, replace=False)]
        )
    rng.shuffle(indices)
    occurrence = np.zeros(len(rows), dtype=int)
    output: list[dict] = []
    for sample_id, source_index in enumerate(indices, start=1):
        occurrence[source_index] += 1
        source = rows[int(source_index)]
        output.append(
            {
                "reference_sample_id": sample_id,
                "data_origin": ORIGIN,
                "source_row_index": int(source_index) + 1,
                "source_resample_occurrence": int(occurrence[source_index]),
                **source,
            }
        )
    return output


def summarize(
    rows: list[dict], case: str, dataset_kind: str
) -> list[dict[str, float | int | str]]:
    output = []
    for key, _ in METRICS:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{case}/{dataset_kind}/{key} contains NaN or Inf")
        output.append(
            {
                "case": case,
                "dataset_kind": dataset_kind,
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


def plot_boxplots(case1: list[dict], case2: list[dict], outdir: Path) -> None:
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
                "alpha": 0.30,
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
        "Synthetic reference only—balanced empirical resampling; not independent simulations.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.075, right=0.995, bottom=0.27, top=0.94, wspace=0.52
    )
    stem = outdir / "two_defender_failure_boxplots_reference_synthetic_n1000"
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
    if args.samples <= 0:
        parser.error("--samples must be positive")
    args.outdir.mkdir(parents=True, exist_ok=True)

    original_case1 = read_rows(args.case1)
    original_case2 = read_rows(args.case2)
    rng = np.random.default_rng(args.seed)
    synthetic_case1 = balanced_resample(original_case1, args.samples, rng)
    synthetic_case2 = balanced_resample(original_case2, args.samples, rng)

    case1_path = args.outdir / "case1_reference_synthetic_n1000.csv"
    case2_path = args.outdir / "case2_reference_synthetic_n1000.csv"
    write_rows(case1_path, synthetic_case1)
    write_rows(case2_path, synthetic_case2)

    statistics = (
        summarize(original_case1, "case1", "measured_selected_source")
        + summarize(synthetic_case1, "case1", "synthetic_reference")
        + summarize(original_case2, "case2", "measured_selected_source")
        + summarize(synthetic_case2, "case2", "synthetic_reference")
    )
    stats_path = args.outdir / "reference_synthetic_n1000_statistics.csv"
    write_rows(stats_path, statistics)
    plot_boxplots(synthetic_case1, synthetic_case2, args.outdir)

    manifest = {
        "purpose": "experiment-planning reference only",
        "data_status": "synthetic; not independent simulation trials",
        "method": (
            "balanced empirical resampling of complete rows without jitter; "
            "cross-metric associations and observed support are retained"
        ),
        "random_seed": args.seed,
        "samples_per_case": args.samples,
        "case1_source": str(args.case1),
        "case1_source_rows": len(original_case1),
        "case2_source": str(args.case2),
        "case2_source_rows": len(original_case2),
        "simulator_executed": False,
        "policy_evaluation_executed": False,
        "metric_jitter_added": False,
        "reporting_restriction": (
            "Do not report these rows as Monte Carlo trials or experimental "
            "evidence. Replace them with independently simulated episodes."
        ),
    }
    manifest_path = args.outdir / "reference_synthetic_n1000_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    readme = f"""# 两机失效 n=1000 合成参考数据\n\n+本目录仅用于规划后续真实蒙特卡洛实验。数据不是新增仿真结果。\n+\n+- Case 1 来源：{len(original_case1)} 条当前筛选结果；\n+- Case 2 来源：{len(original_case2)} 条 `E_t <= 45 s` 当前筛选结果；\n+- 每个工况输出：{args.samples} 条；\n+- 方法：对完整指标行做平衡经验重采样，不添加扰动；\n+- 固定随机种子：{args.seed}；\n+- 四项指标的联合对应关系和原始取值范围均被保留。\n+\n+这些数据不得表述为1000次独立蒙特卡洛试验，也不能用于计算ISR。后续真实实验完成后，应使用1000条独立仿真数据整体替换本目录中的参考CSV和图片。\n+"""
    (args.outdir / "README.md").write_text(readme, encoding="utf-8")

    products = sorted(
        path
        for path in args.outdir.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    checksum_text = "\n".join(
        f"{sha256(path)}  {path.name}" for path in products
    ) + "\n"
    (args.outdir / "checksums.sha256").write_text(
        checksum_text, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
