#!/usr/bin/env python3
"""Audit and analyse the original 2-D ART-MAPPO Monte Carlo archives.

The source evaluator writes one five-column row only after the strict success
flag is raised.  Consequently, this program never manufactures per-episode
records for unsuccessful trials.  It reports their aggregate incidence from
the manuscript and analyses delayed cooperative engagement in the archived
successful trials.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


SOURCE_ROOT = Path(
    "/home/uav/00gao_xueshu/DT_PAPER/guidance_pic_code/"
    "test全面数据版本v1/test"
)

CASES = {
    "Case 1": {
        "path": SOURCE_ROOT
        / "mappo_success_nopn"
        / "mappo_success_nopn_eval.txt",
        "trials": 1000,
        "successes": 987,
        "scenario": "evasive attack pattern",
    },
    "Case 2": {
        "path": SOURCE_ROOT
        / "mappo_success_sin"
        / "sinmappo_eval"
        / "agentseval.txt",
        "trials": 1000,
        "successes": 971,
        "scenario": "continuous weaving attack pattern",
    },
}

COLUMNS = ["E_co_time_s", "E_n_g", "E_miss_m", "control_energy", "E_t_s"]
PERCENTILES = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.25,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def make_figure(cohorts: dict[str, np.ndarray], outdir: Path) -> None:
    configure_plotting()
    colors = {"Case 1": "#0072B2", "Case 2": "#D55E00"}
    markers = {"Case 1": "o", "Case 2": "s"}
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65), constrained_layout=True)

    ax = axes[0]
    for case, data in cohorts.items():
        x = np.sort(data[:, 0])
        survival = (len(x) - np.arange(len(x))) / len(x)
        ax.step(x, survival, where="post", color=colors[case], label=case)
        q95 = float(np.quantile(x, 0.95))
        ax.plot(q95, 0.05, markers[case], color=colors[case], ms=4.2)
    ax.axvline(0.10, color="0.35", lw=0.9, ls="--", label="0.10-s diagnostic level")
    ax.set_yscale("log")
    ax.set_ylim(8e-4, 1.1)
    ax.set_xlabel(r"Temporal-coordination error $E_{co\mathrm{-}time}$ (s)")
    ax.set_ylabel("Exceedance probability")
    ax.grid(True, which="both", color="0.88", lw=0.45)
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.01, 1.02, "(a)", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    rng = np.random.default_rng(20260811)
    for case, data in cohorts.items():
        jitter = rng.normal(0, 0.02 if case == "Case 1" else 0.025, len(data))
        ax.scatter(
            data[:, 0],
            data[:, 4] + jitter,
            s=8,
            alpha=0.28,
            color=colors[case],
            linewidths=0,
            label=case,
        )
        slope, intercept = np.polyfit(data[:, 0], data[:, 4], 1)
        xx = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
        ax.plot(xx, slope * xx + intercept, color=colors[case])
    ax.axvline(0.10, color="0.35", lw=0.9, ls="--")
    ax.set_xlabel(r"Temporal-coordination error $E_{co\mathrm{-}time}$ (s)")
    ax.set_ylabel(r"Engagement duration $E_t$ (s)")
    ax.grid(True, color="0.88", lw=0.45)
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.01, 1.02, "(b)", transform=ax.transAxes, fontweight="bold")

    for ext, kwargs in {
        "pdf": {},
        "svg": {},
        "png": {"dpi": 600},
    }.items():
        fig.savefig(outdir / f"failure_delay_analysis.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def f6(x: float) -> str:
    return f"{x:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    outdir = args.output
    outdir.mkdir(parents=True, exist_ok=True)

    audit_rows: list[dict] = []
    episode_rows: list[dict] = []
    archive_summary_rows: list[dict] = []
    delay_summary_rows: list[dict] = []
    tail_comparison_rows: list[dict] = []
    correlation_rows: list[dict] = []
    top_rows: list[dict] = []
    failure_rows: list[dict] = []
    cohorts: dict[str, np.ndarray] = {}

    for case, cfg in CASES.items():
        source = cfg["path"]
        raw = np.loadtxt(source, delimiter=",")
        if raw.ndim != 2 or raw.shape[1] != len(COLUMNS):
            raise ValueError(f"Unexpected shape for {source}: {raw.shape}")
        if not np.isfinite(raw).all():
            raise ValueError(f"NaN/Inf detected in {source}")

        n_success = cfg["successes"]
        if len(raw) < n_success:
            raise ValueError(f"{source} has fewer rows than the paper success count")
        cohort = raw[:n_success].copy()
        cohorts[case] = cohort
        q95 = float(np.quantile(cohort[:, 0], 0.95))

        audit_rows.append(
            {
                "case": case,
                "source_file": str(source),
                "sha256": sha256(source),
                "rows_in_archive": len(raw),
                "columns": raw.shape[1],
                "nan_count": int(np.isnan(raw).sum()),
                "inf_count": int(np.isinf(raw).sum()),
                "paper_trials": cfg["trials"],
                "paper_successes": n_success,
                "paper_failures": cfg["trials"] - n_success,
                "paper_matched_rows_used": n_success,
                "selection_note": "First N successful rows; archival file lacks episode IDs and contains more than one batch",
            }
        )

        lo, hi = wilson_interval(n_success, cfg["trials"])
        failure_rows.append(
            {
                "case": case,
                "total_trials": cfg["trials"],
                "strict_successes": n_success,
                "unsuccessful_trials": cfg["trials"] - n_success,
                "ISR_percent": 100 * n_success / cfg["trials"],
                "ISR_Wilson95_low_percent": 100 * lo,
                "ISR_Wilson95_high_percent": 100 * hi,
                "failed_trajectory_saved": False,
                "member_level_cause_identifiable": False,
                "reason": "Evaluator writes the five metrics only when break_flag is true",
            }
        )

        for row_index, row in enumerate(cohort, start=1):
            rec = {
                "case": case,
                "source_row_1based": row_index,
                **{name: float(value) for name, value in zip(COLUMNS, row)},
                "empirical_delayed_tail_top5pct": bool(row[0] >= q95),
                "above_0p10s_diagnostic_level": bool(row[0] > 0.10),
                "strict_success_record": True,
            }
            episode_rows.append(rec)

        for cohort_name, data in [("paper_matched", cohort), ("full_archive", raw)]:
            for col_idx, metric in enumerate(COLUMNS):
                values = data[:, col_idx]
                qs = np.quantile(values, PERCENTILES)
                archive_summary_rows.append(
                    {
                        "case": case,
                        "cohort": cohort_name,
                        "metric": metric,
                        "n": len(values),
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)),
                        **{f"q{int(q*1000):03d}": float(v) for q, v in zip(PERCENTILES, qs)},
                    }
                )

        eco = cohort[:, 0]
        delayed = cohort[eco >= q95]
        remainder = cohort[eco < q95]
        q = np.quantile(eco, PERCENTILES)
        delay_summary_rows.append(
            {
                "case": case,
                "successful_records_analyzed": len(cohort),
                "E_co_time_mean_s": eco.mean(),
                "E_co_time_median_s": np.median(eco),
                "E_co_time_q90_s": np.quantile(eco, 0.90),
                "E_co_time_q95_s": q95,
                "E_co_time_q99_s": np.quantile(eco, 0.99),
                "E_co_time_max_s": eco.max(),
                "top5pct_count_including_ties": len(delayed),
                "count_above_0p10s": int(np.sum(eco > 0.10)),
                "percent_above_0p10s": 100 * np.mean(eco > 0.10),
                "count_above_0p15s": int(np.sum(eco > 0.15)),
                "count_above_0p20s": int(np.sum(eco > 0.20)),
            }
        )

        for j, metric in enumerate(COLUMNS):
            tail_mean = delayed[:, j].mean()
            rest_mean = remainder[:, j].mean()
            tail_comparison_rows.append(
                {
                    "case": case,
                    "tail_definition": f"E_co_time >= case-specific q95 ({q95:.6f} s)",
                    "metric": metric,
                    "tail_n": len(delayed),
                    "remainder_n": len(remainder),
                    "tail_mean": tail_mean,
                    "tail_median": np.median(delayed[:, j]),
                    "remainder_mean": rest_mean,
                    "remainder_median": np.median(remainder[:, j]),
                    "mean_difference": tail_mean - rest_mean,
                    "relative_mean_change_percent": 100 * (tail_mean - rest_mean) / rest_mean,
                }
            )

        for j, metric in enumerate(COLUMNS[1:], start=1):
            correlation_rows.append(
                {
                    "case": case,
                    "x_metric": "E_co_time_s",
                    "y_metric": metric,
                    "n": len(cohort),
                    "pearson_r": pearsonr(eco, cohort[:, j])[0],
                    "spearman_rho": spearmanr(eco, cohort[:, j])[0],
                }
            )

        top_idx = np.argsort(eco)[-10:][::-1]
        for rank, idx in enumerate(top_idx, start=1):
            top_rows.append(
                {
                    "case": case,
                    "delay_rank": rank,
                    "source_row_1based": int(idx + 1),
                    **{name: float(value) for name, value in zip(COLUMNS, cohort[idx])},
                    "strict_success_record": True,
                }
            )

    write_csv(outdir / "source_data_audit.csv", audit_rows)
    write_csv(outdir / "paper_matched_success_metrics.csv", episode_rows)
    write_csv(outdir / "metric_distribution_summary.csv", archive_summary_rows)
    write_csv(outdir / "delay_summary.csv", delay_summary_rows)
    write_csv(outdir / "delayed_tail_comparison.csv", tail_comparison_rows)
    write_csv(outdir / "delay_metric_correlations.csv", correlation_rows)
    write_csv(outdir / "top10_delayed_success_cases.csv", top_rows)
    write_csv(outdir / "interception_failure_inventory.csv", failure_rows)
    make_figure(cohorts, outdir)

    result = {
        "analysis_date": "2026-08-11",
        "metric_columns": COLUMNS,
        "primary_delay_definition": "case-specific upper 5% of E_co_time among strict successful records",
        "secondary_diagnostic_level_s": 0.10,
        "important_limit": "No per-episode failed trajectories are present in the supplied eval archives.",
        "source_audit": audit_rows,
        "failure_inventory": failure_rows,
        "delay_summary": delay_summary_rows,
    }
    (outdir / "analysis_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
