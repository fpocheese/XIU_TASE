#!/usr/bin/env python3
"""Validate and summarize two-defender-failure Monte Carlo outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


METRICS = [
    ("E_co_time_s", r"$E_{\mathrm{co\text{-}time}}$ (s)"),
    ("E_n_g", r"$E_n$ (g)"),
    ("E_miss_m", r"$E_{\mathrm{miss}}$ (m)"),
    ("E_t_s", r"$E_t$ (s)"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wilson(successes: int, total: int, z: float = 1.9599639845) -> tuple[float, float]:
    p = successes / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / den
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / den
    return center - radius, center + radius


def parse_failed_ids(text: str) -> list[int]:
    return [int(item) for item in text.split(";") if item != ""]


def summarize_case(case_dir: Path) -> tuple[dict, list[dict], dict, list[dict]]:
    episodes = read_csv(case_dir / "episodes.csv")
    targets = read_csv(case_dir / "targets.csv")
    case = episodes[0]["case"] if episodes else case_dir.name
    errors: list[str] = []

    if len(episodes) != 100:
        errors.append(f"expected 100 episodes, found {len(episodes)}")
    seeds = [int(row["seed"]) for row in episodes]
    if len(seeds) != len(set(seeds)):
        errors.append("episode seeds are not unique")
    for row in episodes:
        failed = parse_failed_ids(row["failed_defender_ids"])
        if int(row["failed_count"]) != 2 or len(failed) != 2 or len(set(failed)) != 2:
            errors.append(f"episode {row['episode']} does not contain two unique failed defenders")
        success = bool(int(row["interception_success"]))
        for key, _ in METRICS:
            value = float(row[key])
            if success and not np.isfinite(value):
                errors.append(f"episode {row['episode']} has nonfinite {key} despite success")
            if not success and np.isfinite(value):
                errors.append(f"episode {row['episode']} has finite {key} despite failure")

    successes = [row for row in episodes if int(row["interception_success"])]
    success_count = len(successes)
    isr_low, isr_high = wilson(success_count, len(episodes))
    summary = {
        "case": case,
        "episodes": len(episodes),
        "failed_defenders_per_episode": 2,
        "interception_success_count": success_count,
        "interception_success_rate": success_count / len(episodes),
        "interception_success_ci95_low": isr_low,
        "interception_success_ci95_high": isr_high,
        "all_active_defenders_hit_rate": np.mean(
            [int(row["all_active_defenders_hit"]) for row in episodes]
        ),
        "cooperative_success_rate": np.mean(
            [int(row["cooperative_success"]) for row in episodes]
        ),
    }
    descriptive: list[dict] = []
    for key, label in METRICS:
        values = np.asarray([float(row[key]) for row in successes], dtype=float)
        stats = {
            "case": case,
            "metric": key,
            "label": label,
            "n": values.size,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "median": float(np.median(values)),
            "q1": float(np.quantile(values, 0.25)),
            "q3": float(np.quantile(values, 0.75)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        descriptive.append(stats)
        for field in ("mean", "std", "median", "q1", "q3"):
            summary[f"{key}_{field}"] = stats[field]

    targets_by_episode: dict[int, list[dict[str, str]]] = {}
    for row in targets:
        targets_by_episode.setdefault(int(row["episode"]), []).append(row)
    structural = guidance = both = 0
    for row in episodes:
        if int(row["interception_success"]):
            continue
        rows = targets_by_episode.get(int(row["episode"]), [])
        has_structural = any(
            int(target["active_defenders"]) == 0 and not int(target["target_covered"])
            for target in rows
        )
        has_guidance = any(
            int(target["active_defenders"]) > 0 and not int(target["target_covered"])
            for target in rows
        )
        structural += int(has_structural)
        guidance += int(has_guidance)
        both += int(has_structural and has_guidance)
    failure_causes = {
        "case": case,
        "failed_episodes": len(episodes) - success_count,
        "episodes_with_zero_active_defenders_for_a_target": structural,
        "episodes_with_uncovered_target_despite_active_defender": guidance,
        "episodes_with_both_failure_modes": both,
    }
    target_breakdown: list[dict] = []
    for target_id in range(8):
        selected = [
            row for row in targets if int(row["target_index"]) == target_id
        ]
        uncovered = [row for row in selected if not int(row["target_covered"])]
        target_breakdown.append(
            {
                "case": case,
                "target_index": target_id,
                "assigned_defenders": int(selected[0]["assigned_defenders"]),
                "episodes_uncovered": len(uncovered),
                "episodes_zero_active_defenders": sum(
                    int(row["active_defenders"]) == 0 for row in uncovered
                ),
                "episodes_uncovered_with_active_defender": sum(
                    int(row["active_defenders"]) > 0 for row in uncovered
                ),
            }
        )
    validation = {
        "case": case,
        "valid": not errors,
        "errors": errors,
        "episode_count": len(episodes),
        "unique_seed_count": len(set(seeds)),
        "successful_metric_count": success_count,
        "nan_inf_check": "passed" if not errors else "see errors",
    }
    return (
        summary,
        descriptive,
        {**failure_causes, "_validation": validation},
        target_breakdown,
    )


def format_mean_std(summary: dict, key: str) -> str:
    return f"{summary[f'{key}_mean']:.4f} $\\pm$ {summary[f'{key}_std']:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case1", type=Path, required=True)
    parser.add_argument("--case2", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    descriptives: list[dict] = []
    causes: list[dict] = []
    validations: list[dict] = []
    target_breakdowns: list[dict] = []
    for case_dir in (args.case1, args.case2):
        summary, descriptive, cause, target_breakdown = summarize_case(case_dir)
        validation = cause.pop("_validation")
        summaries.append(summary)
        descriptives.extend(descriptive)
        causes.append(cause)
        validations.append(validation)
        target_breakdowns.extend(target_breakdown)

    write_csv(args.outdir / "combined_summary.csv", summaries)
    write_csv(args.outdir / "metric_descriptive_statistics.csv", descriptives)
    write_csv(args.outdir / "failure_cause_summary.csv", causes)
    write_csv(args.outdir / "uncovered_target_summary.csv", target_breakdowns)
    (args.outdir / "validation_report.json").write_text(
        json.dumps(
            {
                "valid": all(item["valid"] for item in validations),
                "cases": validations,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\caption{Performance under two randomly unavailable interceptors.}",
        r"\label{tab:two_interceptor_failure}",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Case & ISR (\%) & $E_{\mathrm{co\text{-}time}}$ (s) & $E_n$ (g) & $E_{\mathrm{miss}}$ (m) & $E_t$ (s) \\",
        r"\midrule",
    ]
    for summary in summaries:
        lines.append(
            f"{summary['case'].replace('case', 'Case ')} & "
            f"{100.0 * summary['interception_success_rate']:.1f} & "
            f"{format_mean_std(summary, 'E_co_time_s')} & "
            f"{format_mean_std(summary, 'E_n_g')} & "
            f"{format_mean_std(summary, 'E_miss_m')} & "
            f"{format_mean_std(summary, 'E_t_s')} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (args.outdir / "partial_failure_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    if not all(item["valid"] for item in validations):
        raise SystemExit("validation failed; inspect validation_report.json")


if __name__ == "__main__":
    main()
