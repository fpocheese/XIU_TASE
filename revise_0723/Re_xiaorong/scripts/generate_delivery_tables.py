#!/usr/bin/env python3
"""Generate reviewer-facing ablation tables from the immutable analysis CSVs."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEPLOYED_RESULT_ROOT = HERE.parent / "raw_results"
STAGING_RESULT_ROOT = (
    HERE.parent / "results" / "formal_paper_ablation_5seed"
)
RESULT_ROOT = (
    DEPLOYED_RESULT_ROOT
    if DEPLOYED_RESULT_ROOT.exists()
    else STAGING_RESULT_ROOT
).resolve()
ANALYSIS = RESULT_ROOT / "analysis"
OUT = (
    HERE.parent / "analysis_tables"
    if DEPLOYED_RESULT_ROOT.exists()
    else HERE / "derived_results"
)

VARIANTS = [
    "full",
    "no_trust",
    "no_gru",
    "no_attention_residual",
]
METRICS = [
    "final_return",
    "return_auc",
    "target_interception_rate",
    "target_sync_rate",
    "all_target_interception",
    "all_target_sync",
    "mean_sync_spread_s",
]
DISPLAY = {
    "full": "Full ART-MAPPO",
    "no_trust": "w/o trust-aware",
    "no_gru": "w/o GRU",
    "no_attention_residual": "w/o attention-residual",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite value encountered: {value}")
    return parsed


def main() -> None:
    seed_rows = read_csv(ANALYSIS / "ablation_seed_level_metrics.csv")
    paired_rows = read_csv(ANALYSIS / "ablation_paired_statistics.csv")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(row["variant"], row["case"])].append(row)

    if set(grouped) != {
        (variant, case)
        for variant in VARIANTS
        for case in ("case1", "case2")
    }:
        raise RuntimeError("Unexpected variant/case coverage")
    if any(len(rows) != 5 for rows in grouped.values()):
        raise RuntimeError("Each variant/case must contain exactly five seeds")

    descriptive_rows: list[dict] = []
    for case in ("case1", "case2"):
        for variant in VARIANTS:
            for metric in METRICS:
                values = [finite_float(row[metric]) for row in grouped[(variant, case)]]
                descriptive_rows.append(
                    {
                        "case": case,
                        "variant": variant,
                        "variant_label": DISPLAY[variant],
                        "metric": metric,
                        "n_seeds": len(values),
                        "mean": f"{statistics.mean(values):.12g}",
                        "sample_sd": f"{statistics.stdev(values):.12g}",
                    }
                )

    paired_index = {
        (row["metric"], row["comparison"]): row
        for row in paired_rows
        if row["scope"] == "pooled"
    }
    comparison_rows: list[dict] = []
    for variant in VARIANTS[1:]:
        comparison = f"full_vs_{variant}"
        for metric in METRICS:
            full_values = [
                finite_float(row[metric])
                for case in ("case1", "case2")
                for row in grouped[("full", case)]
            ]
            ablated_values = [
                finite_float(row[metric])
                for case in ("case1", "case2")
                for row in grouped[(variant, case)]
            ]
            full_mean = statistics.mean(full_values)
            ablated_mean = statistics.mean(ablated_values)
            delta = full_mean - ablated_mean
            relative = (
                100.0 * delta / abs(ablated_mean)
                if abs(ablated_mean) > 1e-15
                else math.nan
            )
            test = paired_index[(metric, comparison)]
            comparison_rows.append(
                {
                    "comparison": comparison,
                    "ablation_label": DISPLAY[variant],
                    "metric": metric,
                    "matched_seed_case_pairs": 10,
                    "full_mean": f"{full_mean:.12g}",
                    "ablation_mean": f"{ablated_mean:.12g}",
                    "full_minus_ablation": f"{delta:.12g}",
                    "relative_change_percent": (
                        f"{relative:.8g}" if math.isfinite(relative) else ""
                    ),
                    "ci95_half_width_of_paired_difference": test[
                        "ci95_half_width"
                    ],
                    "paired_effect_size_dz": test["paired_effect_size_dz"],
                    "signflip_p_raw": test["signflip_p_raw"],
                    "signflip_p_holm": test["signflip_p_holm"],
                }
            )

    case2_rows: list[dict] = []
    for variant in VARIANTS:
        rows = grouped[(variant, "case2")]
        case2_rows.append(
            {
                "variant": variant,
                "variant_label": DISPLAY[variant],
                "target_interception_rate_percent": (
                    f"{100 * statistics.mean(finite_float(r['target_interception_rate']) for r in rows):.3f}"
                ),
                "target_sync_rate_percent": (
                    f"{100 * statistics.mean(finite_float(r['target_sync_rate']) for r in rows):.3f}"
                ),
                "all_target_interception_percent": (
                    f"{100 * statistics.mean(finite_float(r['all_target_interception']) for r in rows):.3f}"
                ),
                "all_target_sync_percent": (
                    f"{100 * statistics.mean(finite_float(r['all_target_sync']) for r in rows):.3f}"
                ),
                "final_return_million": (
                    f"{statistics.mean(finite_float(r['final_return']) for r in rows) / 1e6:.6f}"
                ),
            }
        )

    write_csv(
        OUT / "descriptive_metrics_with_seed_sd.csv",
        descriptive_rows,
        [
            "case",
            "variant",
            "variant_label",
            "metric",
            "n_seeds",
            "mean",
            "sample_sd",
        ],
    )
    write_csv(
        OUT / "pooled_full_vs_ablation.csv",
        comparison_rows,
        [
            "comparison",
            "ablation_label",
            "metric",
            "matched_seed_case_pairs",
            "full_mean",
            "ablation_mean",
            "full_minus_ablation",
            "relative_change_percent",
            "ci95_half_width_of_paired_difference",
            "paired_effect_size_dz",
            "signflip_p_raw",
            "signflip_p_holm",
        ],
    )
    write_csv(
        OUT / "case2_operational_metrics.csv",
        case2_rows,
        [
            "variant",
            "variant_label",
            "target_interception_rate_percent",
            "target_sync_rate_percent",
            "all_target_interception_percent",
            "all_target_sync_percent",
            "final_return_million",
        ],
    )

    print(f"Generated {len(descriptive_rows)} descriptive rows")
    print(f"Generated {len(comparison_rows)} paired-comparison rows")
    print(f"Generated {len(case2_rows)} difficult-case summary rows")


if __name__ == "__main__":
    main()
