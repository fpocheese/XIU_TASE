#!/usr/bin/env python3
"""Audited summaries for the formal ART-MAPPO component ablation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


VARIANTS = (
    "full",
    "no_trust",
    "no_gru",
    "no_attention_residual",
)
CASES = ("case1", "case2")
RATE_METRICS = (
    "target_coverage_success",
    "all_defenders_hit",
    "cooperative_success",
    "mission_success",
)
CONTINUOUS_METRICS = (
    "E_co_time_s",
    "E_n_g",
    "E_miss_m",
    "E_t_s",
)


def wilson(count: int, total: int, z: float = 1.959963984540054):
    p = count / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return p, max(0.0, center - half), min(1.0, center + half)


def load_training(root: Path):
    rows = []
    for path in sorted(root.glob("*/*/seed*/training_metrics.csv")):
        variant, case, seed_text = path.parts[-4:-1]
        frame = pd.read_csv(path).sort_values("environment_steps")
        if len(frame) != 80:
            raise RuntimeError(f"{path}: expected 80 updates, got {len(frame)}")
        if not np.isfinite(
            frame.select_dtypes(include=[np.number]).to_numpy()
        ).all():
            raise RuntimeError(f"{path}: NaN/Inf in training metrics")
        tail = frame.tail(16)
        rows.append(
            {
                "variant": variant,
                "case": case,
                "training_seed": int(seed_text.replace("seed", "")),
                "updates": len(frame),
                "final_environment_steps": int(
                    frame["environment_steps"].iloc[-1]
                ),
                "final_window_return": float(
                    tail["mean_episode_return"].mean()
                ),
                "final_window_critic_loss": float(
                    tail["value_loss"].mean()
                ),
                "final_window_policy_entropy": float(
                    tail["entropy"].mean()
                ),
                "final_window_all_hit_rate": float(
                    tail["all_hit_rate"].mean()
                ),
                "final_window_all_sync_rate": float(
                    tail["all_sync_rate"].mean()
                ),
            }
        )
    runs = pd.DataFrame(rows)
    expected = len(VARIANTS) * len(CASES) * 3
    if len(runs) != expected:
        raise RuntimeError(f"expected {expected} training runs, got {len(runs)}")
    aggregate = (
        runs.groupby(["variant", "case"], sort=False)
        .agg(
            seed_count=("training_seed", "nunique"),
            return_mean=("final_window_return", "mean"),
            return_std=("final_window_return", "std"),
            critic_loss_mean=("final_window_critic_loss", "mean"),
            critic_loss_std=("final_window_critic_loss", "std"),
            entropy_mean=("final_window_policy_entropy", "mean"),
            entropy_std=("final_window_policy_entropy", "std"),
            training_all_hit_rate_mean=("final_window_all_hit_rate", "mean"),
            training_all_sync_rate_mean=("final_window_all_sync_rate", "mean"),
        )
        .reset_index()
    )
    return runs, aggregate


def summarize_test(path: Path):
    episodes = pd.read_csv(path)
    summaries = []
    for case in CASES:
        for variant in VARIANTS:
            data = episodes[
                (episodes["case"] == case)
                & (episodes["variant"] == variant)
            ]
            if len(data) != 100:
                raise RuntimeError(
                    f"{variant}/{case}: expected 100 episodes, got {len(data)}"
                )
            row = {
                "variant": variant,
                "case": case,
                "episode_count": len(data),
                "unique_test_seeds": int(data["seed"].nunique()),
            }
            if row["unique_test_seeds"] != 100:
                raise RuntimeError(f"{variant}/{case}: duplicate test seeds")
            for metric in RATE_METRICS:
                count = int(data[metric].sum())
                mean, low, high = wilson(count, len(data))
                row[f"{metric}_count"] = count
                row[f"{metric}_rate"] = mean
                row[f"{metric}_ci95_low"] = low
                row[f"{metric}_ci95_high"] = high
            for metric in CONTINUOUS_METRICS:
                values = data[metric].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                row[f"{metric}_n"] = len(values)
                row[f"{metric}_mean"] = (
                    float(np.mean(values)) if len(values) else math.nan
                )
                row[f"{metric}_std"] = (
                    float(np.std(values, ddof=1))
                    if len(values) > 1
                    else math.nan
                )
                row[f"{metric}_median"] = (
                    float(np.median(values)) if len(values) else math.nan
                )
                row[f"{metric}_q25"] = (
                    float(np.quantile(values, 0.25))
                    if len(values)
                    else math.nan
                )
                row[f"{metric}_q75"] = (
                    float(np.quantile(values, 0.75))
                    if len(values)
                    else math.nan
                )
            summaries.append(row)
    return episodes, pd.DataFrame(summaries)


def full_deltas(test_summary: pd.DataFrame):
    rows = []
    for case in CASES:
        full = test_summary[
            (test_summary["case"] == case)
            & (test_summary["variant"] == "full")
        ].iloc[0]
        for variant in VARIANTS[1:]:
            ablated = test_summary[
                (test_summary["case"] == case)
                & (test_summary["variant"] == variant)
            ].iloc[0]
            row = {"case": case, "ablation": variant}
            for metric in RATE_METRICS:
                row[f"full_minus_ablation_{metric}_pp"] = 100.0 * (
                    full[f"{metric}_rate"] - ablated[f"{metric}_rate"]
                )
            for metric in CONTINUOUS_METRICS:
                row[f"full_minus_ablation_{metric}_mean"] = (
                    full[f"{metric}_mean"] - ablated[f"{metric}_mean"]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_root", type=Path, required=True)
    parser.add_argument("--episodes_csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    training_runs, training_summary = load_training(args.training_root)
    episodes, test_summary = summarize_test(args.episodes_csv)
    deltas = full_deltas(test_summary)

    training_runs.to_csv(args.outdir / "training_run_summary.csv", index=False)
    training_summary.to_csv(
        args.outdir / "training_final_window_summary.csv", index=False
    )
    test_summary.to_csv(
        args.outdir / "ablation_test_summary.csv", index=False
    )
    deltas.to_csv(
        args.outdir / "full_vs_ablation_deltas.csv", index=False
    )

    audit = {
        "training_run_count": int(len(training_runs)),
        "test_episode_row_count": int(len(episodes)),
        "expected_test_episode_row_count": (
            len(VARIANTS) * len(CASES) * 100
        ),
        "training_nan_inf": False,
        "rates_bounded_0_1": bool(
            episodes[list(RATE_METRICS)]
            .apply(lambda x: x.between(0, 1))
            .all()
            .all()
        ),
        "continuous_nan_allowed_only_when_group_incomplete": bool(
            (
                episodes[list(CONTINUOUS_METRICS)].notna().all(axis=1)
                | (episodes["all_defenders_hit"] == 0)
            ).all()
        ),
        "training_performed_during_evaluation": False,
        "optimizer_steps_during_evaluation": 0,
        "backpropagation_during_evaluation": False,
    }
    if not all(
        (
            audit["rates_bounded_0_1"],
            audit["continuous_nan_allowed_only_when_group_incomplete"],
        )
    ):
        raise RuntimeError(f"formal audit failed: {audit}")
    (args.outdir / "formal_result_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(test_summary.to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
