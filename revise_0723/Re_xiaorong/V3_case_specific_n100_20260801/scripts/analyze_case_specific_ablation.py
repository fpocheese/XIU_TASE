#!/usr/bin/env python3
"""Audit and summarize case-specific ART-MAPPO ablation results.

The script never trains or edits a model.  It reads the eight training CSVs
and the frozen-policy Monte-Carlo episode table, checks the declared design,
and exports reviewer-ready statistics with explicit eligible sample counts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


VARIANTS = ("full", "no_trust", "no_gru", "no_attention_residual")
CASES = ("case1", "case2")
RATE_METRICS = (
    "target_coverage_success",
    "all_defenders_hit",
    "cooperative_success",
    "mission_success",
)
TERMINAL_METRICS = ("E_co_time_s", "E_n_g", "E_miss_m", "E_t_s")


def wilson(count: int, total: int, z: float = 1.959963984540054):
    p = count / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / den
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4 * total**2)) / den
    return p, max(0.0, center - half), min(1.0, center + half)


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=1).mean().to_numpy()


def first_sustained(x: np.ndarray, y: np.ndarray, threshold: float, span: int):
    good = np.asarray(y >= threshold, dtype=bool)
    if len(good) < span:
        return math.nan
    for start in range(len(good) - span + 1):
        if good[start : start + span].all():
            return float(x[start])
    return math.nan


def training_record(path: Path) -> dict:
    variant, case, seed_text = path.parts[-4:-1]
    frame = pd.read_csv(path).sort_values("environment_steps")
    numeric = frame.select_dtypes(include=[np.number]).to_numpy()
    if len(frame) < 20 or not np.isfinite(numeric).all():
        raise RuntimeError(f"invalid training trace: {path}")
    x = frame["environment_steps"].to_numpy(float)
    returns = frame["mean_episode_return"].to_numpy(float)
    smooth_window = max(5, min(20, len(frame) // 20))
    smooth = trailing_mean(returns, smooth_window)
    edge = max(10, len(frame) // 10)
    initial = float(np.mean(smooth[:edge]))
    asymptote = float(np.mean(smooth[-edge:]))
    threshold = initial + 0.90 * (asymptote - initial)
    sustained_span = max(5, len(frame) // 100)
    # The criterion is directional: most runs improve upward, but applying an
    # unconditional >= comparison to a downward asymptote would falsely mark
    # the first update as converged.  Reflect both series and threshold for a
    # downward trajectory so the same sustained-attainment definition applies.
    direction = 1.0 if asymptote >= initial else -1.0
    convergence = first_sustained(
        x, direction * smooth, direction * threshold, sustained_span
    )
    # Time-normalized return AUC.  It has return units and is comparable
    # within each case because all four variants share the same step budget.
    auc = float(np.trapz(returns, x) / (x[-1] - x[0]))
    first_entropy = float(frame["entropy"].iloc[:edge].mean())
    last_entropy = float(frame["entropy"].iloc[-edge:].mean())
    tail_value = frame["value_loss"].iloc[-edge:].to_numpy(float)
    tail_return = frame["mean_episode_return"].iloc[-edge:].to_numpy(float)
    tail_smoothed_return = smooth[-edge:]
    manifest_path = path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    components = manifest["components"]
    parameter_count = manifest["parameter_count"]
    return {
        "variant": variant,
        "case": case,
        "training_seed": int(seed_text.replace("seed", "")),
        "trust_aware_enabled": bool(components["trust_aware"]),
        "gru_enabled": bool(components["gru"]),
        "attention_residual_enabled": bool(components["attention_residual"]),
        "actor_parameter_count": int(parameter_count["actor"]),
        "critic_parameter_count": int(parameter_count["critic"]),
        "updates": int(len(frame)),
        "final_environment_steps": int(x[-1]),
        "smoothing_window_updates": smooth_window,
        "convergence_definition": "first sustained 90% of own asymptotic improvement",
        "asymptotic_improvement_direction": (
            "increase" if direction > 0.0 else "decrease"
        ),
        "initial_smoothed_return": initial,
        "asymptotic_smoothed_return": asymptote,
        "convergence_environment_steps": convergence,
        "convergence_updates": convergence / 1024.0 if np.isfinite(convergence) else math.nan,
        "return_auc_time_normalized": auc,
        # The final-policy statistics use the last 10% of recorded updates
        # (58 of 585 for the formal runs).  The mean estimates terminal policy
        # quality, while the sample standard deviation quantifies residual
        # training fluctuation without applying visual smoothing.
        "final_window_updates": int(edge),
        "final_window_return": float(np.mean(tail_return)),
        "final_window_return_std": float(np.std(tail_return, ddof=1)),
        "final_window_smoothed_return_std": float(
            np.std(tail_smoothed_return, ddof=1)
        ),
        "final_window_return_cv_percent": float(
            100.0 * np.std(tail_return, ddof=1)
            / max(abs(np.mean(tail_return)), 1e-12)
        ),
        "final_window_target_coverage": float(frame["target_coverage_rate"].iloc[-edge:].mean()),
        "final_window_complete_group_rate": float(frame["complete_group_rate"].iloc[-edge:].mean()),
        "final_window_all_hit_rate": float(frame["all_hit_rate"].iloc[-edge:].mean()),
        "final_window_all_sync_rate": float(frame["all_sync_rate"].iloc[-edge:].mean()),
        "final_window_critic_loss": float(np.mean(tail_value)),
        "final_window_critic_loss_std": float(np.std(tail_value, ddof=1)),
        "initial_window_policy_entropy": first_entropy,
        "final_window_policy_entropy": last_entropy,
        "policy_entropy_change_percent": 100.0 * (last_entropy - first_entropy) / max(abs(first_entropy), 1e-12),
        "final_window_guided_action_fraction": float(frame["guided_action_fraction"].iloc[-edge:].mean()),
        "final_window_actor_update_fraction": float(frame["actor_update_fraction"].iloc[-edge:].mean()),
        "final_window_trust_mean": float(frame["trust_mean"].iloc[-edge:].mean()),
        "wall_time_s": float(frame["wall_time_s"].iloc[-1]),
    }


def load_training(root: Path) -> pd.DataFrame:
    rows = [training_record(path) for path in sorted(root.glob("*/*/seed*/training_metrics.csv"))]
    data = pd.DataFrame(rows)
    expected = {(v, c) for v in VARIANTS for c in CASES}
    found = set(zip(data["variant"], data["case"]))
    if found != expected or len(data) != 8:
        raise RuntimeError(f"expected exactly eight case-specific traces; found {len(data)}: {found}")
    for case in CASES:
        full_row = data[(data.variant == "full") & (data.case == case)].iloc[0]
        full_auc = float(full_row["return_auc_time_normalized"])
        full_final_return = float(full_row["final_window_return"])
        full_final_std = float(full_row["final_window_return_std"])
        full_final_smoothed_std = float(
            full_row["final_window_smoothed_return_std"]
        )
        mask = data.case == case
        data.loc[mask, "return_auc_relative_to_full"] = data.loc[mask, "return_auc_time_normalized"] / full_auc
        data.loc[mask, "final_return_relative_to_full"] = (
            data.loc[mask, "final_window_return"] / full_final_return
        )
        data.loc[mask, "final_return_std_relative_to_full"] = (
            data.loc[mask, "final_window_return_std"] / full_final_std
        )
        data.loc[mask, "smoothed_final_return_std_relative_to_full"] = (
            data.loc[mask, "final_window_smoothed_return_std"]
            / full_final_smoothed_std
        )
    return data


def summarize_episodes(path: Path):
    episodes = pd.read_csv(path)
    rows = []
    for case in CASES:
        for variant in VARIANTS:
            data = episodes[(episodes.case == case) & (episodes.variant == variant)]
            if len(data) != 100 or data["seed"].nunique() != 100:
                raise RuntimeError(f"{variant}/{case}: expected 100 unique test episodes, got {len(data)}")
            row = {"variant": variant, "case": case, "episode_count": 100, "unique_test_seeds": 100}
            for metric in RATE_METRICS:
                count = int(data[metric].sum())
                p, low, high = wilson(count, 100)
                row.update({f"{metric}_count": count, f"{metric}_rate": p,
                            f"{metric}_ci95_low": low, f"{metric}_ci95_high": high})
            # The paper's boxplots use terminal metrics from episodes with a
            # complete interception event.  Finite values are retained and
            # the exact eligible count is exported; no imputation is allowed.
            for metric in TERMINAL_METRICS:
                values = data[metric].to_numpy(float)
                values = values[np.isfinite(values)]
                row[f"{metric}_eligible_n"] = int(len(values))
                for name, func in (
                    ("mean", np.mean), ("median", np.median),
                    ("q25", lambda a: np.quantile(a, 0.25)),
                    ("q75", lambda a: np.quantile(a, 0.75)),
                ):
                    row[f"{metric}_{name}"] = float(func(values)) if len(values) else math.nan
                row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
            rows.append(row)
    return episodes, pd.DataFrame(rows)


def exact_mcnemar_p(full_wins: int, ablation_wins: int) -> float:
    discordant = int(full_wins + ablation_wins)
    if discordant == 0:
        return 1.0
    lower = min(full_wins, ablation_wins)
    tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2.0 ** discordant)
    return min(1.0, 2.0 * tail)


def bootstrap_mean_ci(values: np.ndarray, seed: int, samples: int = 20000):
    values = np.asarray(values, float)
    if not len(values):
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    # Chunking bounds peak memory while preserving a fixed reproducible RNG.
    chunk = 1000
    offset = 0
    while offset < samples:
        count = min(chunk, samples - offset)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        estimates[offset : offset + count] = values[indices].mean(axis=1)
        offset += count
    return float(values.mean()), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def paired_comparisons(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_index, case in enumerate(CASES):
        full = episodes[(episodes.case == case) & (episodes.variant == "full")].set_index("seed").sort_index()
        for ablation_index, variant in enumerate(VARIANTS[1:]):
            ablated = episodes[(episodes.case == case) & (episodes.variant == variant)].set_index("seed").sort_index()
            require_same = full.index.equals(ablated.index)
            if not require_same:
                raise RuntimeError(f"{case}/{variant}: paired seed sets differ")
            for metric_index, metric in enumerate(RATE_METRICS):
                f = full[metric].to_numpy(int)
                a = ablated[metric].to_numpy(int)
                diff = f.astype(float) - a.astype(float)
                mean, low, high = bootstrap_mean_ci(diff, 41000 + case_index * 1000 + ablation_index * 100 + metric_index)
                full_wins = int(np.sum((f == 1) & (a == 0)))
                ablation_wins = int(np.sum((f == 0) & (a == 1)))
                rows.append({
                    "case": case, "ablation": variant, "metric": metric,
                    "metric_type": "binary", "paired_n": len(diff),
                    "full_mean": float(f.mean()), "ablation_mean": float(a.mean()),
                    "full_minus_ablation": mean,
                    "difference_ci95_low": low, "difference_ci95_high": high,
                    "full_only_success_count": full_wins,
                    "ablation_only_success_count": ablation_wins,
                    "mcnemar_exact_p": exact_mcnemar_p(full_wins, ablation_wins),
                    "bootstrap_samples": 20000,
                })
            for metric_index, metric in enumerate(TERMINAL_METRICS):
                f = full[metric].to_numpy(float)
                a = ablated[metric].to_numpy(float)
                eligible = np.isfinite(f) & np.isfinite(a)
                diff = f[eligible] - a[eligible]
                mean, low, high = bootstrap_mean_ci(diff, 51000 + case_index * 1000 + ablation_index * 100 + metric_index)
                rows.append({
                    "case": case, "ablation": variant, "metric": metric,
                    "metric_type": "continuous", "paired_n": int(eligible.sum()),
                    "full_mean": float(np.mean(f[eligible])) if eligible.any() else math.nan,
                    "ablation_mean": float(np.mean(a[eligible])) if eligible.any() else math.nan,
                    "full_median": float(np.median(f[eligible])) if eligible.any() else math.nan,
                    "ablation_median": float(np.median(a[eligible])) if eligible.any() else math.nan,
                    "full_minus_ablation": mean,
                    "difference_ci95_low": low, "difference_ci95_high": high,
                    "full_only_success_count": math.nan,
                    "ablation_only_success_count": math.nan,
                    "mcnemar_exact_p": math.nan,
                    "bootstrap_samples": 20000,
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_root", type=Path, required=True)
    parser.add_argument("--episodes_csv", type=Path, required=True)
    parser.add_argument("--evaluation_manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    training = load_training(args.training_root)
    episodes, testing = summarize_episodes(args.episodes_csv)
    paired = paired_comparisons(episodes)
    manifest = json.loads(args.evaluation_manifest.read_text(encoding="utf-8"))
    audit = {
        "training_trace_count": int(len(training)),
        "expected_training_trace_count": 8,
        "test_episode_count": int(len(episodes)),
        "expected_test_episode_count": 800,
        "each_variant_case_has_100_unique_seeds": True,
        "paired_test_seeds_across_variants": bool(manifest["paired_test_seeds_across_variants"]),
        "selection_uses_test_seeds": bool(manifest["selection_uses_test_seeds"]),
        "training_performed_during_evaluation": bool(manifest["training_performed"]),
        "optimizer_steps_during_evaluation": int(manifest["optimizer_steps"]),
        "backpropagation_during_evaluation": bool(manifest["backpropagation_performed"]),
        "training_numeric_nan_inf": False,
        "episode_rates_bounded": bool(episodes[list(RATE_METRICS)].apply(lambda x: x.between(0, 1)).all().all()),
    }
    if len(episodes) != 800 or audit["selection_uses_test_seeds"] or audit["training_performed_during_evaluation"] or audit["optimizer_steps_during_evaluation"] or audit["backpropagation_during_evaluation"] or not audit["episode_rates_bounded"]:
        raise RuntimeError(f"audit failed: {audit}")

    training.to_csv(args.outdir / "training_effect_summary.csv", index=False)
    testing.to_csv(args.outdir / "monte_carlo_n100_summary.csv", index=False)
    paired.to_csv(args.outdir / "paired_test_comparisons.csv", index=False)
    (args.outdir / "result_integrity_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(training.to_string(index=False))
    print(testing.to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
