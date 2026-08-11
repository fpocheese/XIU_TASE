#!/usr/bin/env python3
"""Build the compact Case-1 ablation table requested for the revised paper.

The script performs no training and never edits policy weights.  It checks and
combines three disjoint frozen-policy test blocks (100 + 500 + 400 trials) from
Case 1, yielding exactly 1000 trials per algorithm.  Training quantities are
reported directly from the Case-1 training trace.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


VARIANTS = ("full", "no_trust", "no_gru", "no_attention_residual")
CASES = ("case1",)
LABELS = {
    "full": "Full ART-MAPPO",
    "no_trust": "w/o trust-aware mechanism",
    "no_gru": "w/o GRU temporal encoder",
    "no_attention_residual": "w/o attention--residual backbone",
}
TERMINAL = ("E_n_g", "E_miss_m", "E_co_time_s", "E_t_s")


def wilson(successes: int, total: int, z: float = 1.959963984540054):
    p = successes / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / den
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4 * total**2)) / den
    return max(0.0, center - half), min(1.0, center + half)


def load_evaluations(evaluation_roots: list[Path], batch_counts: list[int]) -> pd.DataFrame:
    if len(evaluation_roots) != len(batch_counts):
        raise RuntimeError("evaluation roots and batch counts must have equal length")
    frames = []
    audit = []
    for batch_index, (evaluation_root, expected_count) in enumerate(
        zip(evaluation_roots, batch_counts)
    ):
        for variant in VARIANTS:
            case = "case1"
            path = evaluation_root / variant / case / "seed8303" / "episodes.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            data = pd.read_csv(path)
            if len(data) != expected_count or data["seed"].nunique() != expected_count:
                raise RuntimeError(
                    f"batch {batch_index}, {variant}/{case}: expected {expected_count} "
                    "rows and unique seeds; "
                    f"found {len(data)} rows and {data['seed'].nunique()} seeds"
                )
            if set(data["variant"].astype(str)) != {variant}:
                raise RuntimeError(f"variant mismatch in {path}")
            if set(data["case"].astype(str)) != {case}:
                raise RuntimeError(f"case mismatch in {path}")
            required = ["target_coverage_success", *TERMINAL]
            numeric = data[required].apply(pd.to_numeric, errors="coerce")
            if not np.isfinite(numeric.to_numpy()).all():
                raise RuntimeError(f"NaN/Inf in required columns: {path}")
            data = data.copy()
            data["evaluation_batch"] = batch_index
            frames.append(data)
            audit.append(
                {
                    "evaluation_batch": batch_index,
                    "source_path": str(path),
                    "variant": variant,
                    "case": case,
                    "episode_count": len(data),
                    "unique_seed_count": data["seed"].nunique(),
                    "seed_min": int(data["seed"].min()),
                    "seed_max": int(data["seed"].max()),
                }
            )
    combined = pd.concat(frames, ignore_index=True)
    reference = set(combined[combined.variant == "full"].seed)
    if len(reference) != 1000:
        raise RuntimeError(f"expected 1000 disjoint Case-1 seeds, found {len(reference)}")
    for variant in VARIANTS[1:]:
        compared = set(combined[combined.variant == variant].seed)
        if compared != reference:
            raise RuntimeError(f"paired Case-1 seed sets differ for {variant}")
    combined.attrs["audit"] = audit
    return combined


def build_summary(training: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    required_training = {
        "case",
        "variant",
        "return_auc_time_normalized",
        "final_window_return_std",
    }
    missing = required_training - set(training.columns)
    if missing:
        raise RuntimeError(f"training table is missing: {sorted(missing)}")
    rows = []
    for variant in VARIANTS:
        train = training[(training.variant == variant) & (training.case == "case1")]
        if set(train.case) != {"case1"} or len(train) != 1:
            raise RuntimeError(f"expected one Case-1 training row for {variant}")
        test = episodes[episodes.variant == variant]
        if len(test) != 1000:
            raise RuntimeError(f"expected 1000 pooled trials for {variant}")
        successes = int(test["target_coverage_success"].sum())
        low, high = wilson(successes, 1000)
        row = {
            "variant": variant,
            "algorithm": LABELS[variant].replace("--", "–"),
            "training_auc_1e3": float(train["return_auc_time_normalized"].iloc[0] / 1e3),
            "final_return_std_1e3": float(train["final_window_return_std"].iloc[0] / 1e3),
            "interception_success_count": successes,
            "interception_success_rate_percent": 100.0 * successes / 1000.0,
            "interception_success_ci95_low_percent": 100.0 * low,
            "interception_success_ci95_high_percent": 100.0 * high,
            "test_episode_count": 1000,
        }
        successful_test = test[test["target_coverage_success"] == 1]
        row["terminal_metric_eligible_count"] = int(len(successful_test))
        for metric in TERMINAL:
            values = successful_test[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_terminal_effects(episodes: pd.DataFrame, bootstrap_samples: int = 20000):
    """Return pooled paired full-minus-ablation effects and bootstrap CIs."""
    rows = []
    for ablation_index, variant in enumerate(VARIANTS[1:]):
        full = episodes[episodes.variant == "full"].set_index(["case", "seed"])
        ablated = episodes[episodes.variant == variant].set_index(["case", "seed"])
        full = full.sort_index()
        ablated = ablated.sort_index()
        if not full.index.equals(ablated.index):
            raise RuntimeError(f"pooled paired keys differ for {variant}")
        eligible = (
            (full["target_coverage_success"].to_numpy(int) == 1)
            & (ablated["target_coverage_success"].to_numpy(int) == 1)
        )
        for metric_index, metric in enumerate(TERMINAL):
            differences = (
                full[metric].to_numpy(float)[eligible]
                - ablated[metric].to_numpy(float)[eligible]
            )
            rng = np.random.default_rng(74000 + 100 * ablation_index + metric_index)
            estimates = np.empty(bootstrap_samples, dtype=float)
            offset = 0
            while offset < bootstrap_samples:
                count = min(500, bootstrap_samples - offset)
                idx = rng.integers(0, len(differences), size=(count, len(differences)))
                estimates[offset : offset + count] = differences[idx].mean(axis=1)
                offset += count
            rows.append(
                {
                    "ablation": variant,
                    "metric": metric,
                    "paired_eligible_count": int(len(differences)),
                    "full_mean": float(full[metric].to_numpy(float)[eligible].mean()),
                    "ablation_mean": float(
                        ablated[metric].to_numpy(float)[eligible].mean()
                    ),
                    "full_minus_ablation": float(differences.mean()),
                    "paired_bootstrap_ci95_low": float(
                        np.quantile(estimates, 0.025)
                    ),
                    "paired_bootstrap_ci95_high": float(
                        np.quantile(estimates, 0.975)
                    ),
                    "bootstrap_samples": bootstrap_samples,
                }
            )
    return pd.DataFrame(rows)


def write_latex(summary: pd.DataFrame, path: Path):
    row_by_variant = summary.set_index("variant")
    lines = [
        r"\begin{table*}[!b]",
        r"\centering",
        r"\scriptsize",
        r"\color{blue}",
        r"\caption{Training and 1000-trial Case~1 test results of the ART-MAPPO component ablation.}",
        r"\label{tab:compact_ablation_v3}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Variant & Training AUC ($10^3$) $\uparrow$ & Std. final return ($10^3$) $\downarrow$ & ISR (\%) $\uparrow$ & $E_n$ (g) $\downarrow$ & $E_{miss}$ (m) $\downarrow$ & $E_{co\text{-}time}$ (s) $\downarrow$ & $E_t$ (s) $\downarrow$ \\",
        r"\midrule",
    ]
    for variant in VARIANTS:
        row = row_by_variant.loc[variant]
        label = LABELS[variant]
        lines.append(
            f"{label} & {row.training_auc_1e3:.2f} & "
            f"{row.final_return_std_1e3:.2f} & "
            f"{row.interception_success_rate_percent:.1f} & "
            f"{row.E_n_g_mean:.4f} & {row.E_miss_m_mean:.3f} & "
            f"{row.E_co_time_s_mean:.4f} & {row.E_t_s_mean:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{1mm}\parbox{0.99\linewidth}{\footnotesize Each algorithm is evaluated in exactly 1000 paired, frozen-policy Monte Carlo trials in Case~1. ISR denotes complete target interception. Consistent with Section~\ref{subsec:montecarlo}, the four terminal entries are sample means over successful interception trials only; no missing value is imputed. Training AUC is the horizon-normalized area under the raw episodic-return curve. Final-return variability is the sample standard deviation over the last 10\% (58) of training updates. One training seed is used per variant; hence, the two training columns are descriptive, whereas the 1000 test trials quantify uncertainty due to environmental initial-state perturbations.}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--batch-counts", type=int, nargs="+", required=True)
    parser.add_argument("--training-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_evaluations(args.evaluation_roots, args.batch_counts)
    training = pd.read_csv(args.training_csv)
    summary = build_summary(training, episodes)
    paired = paired_terminal_effects(episodes)
    episodes.to_csv(args.output_dir / "case1_ablation_episodes_n1000.csv", index=False)
    summary.to_csv(args.output_dir / "paper_compact_ablation_n1000.csv", index=False)
    paired.to_csv(args.output_dir / "case1_paired_terminal_effects_n1000.csv", index=False)
    pd.DataFrame(episodes.attrs["audit"]).to_csv(
        args.output_dir / "case1_ablation_n1000_audit.csv", index=False
    )
    write_latex(summary, args.output_dir / "paper_compact_ablation_n1000.tex")
    manifest = {
        "training_updates_per_model": 585,
        "test_trials_per_algorithm": 1000,
        "test_scenario": "case1",
        "test_batch_counts_per_algorithm": args.batch_counts,
        "test_is_frozen_policy_inference": True,
        "training_or_optimizer_updates_during_test": False,
        "paired_seed_design": True,
        "required_values_all_finite": True,
    }
    (args.output_dir / "compact_ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
