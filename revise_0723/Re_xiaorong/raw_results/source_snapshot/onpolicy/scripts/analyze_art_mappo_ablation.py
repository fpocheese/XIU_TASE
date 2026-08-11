#!/usr/bin/env python
"""Statistical analysis and IEEE-style plotting for the component ablation."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANTS = [
    "full",
    "no_trust",
    "no_gru",
    "no_attention_residual",
]
LABELS = {
    "full": "Full ART-MAPPO",
    "no_trust": "w/o trust-aware",
    "no_gru": "w/o GRU",
    "no_attention_residual": "w/o attn.-res.",
}
COLORS = {
    "full": "#0072B2",
    "no_trust": "#E69F00",
    "no_gru": "#009E73",
    "no_attention_residual": "#CC79A7",
}
MARKERS = {
    "full": "o",
    "no_trust": "s",
    "no_gru": "^",
    "no_attention_residual": "D",
}


def read_csv(path):
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), 0.0
    # Student-t 95% critical values for the sample sizes used here.
    table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
             6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    critical = table.get(values.size, 1.96)
    return (
        float(np.mean(values)),
        float(critical * np.std(values, ddof=1) / np.sqrt(values.size)),
    )


def exact_signflip_p(differences):
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return np.nan
    observed = abs(float(np.mean(differences)))
    if differences.size <= 16:
        permuted = []
        for signs in itertools.product((-1.0, 1.0), repeat=differences.size):
            permuted.append(abs(float(np.mean(differences * signs))))
        return float(np.mean(np.asarray(permuted) >= observed - 1e-15))
    rng = np.random.RandomState(240724)
    signs = rng.choice(
        [-1.0, 1.0], size=(100_000, differences.size)
    )
    return float(
        np.mean(np.abs(np.mean(signs * differences, axis=1)) >= observed)
    )


def holm_adjust(p_values):
    valid = [(idx, p) for idx, p in enumerate(p_values) if np.isfinite(p)]
    adjusted = [np.nan] * len(p_values)
    running = 0.0
    for rank, (idx, p) in enumerate(sorted(valid, key=lambda item: item[1])):
        value = min(1.0, (len(valid) - rank) * p)
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def discover_runs(training_root):
    runs = {}
    for variant in VARIANTS:
        for case_dir in sorted((training_root / variant).glob("case*")):
            for seed_dir in sorted(case_dir.glob("seed*")):
                path = seed_dir / "training_metrics.csv"
                if not path.exists():
                    continue
                seed = int(seed_dir.name.replace("seed", ""))
                rows = read_csv(path)
                steps = np.array(
                    [float(row["environment_steps"]) for row in rows]
                )
                returns = np.array(
                    [float(row["mean_episode_return"]) for row in rows]
                )
                runs[(variant, case_dir.name, seed)] = {
                    "steps": steps,
                    "returns": returns,
                }
    return runs


def load_evaluation(eval_root):
    data = {}
    for variant in VARIANTS:
        for case_dir in sorted((eval_root / variant).glob("case*")):
            for seed_dir in sorted(case_dir.glob("seed*")):
                seed = int(seed_dir.name.replace("seed", ""))
                path = (
                    seed_dir
                    / case_dir.name
                    / f"{case_dir.name}_episode_summary.csv"
                )
                if not path.exists():
                    continue
                rows = read_csv(path)
                metrics = {
                    "target_interception_rate": [],
                    "all_target_interception": [],
                    "target_sync_rate": [],
                    "all_target_sync": [],
                    "mean_sync_spread_s": [],
                    "mean_agent_return": [],
                    "team_return": [],
                }
                for row in rows:
                    target_num = max(float(row["target_num"]), 1.0)
                    metrics["target_interception_rate"].append(
                        float(row["target_hit_count"]) / target_num
                    )
                    metrics["all_target_interception"].append(
                        float(str(row["all_hit"]).lower() in {"true", "1"})
                    )
                    metrics["target_sync_rate"].append(
                        float(row["target_sync_count"]) / target_num
                    )
                    metrics["all_target_sync"].append(
                        float(str(row["all_sync"]).lower() in {"true", "1"})
                    )
                    spread = float(row["mean_sync_spread"])
                    metrics["mean_sync_spread_s"].append(spread)
                    metrics["mean_agent_return"].append(
                        float(row["mean_agent_return"])
                    )
                    metrics["team_return"].append(
                        float(row["team_return"])
                    )
                data[(variant, case_dir.name, seed)] = {
                    key: np.asarray(value, dtype=float)
                    for key, value in metrics.items()
                }
    return data


def training_seed_metrics(runs):
    output = {}
    for key, run in runs.items():
        steps = run["steps"]
        values = run["returns"]
        tail = max(1, int(math.ceil(0.2 * len(values))))
        span = max(float(steps[-1] - steps[0]), 1.0)
        output[key] = {
            "final_return": float(np.mean(values[-tail:])),
            "return_auc": float(np.trapz(values, steps) / span),
        }
    return output


def evaluation_seed_metrics(data):
    output = {}
    for key, metrics in data.items():
        output[key] = {}
        for metric, values in metrics.items():
            finite = values[np.isfinite(values)]
            output[key][metric] = (
                float(np.mean(finite)) if finite.size else np.nan
            )
    return output


def validate_inputs(
    runs,
    evaluations,
    expected_seed_count,
    expected_episodes,
    expected_case_steps,
):
    expected_runs = len(VARIANTS) * 2 * expected_seed_count
    errors = []
    if len(runs) != expected_runs:
        errors.append(
            f"training run count {len(runs)} != expected {expected_runs}"
        )
    if len(evaluations) != expected_runs:
        errors.append(
            "evaluation seed-case count "
            f"{len(evaluations)} != expected {expected_runs}"
        )
    expected_keys = set()
    for variant in VARIANTS:
        for case in ("case1", "case2"):
            seeds = sorted(
                seed
                for v, c, seed in set(runs) | set(evaluations)
                if v == variant and c == case
            )
            if len(seeds) != expected_seed_count:
                errors.append(
                    f"{variant}/{case} has {len(seeds)} seeds, "
                    f"expected {expected_seed_count}"
                )
            expected_keys.update((variant, case, seed) for seed in seeds)
    if set(runs) != set(evaluations):
        missing_eval = sorted(set(runs) - set(evaluations))
        missing_train = sorted(set(evaluations) - set(runs))
        errors.append(
            f"training/evaluation key mismatch: missing_eval={missing_eval}, "
            f"missing_train={missing_train}"
        )
    for key, run in sorted(runs.items()):
        steps = np.asarray(run["steps"], dtype=float)
        returns = np.asarray(run["returns"], dtype=float)
        if steps.size == 0 or returns.size == 0:
            errors.append(f"{key}: empty training curve")
            continue
        if not np.all(np.isfinite(steps)) or not np.all(np.isfinite(returns)):
            errors.append(f"{key}: non-finite training value")
        if np.any(np.diff(steps) <= 0):
            errors.append(f"{key}: training steps are not strictly increasing")
        expected_last = float(expected_case_steps[key[1]])
        if not np.isclose(steps[-1], expected_last):
            errors.append(
                f"{key}: final step {steps[-1]} != {expected_last}"
            )
    core_eval_metrics = (
        "target_interception_rate",
        "all_target_interception",
        "target_sync_rate",
        "all_target_sync",
        "mean_agent_return",
        "team_return",
    )
    for key, metrics in sorted(evaluations.items()):
        lengths = {name: len(values) for name, values in metrics.items()}
        if set(lengths.values()) != {expected_episodes}:
            errors.append(
                f"{key}: evaluation lengths {lengths}, "
                f"expected {expected_episodes}"
            )
        for name in core_eval_metrics:
            values = np.asarray(metrics[name], dtype=float)
            if not np.all(np.isfinite(values)):
                errors.append(f"{key}/{name}: non-finite evaluation value")
    report = {
        "passed": not errors,
        "training_runs": len(runs),
        "evaluation_seed_case_runs": len(evaluations),
        "expected_runs": expected_runs,
        "episodes_per_seed_case": expected_episodes,
        "errors": errors,
    }
    if errors:
        raise RuntimeError(
            "formal ablation input validation failed:\n- "
            + "\n- ".join(errors)
        )
    return report


def paired_rows(train_metrics, eval_metrics):
    combined = {}
    for key, values in train_metrics.items():
        combined.setdefault(key, {}).update(values)
    for key, values in eval_metrics.items():
        combined.setdefault(key, {}).update(values)
    metrics = [
        "final_return",
        "return_auc",
        "target_interception_rate",
        "all_target_interception",
        "target_sync_rate",
        "all_target_sync",
        "mean_sync_spread_s",
        "mean_agent_return",
        "team_return",
    ]
    rows = []
    for scope in ["pooled", "case1", "case2"]:
        for metric in metrics:
            raw_rows = []
            for variant in VARIANTS[1:]:
                differences = []
                for (v, case, seed), full_values in combined.items():
                    if v != "full" or metric not in full_values:
                        continue
                    if scope != "pooled" and case != scope:
                        continue
                    other = combined.get((variant, case, seed), {})
                    if metric not in other:
                        continue
                    # Positive means the full model is better. Lower spread is
                    # better, so reverse that metric.
                    if metric == "mean_sync_spread_s":
                        diff = other[metric] - full_values[metric]
                    else:
                        diff = full_values[metric] - other[metric]
                    differences.append(diff)
                differences = np.asarray(differences, dtype=float)
                mean_diff, ci = mean_ci(differences)
                std = (
                    float(np.std(differences, ddof=1))
                    if np.isfinite(differences).sum() > 1
                    else np.nan
                )
                raw_rows.append(
                    {
                        "scope": scope,
                        "metric": metric,
                        "comparison": f"full_vs_{variant}",
                        "matched_seed_case_pairs": int(
                            np.isfinite(differences).sum()
                        ),
                        "mean_full_minus_ablation": mean_diff,
                        "ci95_half_width": ci,
                        "paired_effect_size_dz": (
                            mean_diff / std
                            if np.isfinite(std) and std > 0.0
                            else np.nan
                        ),
                        "signflip_p_raw": exact_signflip_p(differences),
                    }
                )
            adjusted = holm_adjust(
                [row["signflip_p_raw"] for row in raw_rows]
            )
            for row, p_adj in zip(raw_rows, adjusted):
                row["signflip_p_holm"] = p_adj
                rows.append(row)
    return rows


def aggregate_rows(train_metrics, eval_metrics):
    combined = {}
    for key, values in train_metrics.items():
        combined.setdefault(key, {}).update(values)
    for key, values in eval_metrics.items():
        combined.setdefault(key, {}).update(values)
    metric_names = sorted(
        {metric for values in combined.values() for metric in values}
    )
    rows = []
    for case in ["case1", "case2"]:
        for variant in VARIANTS:
            for metric in metric_names:
                values = [
                    metrics[metric]
                    for (v, c, _), metrics in combined.items()
                    if v == variant and c == case and metric in metrics
                ]
                mean, ci = mean_ci(values)
                rows.append(
                    {
                        "variant": variant,
                        "case": case,
                        "metric": metric,
                        "independent_seeds": int(
                            np.isfinite(np.asarray(values, dtype=float)).sum()
                        ),
                        "mean": mean,
                        "ci95_half_width": ci,
                    }
                )
    return rows


def plot_figure(runs, eval_metrics, output_dir):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Tinos",
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.85,
            "lines.linewidth": 1.35,
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.25))
    for ax, case, letter in zip(
        axes[0], ["case1", "case2"], ["(a)", "(b)"]
    ):
        for variant in VARIANTS:
            series = [
                run
                for (v, c, _), run in runs.items()
                if v == variant and c == case
            ]
            if not series:
                continue
            xmax = min(run["steps"][-1] for run in series)
            grid = np.linspace(0.0, xmax, 160)
            values = np.vstack(
                [
                    np.interp(grid, run["steps"], run["returns"])
                    for run in series
                ]
            )
            mean = np.mean(values, axis=0) / 1e6
            if values.shape[0] > 1:
                critical = 2.776 if values.shape[0] == 5 else 1.96
                ci = (
                    critical
                    * np.std(values / 1e6, axis=0, ddof=1)
                    / np.sqrt(values.shape[0])
                )
            else:
                ci = np.zeros_like(mean)
            ax.plot(
                grid / 1e3,
                mean,
                color=COLORS[variant],
                label=LABELS[variant],
                marker=MARKERS[variant],
                markevery=24,
                markersize=3.0,
                markerfacecolor="white",
                markeredgewidth=0.7,
                linewidth=1.65 if variant == "full" else 1.25,
            )
            ax.fill_between(
                grid / 1e3,
                mean - ci,
                mean + ci,
                color=COLORS[variant],
                alpha=0.14,
                linewidth=0,
            )
        ax.set_xlabel("Environment steps ($\\times 10^3$)")
        ax.set_ylabel("Episode return ($\\times 10^6$)")
        ax.text(
            0.01,
            0.98,
            letter,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        ax.grid(True, color="#D9D9D9", linewidth=0.45, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].legend(
        loc="best", frameon=False, ncol=2, columnspacing=0.8
    )

    x = np.arange(len(VARIANTS))
    width = 0.34
    for ax, metric, ylabel, letter in [
        (
            axes[1, 0],
            "target_interception_rate",
            "Per-target interception rate",
            "(c)",
        ),
        (
            axes[1, 1],
            "target_sync_rate",
            "Per-target synchronized rate",
            "(d)",
        ),
    ]:
        for offset, case, hatch in [
            (-width / 2, "case1", ""),
            (width / 2, "case2", "//"),
        ]:
            means, errors = [], []
            for variant in VARIANTS:
                values = [
                    metrics[metric]
                    for (v, c, _), metrics in eval_metrics.items()
                    if v == variant and c == case
                ]
                mean, ci = mean_ci(values)
                means.append(mean)
                errors.append(ci)
            bars = ax.bar(
                x + offset,
                means,
                width,
                yerr=errors,
                capsize=2.2,
                color=[COLORS[v] for v in VARIANTS],
                alpha=0.82,
                edgecolor="black",
                linewidth=0.45,
                hatch=hatch,
                label="Case 1" if case == "case1" else "Case 2",
            )
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["Full", "No trust", "No GRU", "No attn.-res."],
            rotation=12,
            ha="right",
        )
        ax.text(
            0.01,
            0.98,
            letter,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1, 0].legend(loc="best", frameon=False)
    fig.tight_layout(pad=0.7, w_pad=1.0, h_pad=0.9)
    for suffix, kwargs in [
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 600}),
    ]:
        fig.savefig(
            output_dir / f"art_mappo_component_ablation.{suffix}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_root", required=True)
    parser.add_argument("--evaluation_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_seed_count", type=int, default=5)
    parser.add_argument(
        "--expected_episodes_per_seed_case", type=int, default=20
    )
    parser.add_argument("--expected_case1_steps", type=int, default=600_000)
    parser.add_argument("--expected_case2_steps", type=int, default=1_800_000)
    args = parser.parse_args()
    training_root = Path(args.training_root)
    evaluation_root = Path(args.evaluation_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(training_root)
    evaluations = load_evaluation(evaluation_root)
    if not runs or not evaluations:
        raise RuntimeError("training or evaluation data are missing")
    validation = validate_inputs(
        runs,
        evaluations,
        expected_seed_count=args.expected_seed_count,
        expected_episodes=args.expected_episodes_per_seed_case,
        expected_case_steps={
            "case1": args.expected_case1_steps,
            "case2": args.expected_case2_steps,
        },
    )
    train_seed = training_seed_metrics(runs)
    eval_seed = evaluation_seed_metrics(evaluations)
    statistics = paired_rows(train_seed, eval_seed)
    aggregates = aggregate_rows(train_seed, eval_seed)
    plot_figure(runs, eval_seed, output_dir)
    write_csv(output_dir / "ablation_paired_statistics.csv", statistics)
    write_csv(output_dir / "ablation_aggregate_metrics.csv", aggregates)

    seed_rows = []
    keys = sorted(set(train_seed) | set(eval_seed))
    for variant, case, seed in keys:
        row = {"variant": variant, "case": case, "seed": seed}
        row.update(train_seed.get((variant, case, seed), {}))
        row.update(eval_seed.get((variant, case, seed), {}))
        seed_rows.append(row)
    write_csv(output_dir / "ablation_seed_level_metrics.csv", seed_rows)

    manifest = {
        "training_runs": len(runs),
        "evaluation_seed_case_runs": len(evaluations),
        "variants": VARIANTS,
        "statistics": (
            "paired by training seed and case; two-sided exact sign-flip "
            "tests with Holm correction within each metric"
        ),
        "confidence_intervals": (
            "95% Student-t intervals over independent seed-level estimates"
        ),
        "figure_evaluation_metrics": [
            "target_interception_rate",
            "target_sync_rate",
        ],
        "strict_trial_metrics_retained_in_csv": [
            "all_target_interception",
            "all_target_sync",
        ],
        "input_validation": validation,
    }
    with open(
        output_dir / "analysis_manifest.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
