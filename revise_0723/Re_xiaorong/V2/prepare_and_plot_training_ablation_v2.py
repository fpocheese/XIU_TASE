#!/usr/bin/env python3
"""Prepare and plot ART-MAPPO training-component ablations.

The script adapts the formal ablation CSV files to the per-seed NPY and
four-column plotted-CSV interface used by ``simple_converge_v7``.  It reuses
the reference figure's IEEE font and line conventions, but deliberately does
not reproduce its method-specific ideal-curve blending, synthetic saturation,
variance annealing, or injected random noise.  Every V2 curve is calculated
from the recorded values using the same trailing moving average for all four
variants and a 95% Student-t confidence interval across five training seeds.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


VARIANTS = [
    "full",
    "no_trust",
    "no_gru",
    "no_attention_residual",
]
CASES = ["case1", "case2"]
LABELS = {
    "full": "Full ART-MAPPO",
    "no_trust": "w/o trust-aware",
    "no_gru": "w/o GRU",
    "no_attention_residual": "w/o attn.-res.",
}
FILE_LABELS = {
    "full": "Full-ART-MAPPO",
    "no_trust": "No-Trust",
    "no_gru": "No-GRU",
    "no_attention_residual": "No-Attn-Residual",
}
COLORS = {
    "full": "#0072B2",
    "no_trust": "#E69F00",
    "no_gru": "#009E73",
    "no_attention_residual": "#CC79A7",
}
LINESTYLES = {
    "full": "-",
    "no_trust": "--",
    "no_gru": "-.",
    "no_attention_residual": ":",
}
MARKERS = {
    "full": "o",
    "no_trust": "s",
    "no_gru": "^",
    "no_attention_residual": "D",
}
EXPECTED_UPDATES = {"case1": 100, "case2": 300}
T95_N5 = 2.7764451051977987
RAW_FIELDS = {
    "reward": "mean_episode_return",
    "critic_loss": "value_loss",
    "entropy": "entropy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average with the same edge handling as the reference."""
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    csum = np.cumsum(values)
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        total = csum[idx] - (csum[start - 1] if start else 0.0)
        result[idx] = total / (idx - start + 1)
    return result


def mean_ci95(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if matrix.shape[0] != 5:
        raise ValueError("Formal V2 curves require exactly five seeds")
    mean = np.mean(matrix, axis=0)
    sd = np.std(matrix, axis=0, ddof=1)
    return mean, T95_N5 * sd / math.sqrt(5)


def exact_signflip_p(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if not len(differences) or np.allclose(differences, 0.0):
        return 1.0
    observed = abs(float(np.mean(differences)))
    count = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        statistic = abs(float(np.mean(differences * np.asarray(signs))))
        count += statistic >= observed - 1e-15
        total += 1
    return count / total


def holm_adjust(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    adjusted = [1.0] * len(values)
    running = 0.0
    m = len(values)
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * values[idx])
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted


def load_training(
    root: Path,
) -> tuple[
    dict[tuple[str, str, int], dict[str, np.ndarray]],
    dict,
]:
    data: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    errors: list[str] = []
    for variant in VARIANTS:
        for case in CASES:
            paths = sorted((root / variant / case).glob("seed*/training_metrics.csv"))
            if len(paths) != 5:
                errors.append(f"{variant}/{case}: expected 5 CSVs, found {len(paths)}")
            for path in paths:
                seed = int(path.parent.name.replace("seed", ""))
                rows = read_csv(path)
                if len(rows) != EXPECTED_UPDATES[case]:
                    errors.append(
                        f"{variant}/{case}/seed{seed}: "
                        f"expected {EXPECTED_UPDATES[case]} rows, found {len(rows)}"
                    )
                required = ["environment_steps", *RAW_FIELDS.values()]
                missing = [field for field in required if field not in rows[0]]
                if missing:
                    errors.append(f"{path}: missing {missing}")
                    continue
                arrays = {
                    "environment_steps": np.asarray(
                        [float(row["environment_steps"]) for row in rows]
                    )
                }
                for metric, field in RAW_FIELDS.items():
                    arrays[metric] = np.asarray(
                        [float(row[field]) for row in rows], dtype=float
                    )
                for name, values in arrays.items():
                    if not np.all(np.isfinite(values)):
                        errors.append(f"{path}: non-finite {name}")
                if np.any(np.diff(arrays["environment_steps"]) <= 0):
                    errors.append(f"{path}: non-increasing environment steps")
                expected_final = 600_000 if case == "case1" else 1_800_000
                if int(arrays["environment_steps"][-1]) != expected_final:
                    errors.append(
                        f"{path}: final steps {arrays['environment_steps'][-1]} "
                        f"!= {expected_final}"
                    )
                data[(variant, case, seed)] = arrays
    if len(data) != 40:
        errors.append(f"expected 40 training runs, found {len(data)}")
    if errors:
        raise RuntimeError("\n".join(errors))
    return data, {
        "passed": True,
        "training_runs": len(data),
        "seeds_per_variant_case": 5,
        "case1_updates": 100,
        "case2_updates": 300,
        "errors": [],
    }


def export_reference_npy(data: dict, out: Path) -> None:
    for (variant, case, seed), arrays in data.items():
        case_dir = out / "data" / "converted_npy" / case
        case_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{FILE_LABELS[variant]}_seed{seed}"
        np.save(case_dir / f"{prefix}_steps.npy", arrays["environment_steps"])
        np.save(case_dir / f"{prefix}_rewards.npy", arrays["reward"])
        np.save(case_dir / f"{prefix}_critic_loss.npy", arrays["critic_loss"])
        np.save(case_dir / f"{prefix}_entropy.npy", arrays["entropy"])


def compute_plot_curves(data: dict, out: Path) -> dict:
    curves = {}
    for case in CASES:
        for variant in VARIANTS:
            seeds = sorted(
                seed
                for v, c, seed in data
                if v == variant and c == case
            )
            steps = data[(variant, case, seeds[0])]["environment_steps"]
            for seed in seeds[1:]:
                if not np.array_equal(
                    steps, data[(variant, case, seed)]["environment_steps"]
                ):
                    raise RuntimeError(f"Step mismatch for {variant}/{case}")
            window = max(3, int(round(len(steps) * 0.05)))
            for metric in RAW_FIELDS:
                matrix = np.vstack(
                    [
                        moving_average(data[(variant, case, seed)][metric], window)
                        for seed in seeds
                    ]
                )
                mean, ci = mean_ci95(matrix)
                scale = 1e6 if metric == "reward" else 1.0
                plotted_mean = mean / scale
                plotted_ci = ci / scale
                lower = plotted_mean - plotted_ci
                if metric == "critic_loss":
                    lower = np.maximum(lower, 1e-8)
                rows = []
                for idx in range(len(steps)):
                    rows.append(
                        {
                            "environment_steps_k": f"{steps[idx] / 1000:.8f}",
                            "mean": f"{plotted_mean[idx]:.10g}",
                            "shadow_lower": f"{lower[idx]:.10g}",
                            "shadow_upper": f"{(plotted_mean[idx] + plotted_ci[idx]):.10g}",
                            "n_seeds": 5,
                            "moving_average_window_updates": window,
                        }
                    )
                write_csv(
                    out
                    / "data"
                    / "plot_csv"
                    / metric
                    / case
                    / f"{variant}.csv",
                    rows,
                    [
                        "environment_steps_k",
                        "mean",
                        "shadow_lower",
                        "shadow_upper",
                        "n_seeds",
                        "moving_average_window_updates",
                    ],
                )
                curves[(metric, case, variant)] = {
                    "x": steps / 1000,
                    "mean": plotted_mean,
                    "lower": lower,
                    "upper": plotted_mean + plotted_ci,
                    "window": window,
                }
    return curves


def per_seed_training_summary(data: dict) -> list[dict]:
    rows = []
    for (variant, case, seed), arrays in sorted(data.items()):
        n = len(arrays["reward"])
        tail20 = slice(int(0.8 * n), n)
        first10 = slice(0, max(1, int(0.1 * n)))
        tail10 = slice(int(0.9 * n), n)
        steps = arrays["environment_steps"]
        entropy_initial = float(np.mean(arrays["entropy"][first10]))
        entropy_tail = float(np.mean(arrays["entropy"][tail20]))
        rows.append(
            {
                "variant": variant,
                "case": case,
                "seed": seed,
                "updates": n,
                "final_environment_steps": int(steps[-1]),
                "tail20_reward": float(np.mean(arrays["reward"][tail20])),
                "reward_auc_per_step": float(
                    np.trapz(arrays["reward"], steps) / (steps[-1] - steps[0])
                ),
                "tail20_critic_loss": float(
                    np.mean(arrays["critic_loss"][tail20])
                ),
                "tail10_critic_loss": float(
                    np.mean(arrays["critic_loss"][tail10])
                ),
                "initial10_entropy": entropy_initial,
                "tail20_entropy": entropy_tail,
                "entropy_reduction": entropy_initial - entropy_tail,
                "entropy_auc_per_step": float(
                    np.trapz(arrays["entropy"], steps) / (steps[-1] - steps[0])
                ),
            }
        )
    return rows


def aggregate_training_summary(seed_rows: list[dict]) -> list[dict]:
    metrics = [
        "tail20_reward",
        "reward_auc_per_step",
        "tail20_critic_loss",
        "tail10_critic_loss",
        "initial10_entropy",
        "tail20_entropy",
        "entropy_reduction",
        "entropy_auc_per_step",
    ]
    rows = []
    for case in CASES:
        for variant in VARIANTS:
            selected = [
                row
                for row in seed_rows
                if row["case"] == case and row["variant"] == variant
            ]
            for metric in metrics:
                values = np.asarray([float(row[metric]) for row in selected])
                mean = float(np.mean(values))
                sd = float(np.std(values, ddof=1))
                rows.append(
                    {
                        "case": case,
                        "variant": variant,
                        "variant_label": LABELS[variant],
                        "metric": metric,
                        "n_seeds": 5,
                        "mean": mean,
                        "sample_sd": sd,
                        "ci95_half_width": T95_N5 * sd / math.sqrt(5),
                    }
                )
    return rows


def paired_training_stats(seed_rows: list[dict]) -> list[dict]:
    index = {
        (row["variant"], row["case"], int(row["seed"])): row
        for row in seed_rows
    }
    metrics = [
        ("tail20_reward", "higher"),
        ("reward_auc_per_step", "higher"),
        ("tail20_critic_loss", "lower"),
        ("tail20_entropy", "descriptive"),
        ("entropy_reduction", "higher"),
    ]
    all_rows = []
    for scope in ["pooled", *CASES]:
        for metric, direction in metrics:
            metric_rows = []
            for variant in VARIANTS[1:]:
                differences = []
                for case in CASES:
                    if scope != "pooled" and case != scope:
                        continue
                    for seed in range(701, 706):
                        full = float(index[("full", case, seed)][metric])
                        other = float(index[(variant, case, seed)][metric])
                        if direction == "lower":
                            differences.append(other - full)
                        else:
                            differences.append(full - other)
                differences = np.asarray(differences)
                mean = float(np.mean(differences))
                sd = float(np.std(differences, ddof=1))
                metric_rows.append(
                    {
                        "scope": scope,
                        "metric": metric,
                        "direction": direction,
                        "comparison": f"full_vs_{variant}",
                        "matched_pairs": len(differences),
                        "mean_improvement_full": mean,
                        "ci95_half_width": (
                            (2.262157 if len(differences) == 10 else T95_N5)
                            * sd
                            / math.sqrt(len(differences))
                        ),
                        "paired_effect_size_dz": mean / sd if sd > 0 else math.nan,
                        "signflip_p_raw": exact_signflip_p(differences),
                    }
                )
            adjusted = holm_adjust(
                [float(row["signflip_p_raw"]) for row in metric_rows]
            )
            for row, value in zip(metric_rows, adjusted):
                row["signflip_p_holm"] = value
                all_rows.append(row)
    return all_rows


def load_monte_carlo_summary(eval_root: Path) -> tuple[list[dict], list[dict]]:
    seed_rows = []
    for variant in VARIANTS:
        for case in CASES:
            paths = sorted((eval_root / variant / case).glob("seed*/eval_summary.csv"))
            if len(paths) != 5:
                raise RuntimeError(
                    f"{variant}/{case}: expected 5 eval summaries, found {len(paths)}"
                )
            for path in paths:
                seed = int(path.parent.name.replace("seed", ""))
                rows = read_csv(path)
                if len(rows) != 1 or int(rows[0]["episodes"]) != 20:
                    raise RuntimeError(f"Unexpected eval summary: {path}")
                row = rows[0]
                seed_rows.append(
                    {
                        "variant": variant,
                        "case": case,
                        "seed": seed,
                        "episodes": 20,
                        "interception_success_rate": float(row["all_hit_rate"]),
                        "coordination_success_rate": float(row["all_sync_rate"]),
                        "per_target_interception_rate": float(
                            row["target_success_rate"]
                        ),
                        "per_target_coordination_rate": float(
                            row["sync_success_rate"]
                        ),
                    }
                )
    aggregate = []
    for case in CASES:
        for variant in VARIANTS:
            selected = [
                row
                for row in seed_rows
                if row["case"] == case and row["variant"] == variant
            ]
            for metric in [
                "interception_success_rate",
                "coordination_success_rate",
                "per_target_interception_rate",
                "per_target_coordination_rate",
            ]:
                values = np.asarray([float(row[metric]) for row in selected])
                sd = float(np.std(values, ddof=1))
                aggregate.append(
                    {
                        "case": case,
                        "variant": variant,
                        "variant_label": LABELS[variant],
                        "metric": metric,
                        "monte_carlo_trials": 100,
                        "independent_training_seeds": 5,
                        "episodes_per_seed": 20,
                        "mean_rate": float(np.mean(values)),
                        "seed_level_sd": sd,
                        "ci95_half_width": T95_N5 * sd / math.sqrt(5),
                    }
                )
    return seed_rows, aggregate


def configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax, case: str, panel: str, log_y: bool = False) -> None:
    from matplotlib.ticker import AutoMinorLocator

    ax.set_title(f"{panel} {'Case 1' if case == 'case1' else 'Case 2'}", loc="left")
    ax.grid(True, which="major", color="#d9d9d9", alpha=0.75, linewidth=0.55)
    ax.tick_params(direction="in", which="both", top=False, right=False)
    if log_y:
        ax.set_yscale("log")
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_one_metric(curves: dict, metric: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    ylabels = {
        "reward": r"Episode return ($\times 10^6$)",
        "critic_loss": "Critic loss",
        "entropy": "Policy entropy",
    }
    names = {
        "reward": "ablation_training_reward",
        "critic_loss": "ablation_critic_loss",
        "entropy": "ablation_policy_entropy",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9))
    handles = []
    labels = []
    for col, case in enumerate(CASES):
        ax = axes[col]
        for variant in VARIANTS:
            curve = curves[(metric, case, variant)]
            markevery = max(1, len(curve["x"]) // 6)
            (line,) = ax.plot(
                curve["x"],
                curve["mean"],
                color=COLORS[variant],
                linestyle=LINESTYLES[variant],
                linewidth=1.8 if variant == "full" else 1.35,
                marker=MARKERS[variant],
                markersize=3.8,
                markerfacecolor="white",
                markeredgewidth=0.8,
                markevery=markevery,
                label=LABELS[variant],
                zorder=5 if variant == "full" else 3,
            )
            ax.fill_between(
                curve["x"],
                curve["lower"],
                curve["upper"],
                color=COLORS[variant],
                alpha=0.13 if variant == "full" else 0.09,
                linewidth=0,
                zorder=1,
            )
            if col == 0:
                handles.append(line)
                labels.append(LABELS[variant])
        ax.set_xlabel(r"Environment steps ($\times 10^3$)")
        if col == 0:
            ax.set_ylabel(ylabels[metric])
        style_axis(
            ax,
            case,
            f"({'ab'[col]})",
            log_y=(metric == "critic_loss"),
        )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        frameon=False,
        columnspacing=1.1,
        handlelength=2.2,
    )
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.18, top=0.80, wspace=0.24)
    stem = out / "figures" / names[metric]
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def plot_combined(curves: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = ["reward", "critic_loss", "entropy"]
    ylabels = {
        "reward": r"Episode return ($\times 10^6$)",
        "critic_loss": "Critic loss",
        "entropy": "Policy entropy",
    }
    fig, axes = plt.subplots(3, 2, figsize=(7.16, 7.6))
    handles = []
    labels = []
    panel = 0
    for row, metric in enumerate(metrics):
        for col, case in enumerate(CASES):
            ax = axes[row, col]
            for variant in VARIANTS:
                curve = curves[(metric, case, variant)]
                markevery = max(1, len(curve["x"]) // 6)
                (line,) = ax.plot(
                    curve["x"],
                    curve["mean"],
                    color=COLORS[variant],
                    linestyle=LINESTYLES[variant],
                    linewidth=1.75 if variant == "full" else 1.3,
                    marker=MARKERS[variant],
                    markersize=3.5,
                    markerfacecolor="white",
                    markeredgewidth=0.75,
                    markevery=markevery,
                    label=LABELS[variant],
                )
                ax.fill_between(
                    curve["x"],
                    curve["lower"],
                    curve["upper"],
                    color=COLORS[variant],
                    alpha=0.12 if variant == "full" else 0.08,
                    linewidth=0,
                )
                if row == 0 and col == 0:
                    handles.append(line)
                    labels.append(LABELS[variant])
            if col == 0:
                ax.set_ylabel(ylabels[metric])
            if row == 2:
                ax.set_xlabel(r"Environment steps ($\times 10^3$)")
            style_axis(
                ax,
                case,
                f"({chr(ord('a') + panel)})",
                log_y=(metric == "critic_loss"),
            )
            panel += 1
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        columnspacing=1.05,
        handlelength=2.15,
    )
    fig.subplots_adjust(
        left=0.095, right=0.995, bottom=0.07, top=0.94, hspace=0.37, wspace=0.24
    )
    stem = out / "figures" / "ablation_training_metrics_combined"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    data, validation = load_training(args.training_root)
    export_reference_npy(data, args.output_root)
    curves = compute_plot_curves(data, args.output_root)

    seed_rows = per_seed_training_summary(data)
    aggregate_rows = aggregate_training_summary(seed_rows)
    paired_rows = paired_training_stats(seed_rows)
    mc_seed_rows, mc_aggregate_rows = load_monte_carlo_summary(
        args.evaluation_root
    )

    table_dir = args.output_root / "tables"
    write_csv(
        table_dir / "training_seed_level_metrics.csv",
        seed_rows,
        list(seed_rows[0]),
    )
    write_csv(
        table_dir / "training_aggregate_metrics.csv",
        aggregate_rows,
        list(aggregate_rows[0]),
    )
    write_csv(
        table_dir / "training_paired_statistics.csv",
        paired_rows,
        list(paired_rows[0]),
    )
    write_csv(
        table_dir / "monte_carlo_100_seed_level.csv",
        mc_seed_rows,
        list(mc_seed_rows[0]),
    )
    write_csv(
        table_dir / "monte_carlo_100_summary.csv",
        mc_aggregate_rows,
        list(mc_aggregate_rows[0]),
    )

    configure_matplotlib()
    for metric in RAW_FIELDS:
        plot_one_metric(curves, metric, args.output_root)
    plot_combined(curves, args.output_root)

    audit = {
        "training_input_validation": validation,
        "evaluation_runs": 40,
        "evaluation_episodes": 800,
        "monte_carlo_trials_per_variant_case": 100,
        "raw_fields": RAW_FIELDS,
        "reference_data_interface": {
            "per_seed_npy_suffixes": [
                "_rewards.npy",
                "_critic_loss.npy",
                "_entropy.npy",
            ],
            "plotted_csv_columns": [
                "environment_steps_k",
                "mean",
                "shadow_lower",
                "shadow_upper",
            ],
        },
        "processing": {
            "case1_moving_average_window_updates": 5,
            "case2_moving_average_window_updates": 15,
            "confidence_band": "95% Student-t CI across 5 independent seeds",
            "synthetic_curve_blending": False,
            "synthetic_noise": False,
            "manual_data_modification": False,
        },
        "figure_outputs": {
            "standalone_metrics": list(RAW_FIELDS),
            "formats": ["PDF", "SVG", "PNG"],
            "png_dpi": 600,
        },
    }
    with (args.output_root / "V2_AUDIT.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
