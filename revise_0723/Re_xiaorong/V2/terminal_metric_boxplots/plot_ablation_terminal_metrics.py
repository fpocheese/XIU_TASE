#!/usr/bin/env python3
"""Recompute manuscript terminal metrics from the ART-MAPPO ablation event logs.

The evaluator stored every first-hit event for all 800 held-out episodes.  Three
metrics are reconstructed exactly from those events:

  * E_co-time: mean absolute deviation of group arrival times;
  * E_miss: mean distance at first lethal-radius entry;
  * E_t: time from engagement start to the last group member's arrival.

The event log contains the resultant load only at arrival.  Consequently E_n
below is a terminal-sample estimator, not the terminal-window average in the
latest manuscript.  This distinction is deliberately retained in all output
column names and documentation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIANTS = ["full", "no_trust", "no_gru", "no_attention_residual"]
DISPLAY = {
    "full": "ART-MAPPO",
    "no_trust": "w/o Trust",
    "no_gru": "w/o GRU",
    "no_attention_residual": "w/o A-R",
}
COLORS = {
    "full": "#0072B2",
    "no_trust": "#D55E00",
    "no_gru": "#009E73",
    "no_attention_residual": "#CC79A7",
}
GROUPS = {
    20: {0, 8, 16},
    21: {1, 9, 17},
    22: {2, 10, 18},
    23: {3, 11, 19},
    24: {4, 12},
    25: {5, 13},
    26: {6, 14},
    27: {7, 15},
}
METRICS = [
    ("E_co_time_s", r"$E_{\mathrm{co\mathrm{-}time}}$ (s)"),
    ("E_n_terminal_sample_g", r"$E_n$ (terminal sample, g)"),
    ("E_miss_m", r"$E_{\mathrm{miss}}$ (m)"),
    ("E_t_s", r"$E_t$ (s)"),
]


def configure_style() -> None:
    """Match the requested figures_v9 IEEE-TASE plotting style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "mathtext.fontset": "stix",
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.3,
            "grid.linestyle": "--",
            "grid.color": "#CCCCCC",
            "grid.alpha": 0.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "text.usetex": False,
        }
    )


def to_float(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def discover_runs(root: Path) -> list[tuple[str, str, int, Path, Path]]:
    runs = []
    for variant in VARIANTS:
        for case in ("case1", "case2"):
            pattern = root / variant / case
            for seed_dir in sorted(pattern.glob("seed*")):
                seed = int(seed_dir.name.removeprefix("seed"))
                event_path = seed_dir / case / f"{case}_hit_events.csv"
                summary_path = seed_dir / case / f"{case}_episode_summary.csv"
                if not event_path.is_file() or not summary_path.is_file():
                    raise FileNotFoundError(f"Incomplete evaluation run: {seed_dir}")
                runs.append((variant, case, seed, event_path, summary_path))
    if len(runs) != 40:
        raise RuntimeError(f"Expected 40 evaluation runs, found {len(runs)}")
    return runs


def recompute(root: Path) -> tuple[list[dict], list[dict]]:
    group_rows: list[dict] = []
    episode_rows: list[dict] = []
    for variant, case, seed, event_path, summary_path in discover_runs(root):
        events_by_episode: dict[int, list[dict]] = defaultdict(list)
        for event in read_csv(event_path):
            events_by_episode[int(event["episode"])].append(event)
        summaries = {int(row["episode"]): row for row in read_csv(summary_path)}
        if len(summaries) != 20:
            raise RuntimeError(f"Expected 20 episodes in {summary_path}, found {len(summaries)}")

        for episode in sorted(summaries):
            by_target: dict[int, list[dict]] = defaultdict(list)
            for event in events_by_episode.get(episode, []):
                by_target[int(event["target_id"])].append(event)

            local_groups: list[dict] = []
            for target_id, assigned in GROUPS.items():
                target_events = by_target.get(target_id, [])
                arrived = {int(row["defender_id"]) for row in target_events}
                complete = arrived == assigned
                if not complete:
                    continue
                times = np.asarray([to_float(row["time"]) for row in target_events], dtype=float)
                loads = np.asarray([to_float(row["load_norm"]) for row in target_events], dtype=float)
                misses = np.asarray([to_float(row["dist_to_target"]) for row in target_events], dtype=float)
                if not (np.isfinite(times).all() and np.isfinite(loads).all() and np.isfinite(misses).all()):
                    raise RuntimeError(f"NaN/Inf in terminal events: {event_path}, episode {episode}")
                if len(times) != len(assigned):
                    raise RuntimeError(f"Duplicate event in {event_path}, episode {episode}, target {target_id}")
                mean_time = float(np.mean(times))
                row = {
                    "variant": variant,
                    "variant_label": DISPLAY[variant],
                    "case": case,
                    "seed": seed,
                    "episode": episode,
                    "target_id": target_id,
                    "assigned_member_count": len(assigned),
                    "arrival_time_mean_s": mean_time,
                    "arrival_time_min_s": float(np.min(times)),
                    "arrival_time_max_s": float(np.max(times)),
                    "E_co_time_s": float(np.mean(np.abs(times - mean_time))),
                    "E_n_terminal_sample_g": float(np.mean(loads)),
                    "E_miss_m": float(np.mean(misses)),
                    "E_t_s": float(np.max(times)),  # t0 = 0 in the evaluator.
                }
                local_groups.append(row)

            n_complete = len(local_groups)
            summary = summaries[episode]
            ep_row = {
                "variant": variant,
                "variant_label": DISPLAY[variant],
                "case": case,
                "seed": seed,
                "episode": episode,
                "hit_count": int(summary["hit_count"]),
                "target_hit_count": int(summary["target_hit_count"]),
                "target_sync_count": int(summary["target_sync_count"]),
                "evaluator_all_hit": summary["all_hit"],
                "evaluator_all_sync": summary["all_sync"],
                "complete_group_count": n_complete,
                "has_complete_group": int(n_complete > 0),
                "strict_all_8_groups_complete": int(n_complete == len(GROUPS)),
            }
            for key, _ in METRICS:
                ep_row[key] = (
                    float(np.mean([row[key] for row in local_groups]))
                    if local_groups
                    else float("nan")
                )
            for row in local_groups:
                row["strict_all_8_groups_complete"] = int(n_complete == len(GROUPS))
                group_rows.append(row)
            episode_rows.append(ep_row)
    return group_rows, episode_rows


def percentile_ci(values: np.ndarray, seed: int = 20260728) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


def summarize(episode_rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    strata = {
        "strict_all_8_groups_complete": lambda r: r["strict_all_8_groups_complete"] == 1,
        "complete_group_conditioned_episode": lambda r: r["has_complete_group"] == 1,
    }
    for case in ("case1", "case2"):
        for variant in VARIANTS:
            base = [r for r in episode_rows if r["case"] == case and r["variant"] == variant]
            for stratum, predicate in strata.items():
                selected = [r for r in base if predicate(r)]
                for metric, _ in METRICS:
                    values = np.asarray([r[metric] for r in selected], dtype=float)
                    values = values[np.isfinite(values)]
                    lo, hi = percentile_ci(values)
                    output.append(
                        {
                            "case": case,
                            "variant": variant,
                            "variant_label": DISPLAY[variant],
                            "analysis_stratum": stratum,
                            "metric": metric,
                            "N_episodes": len(values),
                            "mean": float(np.mean(values)) if len(values) else float("nan"),
                            "std": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                            "median": float(np.median(values)) if len(values) else float("nan"),
                            "q1": float(np.percentile(values, 25)) if len(values) else float("nan"),
                            "q3": float(np.percentile(values, 75)) if len(values) else float("nan"),
                            "mean_ci95_low": lo,
                            "mean_ci95_high": hi,
                        }
                    )
    return output


def paired_effects(episode_rows: list[dict]) -> list[dict]:
    """Paired full-minus-ablation differences for matching seed/episode IDs."""
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None
    lookup = {
        (r["variant"], r["case"], r["seed"], r["episode"]): r for r in episode_rows
    }
    rows: list[dict] = []
    for case in ("case1", "case2"):
        stratum = (
            "strict_all_8_groups_complete"
            if case == "case1"
            else "complete_group_conditioned_episode"
        )
        valid_key = (
            "strict_all_8_groups_complete"
            if case == "case1"
            else "has_complete_group"
        )
        for ablation in VARIANTS[1:]:
            for metric, _ in METRICS:
                diffs = []
                for seed in range(701, 706):
                    for episode in range(1, 21):
                        full = lookup[("full", case, seed, episode)]
                        other = lookup[(ablation, case, seed, episode)]
                        if full[valid_key] != 1 or other[valid_key] != 1:
                            continue
                        diffs.append(float(full[metric]) - float(other[metric]))
                arr = np.asarray(diffs, dtype=float)
                if len(arr):
                    lo, hi = percentile_ci(arr, seed=20260728 + len(rows))
                    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan")
                    dz = float(np.mean(arr) / sd) if sd > 0 else float("nan")
                    if wilcoxon is not None and np.any(arr != 0):
                        p_value = float(wilcoxon(arr, zero_method="wilcox").pvalue)
                    else:
                        p_value = float("nan")
                else:
                    lo = hi = sd = dz = p_value = float("nan")
                rows.append(
                    {
                        "case": case,
                        "analysis_stratum": stratum,
                        "comparison": f"full_minus_{ablation}",
                        "metric": metric,
                        "N_paired_episodes": len(arr),
                        "mean_paired_difference": float(np.mean(arr)) if len(arr) else float("nan"),
                        "difference_ci95_low": lo,
                        "difference_ci95_high": hi,
                        "paired_effect_dz": dz,
                        "wilcoxon_p": p_value,
                    }
                )
    return rows


def completion_summary(episode_rows: list[dict]) -> list[dict]:
    rows = []
    for case in ("case1", "case2"):
        for variant in VARIANTS:
            selected = [
                row
                for row in episode_rows
                if row["case"] == case and row["variant"] == variant
            ]
            n = len(selected)
            rows.append(
                {
                    "case": case,
                    "variant": variant,
                    "variant_label": DISPLAY[variant],
                    "N_episodes": n,
                    "interceptor_hit_rate": sum(row["hit_count"] for row in selected)
                    / (20 * n),
                    "target_hit_rate": sum(row["target_hit_count"] for row in selected)
                    / (8 * n),
                    "coordinated_target_rate": sum(
                        row["target_sync_count"] for row in selected
                    )
                    / (8 * n),
                    "complete_group_rate": sum(
                        row["complete_group_count"] for row in selected
                    )
                    / (8 * n),
                    "all_targets_hit_episode_rate": sum(
                        row["evaluator_all_hit"] == "True" for row in selected
                    )
                    / n,
                    "all_targets_coordinated_episode_rate": sum(
                        row["evaluator_all_sync"] == "True" for row in selected
                    )
                    / n,
                    "strict_all_8_groups_complete_episode_rate": sum(
                        row["strict_all_8_groups_complete"] for row in selected
                    )
                    / n,
                    "complete_group_conditioned_episode_count": sum(
                        row["has_complete_group"] for row in selected
                    ),
                    "complete_group_count": sum(
                        row["complete_group_count"] for row in selected
                    ),
                }
            )
    return rows


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, out: Path, stem: str) -> None:
    for ext in ("pdf", "svg"):
        fig.savefig(out / f"{stem}.{ext}", format=ext)
    fig.savefig(out / f"{stem}.png", dpi=600)


def plot_case(
    episode_rows: list[dict], case: str, stratum: str, out: Path, stem: str
) -> None:
    if stratum == "strict_all_8_groups_complete":
        predicate = lambda row: row["strict_all_8_groups_complete"] == 1
    else:
        predicate = lambda row: row["has_complete_group"] == 1
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.32))
    for ax, (metric, label) in zip(axes, METRICS):
        values = [
            np.asarray(
                [
                    row[metric]
                    for row in episode_rows
                    if row["case"] == case
                    and row["variant"] == variant
                    and predicate(row)
                    and np.isfinite(row[metric])
                ],
                dtype=float,
            )
            for variant in VARIANTS
        ]
        bp = ax.boxplot(
            values,
            labels=[DISPLAY[v] for v in VARIANTS],
            widths=0.56,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 3.2,
            },
            medianprops={"color": "black", "linewidth": 1.4},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
            flierprops={"marker": "o", "markersize": 2, "alpha": 0.28},
        )
        for box, variant in zip(bp["boxes"], VARIANTS):
            box.set_facecolor(COLORS[variant])
            box.set_alpha(0.68)
            box.set_linewidth(0.8)
        ax.set_ylabel(label, fontsize=9.5)
        ax.set_xticklabels(
            [DISPLAY[v] for v in VARIANTS], fontsize=7.2, rotation=18, ha="right"
        )
        ax.grid(True, axis="y", linestyle="--", linewidth=0.25, color="#CCCCCC", alpha=0.55)
        despine(ax)
        ax.text(
            0.02,
            0.98,
            f"N={','.join(str(len(v)) for v in values)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            color="0.35",
        )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.29, top=0.98, wspace=0.55)
    save_figure(fig, out, stem)
    plt.close(fig)


def plot_combined(episode_rows: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.55))
    for row_index, case in enumerate(("case1", "case2")):
        key = "strict_all_8_groups_complete" if case == "case1" else "has_complete_group"
        for ax, (metric, label) in zip(axes[row_index], METRICS):
            values = [
                np.asarray(
                    [
                        row[metric]
                        for row in episode_rows
                        if row["case"] == case
                        and row["variant"] == variant
                        and row[key] == 1
                        and np.isfinite(row[metric])
                    ],
                    dtype=float,
                )
                for variant in VARIANTS
            ]
            bp = ax.boxplot(
                values,
                labels=[DISPLAY[v] for v in VARIANTS],
                widths=0.56,
                patch_artist=True,
                showmeans=True,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "white",
                    "markeredgecolor": "black",
                    "markersize": 3.0,
                },
                medianprops={"color": "black", "linewidth": 1.35},
                whiskerprops={"linewidth": 0.8},
                capprops={"linewidth": 0.8},
                flierprops={"marker": "o", "markersize": 1.8, "alpha": 0.25},
            )
            for box, variant in zip(bp["boxes"], VARIANTS):
                box.set_facecolor(COLORS[variant])
                box.set_alpha(0.68)
                box.set_linewidth(0.8)
            ax.set_ylabel(label, fontsize=9)
            ax.set_xticklabels(
                [DISPLAY[v] for v in VARIANTS], fontsize=6.8, rotation=18, ha="right"
            )
            ax.grid(True, axis="y", linestyle="--", linewidth=0.25, color="#CCCCCC", alpha=0.55)
            despine(ax)
            ax.text(
                0.02,
                0.98,
                f"N={','.join(str(len(v)) for v in values)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.2,
                color="0.35",
            )
    fig.text(0.012, 0.985, "(a) Case 1", ha="left", va="top", fontsize=9, fontweight="bold")
    fig.text(0.012, 0.500, "(b) Case 2", ha="left", va="top", fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.155, top=0.96, wspace=0.58, hspace=0.52)
    save_figure(fig, out, "ablation_terminal_metrics_boxplot_compare")
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "/home/uav/00gao_xueshu/DT_PAPER/XIU_tase_paper_V1/"
            "revise_0723/Re_xiaorong/raw_results/evaluation"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    group_rows, episode_rows = recompute(args.input_root)
    summary_rows = summarize(episode_rows)
    effect_rows = paired_effects(episode_rows)
    completion_rows = completion_summary(episode_rows)

    group_fields = [
        "variant",
        "variant_label",
        "case",
        "seed",
        "episode",
        "target_id",
        "assigned_member_count",
        "strict_all_8_groups_complete",
        "arrival_time_mean_s",
        "arrival_time_min_s",
        "arrival_time_max_s",
        *[key for key, _ in METRICS],
    ]
    episode_fields = [
        "variant",
        "variant_label",
        "case",
        "seed",
        "episode",
        "hit_count",
        "target_hit_count",
        "target_sync_count",
        "evaluator_all_hit",
        "evaluator_all_sync",
        "complete_group_count",
        "has_complete_group",
        "strict_all_8_groups_complete",
        *[key for key, _ in METRICS],
    ]
    summary_fields = [
        "case",
        "variant",
        "variant_label",
        "analysis_stratum",
        "metric",
        "N_episodes",
        "mean",
        "std",
        "median",
        "q1",
        "q3",
        "mean_ci95_low",
        "mean_ci95_high",
    ]
    effect_fields = [
        "case",
        "analysis_stratum",
        "comparison",
        "metric",
        "N_paired_episodes",
        "mean_paired_difference",
        "difference_ci95_low",
        "difference_ci95_high",
        "paired_effect_dz",
        "wilcoxon_p",
    ]
    completion_fields = [
        "case",
        "variant",
        "variant_label",
        "N_episodes",
        "interceptor_hit_rate",
        "target_hit_rate",
        "coordinated_target_rate",
        "complete_group_rate",
        "all_targets_hit_episode_rate",
        "all_targets_coordinated_episode_rate",
        "strict_all_8_groups_complete_episode_rate",
        "complete_group_conditioned_episode_count",
        "complete_group_count",
    ]
    write_csv(args.output_dir / "ablation_terminal_metrics_group.csv", group_rows, group_fields)
    write_csv(args.output_dir / "ablation_terminal_metrics_episode.csv", episode_rows, episode_fields)
    write_csv(args.output_dir / "ablation_terminal_metrics_summary.csv", summary_rows, summary_fields)
    write_csv(args.output_dir / "ablation_terminal_metrics_paired_effects.csv", effect_rows, effect_fields)
    write_csv(args.output_dir / "ablation_completion_summary.csv", completion_rows, completion_fields)

    plot_case(
        episode_rows,
        "case1",
        "strict_all_8_groups_complete",
        args.output_dir,
        "case1_ablation_metric_boxplot_compare",
    )
    plot_case(
        episode_rows,
        "case2",
        "complete_group_conditioned_episode",
        args.output_dir,
        "case2_ablation_metric_boxplot_compare",
    )
    plot_combined(episode_rows, args.output_dir)

    strict_counts = {
        f"{case}:{variant}": sum(
            row["strict_all_8_groups_complete"]
            for row in episode_rows
            if row["case"] == case and row["variant"] == variant
        )
        for case in ("case1", "case2")
        for variant in VARIANTS
    }
    conditional_counts = {
        f"{case}:{variant}": sum(
            row["has_complete_group"]
            for row in episode_rows
            if row["case"] == case and row["variant"] == variant
        )
        for case in ("case1", "case2")
        for variant in VARIANTS
    }
    validation = {
        "input_run_count": 40,
        "episode_count": len(episode_rows),
        "complete_group_row_count": len(group_rows),
        "strict_complete_episode_counts": strict_counts,
        "complete_group_conditioned_episode_counts": conditional_counts,
        "nan_inf_in_group_metrics": int(
            sum(
                not np.isfinite(row[key])
                for row in group_rows
                for key, _ in METRICS
            )
        ),
        "nan_metrics_only_when_no_complete_group": bool(
            all(
                all(np.isfinite(row[key]) for key, _ in METRICS)
                if row["has_complete_group"]
                else all(not np.isfinite(row[key]) for key, _ in METRICS)
                for row in episode_rows
            )
        ),
        "metric_note": (
            "E_n_terminal_sample_g is the mean resultant load recorded at first arrival; "
            "the current event logs do not contain the complete terminal measurement window."
        ),
    }
    with (args.output_dir / "validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2, ensure_ascii=False)

    manifest = []
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "sha256_manifest.csv":
            manifest.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_csv(args.output_dir / "sha256_manifest.csv", manifest, ["file", "bytes", "sha256"])
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
