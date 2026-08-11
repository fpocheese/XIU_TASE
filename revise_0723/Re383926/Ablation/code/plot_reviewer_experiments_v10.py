#!/usr/bin/env python3
"""IEEE-TASE V10-style figures for all reviewer supplementary experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SINGLE_COL_WIDTH = 3.5
DOUBLE_COL_WIDTH = 7.16
COLORS = {
    "full": "#0072B2",
    "no_trust": "#D55E00",
    "no_gru": "#009E73",
    "no_attention_residual": "#CC79A7",
}
LABELS = {
    "full": "Full ART-MAPPO",
    "no_trust": "No trust",
    "no_gru": "No GRU",
    "no_attention_residual": "No attention-residual",
}
SHORT_LABELS = {
    "full": "Full",
    "no_trust": "No trust",
    "no_gru": "No GRU",
    "no_attention_residual": "No A-R",
}
MARKERS = {
    "full": "o",
    "no_trust": "s",
    "no_gru": "^",
    "no_attention_residual": "D",
}
FAILURE_COLORS = {
    "mission_success": "#009E73",
    "delayed_cooperative_engagement": "#E69F00",
    "incomplete_cooperative_group": "#CC79A7",
    "unsuccessful_interception": "#D55E00",
}
FAILURE_LABELS = {
    "mission_success": "Mission success",
    "delayed_cooperative_engagement": "Delayed cooperation",
    "incomplete_cooperative_group": "Incomplete group",
    "unsuccessful_interception": "Unsuccessful interception",
}


def apply_v10_style():
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
            "legend.framealpha": 0.95,
            "legend.edgecolor": "0.7",
            "legend.fancybox": False,
            "legend.frameon": True,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "text.usetex": False,
        }
    )


def save_figure(fig, outdir: Path, stem: str):
    outdir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        fig.savefig(outdir / f"{stem}.{suffix}")
    fig.savefig(outdir / f"{stem}.png", dpi=600)
    plt.close(fig)


def panel_label(ax, label):
    ax.text(
        0.01,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )


def _rolling(values, window=5):
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(window, min_periods=1, center=True)
        .mean()
        .to_numpy()
    )


def load_training(root: Path):
    records = []
    for path in sorted(root.glob("**/training_metrics.csv")):
        parts = path.parts
        variant = next((v for v in LABELS if v in parts), None)
        case = next((v for v in ("case1", "case2") if v in parts), None)
        seed_part = next((v for v in parts if v.startswith("seed")), None)
        if variant is None or case is None or seed_part is None:
            continue
        data = pd.read_csv(path)
        data["variant"] = variant
        data["case"] = case
        data["seed"] = int(seed_part.replace("seed", ""))
        records.append(data)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def plot_training(training_root: Path, outdir: Path):
    data = load_training(training_root)
    if data.empty:
        return []
    stems = []
    for case in sorted(data["case"].unique()):
        subset = data[data["case"] == case]
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(DOUBLE_COL_WIDTH, 2.25),
            constrained_layout=True,
        )
        definitions = [
            ("mean_episode_return", "Episode return"),
            ("value_loss", "Critic loss"),
            ("entropy", "Policy entropy"),
        ]
        for ax, (metric, ylabel), label in zip(
            axes, definitions, ("(a)", "(b)", "(c)")
        ):
            for variant in LABELS:
                runs = []
                x_reference = None
                for _, seed_data in (
                    subset[subset["variant"] == variant]
                    .groupby("seed", sort=True)
                ):
                    seed_data = seed_data.sort_values("environment_steps")
                    x = seed_data["environment_steps"].to_numpy(dtype=float)
                    y = _rolling(seed_data[metric].to_numpy(dtype=float), 5)
                    if x_reference is None:
                        x_reference = x
                    if len(x) == len(x_reference) and np.allclose(
                        x, x_reference
                    ):
                        runs.append(y)
                if not runs:
                    continue
                values = np.vstack(runs)
                mean = np.mean(values, axis=0)
                std = (
                    np.std(values, axis=0, ddof=1)
                    if values.shape[0] > 1
                    else np.zeros_like(mean)
                )
                ax.plot(
                    x_reference,
                    mean,
                    color=COLORS[variant],
                    marker=MARKERS[variant],
                    markevery=max(1, len(mean) // 8),
                    label=LABELS[variant],
                )
                ax.fill_between(
                    x_reference,
                    mean - std,
                    mean + std,
                    color=COLORS[variant],
                    alpha=0.13,
                    linewidth=0,
                )
            ax.set_xlabel("Environment steps")
            ax.set_ylabel(ylabel)
            ax.grid(True)
            panel_label(ax, label)
            if metric == "value_loss":
                positive = subset[metric].to_numpy(dtype=float)
                if np.all(positive > 0) and (
                    np.nanmax(positive) / max(np.nanmin(positive), 1e-12) > 100
                ):
                    ax.set_yscale("log")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.20),
            ncol=4,
            columnspacing=0.8,
            handlelength=1.5,
        )
        stem = f"ablation_training_{case}_v10"
        save_figure(fig, outdir, stem)
        stems.append(stem)

        # Also export the three training diagnostics as independent
        # single-column figures.  The combined panel is convenient for the
        # response letter; these standalone files can be inserted separately
        # when the manuscript layout requires one figure per diagnostic.
        standalone_names = {
            "mean_episode_return": "reward",
            "value_loss": "critic_loss",
            "entropy": "policy_entropy",
        }
        for metric, ylabel in definitions:
            single, ax = plt.subplots(
                figsize=(SINGLE_COL_WIDTH, 2.55),
                constrained_layout=True,
            )
            for variant in LABELS:
                runs = []
                x_reference = None
                for _, seed_data in (
                    subset[subset["variant"] == variant]
                    .groupby("seed", sort=True)
                ):
                    seed_data = seed_data.sort_values("environment_steps")
                    x = seed_data["environment_steps"].to_numpy(dtype=float)
                    y = _rolling(
                        seed_data[metric].to_numpy(dtype=float), 5
                    )
                    if x_reference is None:
                        x_reference = x
                    if len(x) == len(x_reference) and np.allclose(
                        x, x_reference
                    ):
                        runs.append(y)
                if not runs:
                    continue
                values = np.vstack(runs)
                mean = np.mean(values, axis=0)
                std = (
                    np.std(values, axis=0, ddof=1)
                    if values.shape[0] > 1
                    else np.zeros_like(mean)
                )
                ax.plot(
                    x_reference,
                    mean,
                    color=COLORS[variant],
                    marker=MARKERS[variant],
                    markevery=max(1, len(mean) // 8),
                    label=LABELS[variant],
                )
                ax.fill_between(
                    x_reference,
                    mean - std,
                    mean + std,
                    color=COLORS[variant],
                    alpha=0.13,
                    linewidth=0,
                )
            ax.set_xlabel("Environment steps")
            ax.set_ylabel(ylabel)
            ax.grid(True)
            if metric == "value_loss":
                positive = subset[metric].to_numpy(dtype=float)
                if np.all(positive > 0) and (
                    np.nanmax(positive)
                    / max(np.nanmin(positive), 1e-12)
                    > 100
                ):
                    ax.set_yscale("log")
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.19),
                ncol=2,
                columnspacing=0.8,
                handlelength=1.5,
            )
            single_stem = (
                f"ablation_training_{standalone_names[metric]}_"
                f"{case}_v10"
            )
            save_figure(single, outdir, single_stem)
            stems.append(single_stem)
    return stems


def load_episode_files(root: Path):
    frames = []
    for path in sorted(root.glob("**/episodes.csv")):
        if "_chunks" in path.parts or "checkpoints" in path.parts:
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if len(frame):
            frame["source_file"] = str(path)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_target_files(root: Path):
    frames = []
    for path in sorted(root.glob("**/targets.csv")):
        if "_chunks" in path.parts or "checkpoints" in path.parts:
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if len(frame):
            frame["source_file"] = str(path)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _rate_ci(series):
    values = np.asarray(series, dtype=float)
    n = values.size
    if n == 0:
        return np.nan, np.nan, np.nan
    p = values.mean()
    z = 1.959963984540054
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = (
        z
        * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
        / denominator
    )
    return p, max(0.0, center - half), min(1.0, center + half)


def _boxplot(ax, groups, labels, colors, ylabel):
    clean = [
        np.asarray(values, dtype=float)[
            np.isfinite(np.asarray(values, dtype=float))
        ]
        for values in groups
    ]
    if not any(len(values) for values in clean):
        ax.set_xticks(np.arange(1, len(labels) + 1), labels)
        ax.set_xlim(0.5, len(labels) + 0.5)
        ax.text(
            0.5,
            0.5,
            "No successful trials",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="0.35",
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y")
        return
    bp = ax.boxplot(
        clean,
        labels=labels,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
        boxprops={"linewidth": 0.8},
    )
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.62)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y")


def plot_ablation_evaluation(evaluation_root: Path, outdir: Path):
    episodes = load_episode_files(evaluation_root)
    if episodes.empty or "variant" not in episodes:
        return []
    stems = []
    for case in sorted(episodes["case"].unique()):
        data = episodes[episodes["case"] == case]
        variants = [v for v in LABELS if v in set(data["variant"])]
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(DOUBLE_COL_WIDTH, 4.4),
            constrained_layout=True,
        )
        for ax, metric, ylabel, label in [
            (
                axes[0, 0],
                "target_coverage_success",
                "Target-coverage rate",
                "(a)",
            ),
            (
                axes[0, 1],
                "cooperative_success",
                "Cooperative-success rate",
                "(b)",
            ),
        ]:
            intervals = [
                _rate_ci(data[data["variant"] == v][metric])
                for v in variants
            ]
            means = np.asarray([item[0] for item in intervals])
            errors = np.asarray(
                [
                    [max(0.0, item[0] - item[1]) for item in intervals],
                    [max(0.0, item[2] - item[0]) for item in intervals],
                ]
            )
            x = np.arange(len(variants))
            ax.bar(
                x,
                means,
                yerr=errors,
                color=[COLORS[v] for v in variants],
                alpha=0.76,
                capsize=2.5,
                linewidth=0.6,
                edgecolor="black",
            )
            ax.set_xticks(x, [SHORT_LABELS[v] for v in variants])
            ax.set_ylim(0.0, 1.05)
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y")
            panel_label(ax, label)
        for ax, metric, ylabel, label in [
            (axes[1, 0], "E_co_time_s", r"$E_{\mathrm{co\mathrm{-}time}}$ (s)", "(c)"),
            (axes[1, 1], "E_miss_m", r"$E_{\mathrm{miss}}$ (m)", "(d)"),
        ]:
            _boxplot(
                ax,
                [data[data["variant"] == v][metric] for v in variants],
                [SHORT_LABELS[v] for v in variants],
                [COLORS[v] for v in variants],
                ylabel,
            )
            panel_label(ax, label)
        stem = f"ablation_monte_carlo_{case}_v10"
        save_figure(fig, outdir, stem)
        stems.append(stem)

        fig2, axes2 = plt.subplots(
            2,
            2,
            figsize=(DOUBLE_COL_WIDTH, 4.4),
            constrained_layout=True,
        )
        for ax, metric, ylabel, label in zip(
            axes2.flat,
            ("E_co_time_s", "E_n_g", "E_miss_m", "E_t_s"),
            (
                r"$E_{\mathrm{co\mathrm{-}time}}$ (s)",
                r"$E_n$ (g)",
                r"$E_{\mathrm{miss}}$ (m)",
                r"$E_t$ (s)",
            ),
            ("(a)", "(b)", "(c)", "(d)"),
        ):
            _boxplot(
                ax,
                [data[data["variant"] == v][metric] for v in variants],
                [SHORT_LABELS[v] for v in variants],
                [COLORS[v] for v in variants],
                ylabel,
            )
            panel_label(ax, label)
        terminal_stem = f"ablation_terminal_metrics_{case}_v10"
        save_figure(fig2, outdir, terminal_stem)
        stems.append(terminal_stem)
    return stems


def plot_failure_cases(supp_root: Path, outdir: Path):
    episodes = load_episode_files(supp_root)
    if episodes.empty or "failure_class" not in episodes:
        return []
    mask = episodes["condition"].astype(str).str.startswith("failure_")
    data = episodes[mask].copy()
    if data.empty:
        return []
    conditions = list(dict.fromkeys(data["condition"].tolist()))
    display = [c.replace("failure_", "").replace("_", " ") for c in conditions]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COL_WIDTH, 4.45),
        constrained_layout=True,
    )
    bottom = np.zeros(len(conditions))
    for category in FAILURE_LABELS:
        fractions = []
        for condition in conditions:
            values = data[data["condition"] == condition]["failure_class"]
            fractions.append(float(np.mean(values == category)))
        axes[0, 0].barh(
            np.arange(len(conditions)),
            fractions,
            left=bottom,
            color=FAILURE_COLORS[category],
            label=FAILURE_LABELS[category],
        )
        bottom += np.asarray(fractions)
    axes[0, 0].set_yticks(np.arange(len(conditions)), display)
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_xlabel("Episode fraction")
    failure_handles, failure_legend_labels = (
        axes[0, 0].get_legend_handles_labels()
    )
    fig.legend(
        failure_handles,
        failure_legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=4,
        columnspacing=0.8,
        handlelength=1.6,
    )
    panel_label(axes[0, 0], "(a)")

    for metric, color, marker, label in [
        ("target_coverage_success", "#0072B2", "o", "Target coverage"),
        ("cooperative_success", "#D55E00", "s", "Cooperative success"),
        ("mission_success", "#009E73", "^", "Mission success"),
    ]:
        means = []
        intervals = []
        for condition in conditions:
            mean, low, high = _rate_ci(
                data[data["condition"] == condition][metric]
            )
            means.append(mean)
            intervals.append(
                (max(0.0, mean - low), max(0.0, high - mean))
            )
        axes[0, 1].errorbar(
            np.arange(len(conditions)),
            means,
            yerr=np.asarray(intervals, dtype=float).T,
            color=color,
            marker=marker,
            capsize=2.5,
            label=label,
        )
    axes[0, 1].set_xticks(
        np.arange(len(conditions)), display, rotation=20
    )
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_ylabel("Success rate")
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    panel_label(axes[0, 1], "(b)")

    _boxplot(
        axes[1, 0],
        [
            data[data["condition"] == c]["worst_closest_approach_m"]
            for c in conditions
        ],
        display,
        ["#56B4E9"] * len(conditions),
        "Worst closest approach (m)",
    )
    axes[1, 0].tick_params(axis="x", rotation=20)
    panel_label(axes[1, 0], "(c)")
    _boxplot(
        axes[1, 1],
        [
            data[data["condition"] == c]["E_co_time_s"]
            for c in conditions
        ],
        display,
        ["#E69F00"] * len(conditions),
        r"$E_{\mathrm{co\mathrm{-}time}}$ (s)",
    )
    axes[1, 1].tick_params(axis="x", rotation=20)
    panel_label(axes[1, 1], "(d)")
    stem = "failure_case_analysis_v10"
    save_figure(fig, outdir, stem)
    return [stem]


def plot_generalization(supp_root: Path, outdir: Path):
    episodes = load_episode_files(supp_root)
    if episodes.empty:
        return []
    mask = episodes["condition"].astype(str).str.startswith("generalization_")
    data = episodes[mask].copy()
    if data.empty:
        return []
    order = ["nominal", "chirp", "multisine", "jink"]
    patterns = [p for p in order if p in set(data["attack_pattern"])]
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"][: len(patterns)]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COL_WIDTH, 4.4),
        constrained_layout=True,
    )
    for ax, metric, ylabel, label in [
        (axes[0, 0], "target_coverage_success", "Target-coverage rate", "(a)"),
        (axes[0, 1], "mission_success", "Mission-success rate", "(b)"),
    ]:
        stats = [
            _rate_ci(data[data["attack_pattern"] == p][metric])
            for p in patterns
        ]
        means = [x[0] for x in stats]
        errors = np.asarray(
            [
                [max(0.0, x[0] - x[1]) for x in stats],
                [max(0.0, x[2] - x[0]) for x in stats],
            ],
            dtype=float,
        )
        x = np.arange(len(patterns))
        ax.bar(
            x,
            means,
            yerr=errors,
            color=colors,
            capsize=2.5,
            alpha=0.78,
            edgecolor="black",
            linewidth=0.6,
        )
        ax.set_xticks(x, [p.capitalize() for p in patterns])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y")
        panel_label(ax, label)
    for ax, metric, ylabel, label in [
        (axes[1, 0], "E_co_time_s", r"$E_{\mathrm{co\mathrm{-}time}}$ (s)", "(c)"),
        (axes[1, 1], "E_miss_m", r"$E_{\mathrm{miss}}$ (m)", "(d)"),
    ]:
        _boxplot(
            ax,
            [data[data["attack_pattern"] == p][metric] for p in patterns],
            [p.capitalize() for p in patterns],
            colors,
            ylabel,
        )
        panel_label(ax, label)
    stem = "unseen_maneuver_generalization_v10"
    save_figure(fig, outdir, stem)
    return [stem]


def plot_end_to_end(supp_root: Path, outdir: Path):
    episodes = load_episode_files(supp_root)
    if episodes.empty:
        return []
    mask = episodes["condition"].astype(str).str.startswith("end_to_end_")
    data = episodes[mask].copy()
    if data.empty:
        return []
    modes = [m for m in ("fixed", "idbo") if m in set(data["assignment_mode"])]
    colors = {"fixed": "#999999", "idbo": "#0072B2"}
    cases = [c for c in ("case1", "case2") if c in set(data["case"])]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COL_WIDTH, 4.4),
        constrained_layout=True,
    )
    width = 0.34
    x = np.arange(len(cases))
    for offset_index, mode in enumerate(modes):
        offsets = (offset_index - (len(modes) - 1) / 2) * width
        rates = []
        intervals = []
        for case in cases:
            values = data[
                (data["case"] == case) & (data["assignment_mode"] == mode)
            ]["mission_success"]
            mean, low, high = _rate_ci(values)
            rates.append(mean)
            intervals.append(
                (max(0.0, mean - low), max(0.0, high - mean))
            )
        axes[0, 0].bar(
            x + offsets,
            rates,
            width,
            yerr=np.asarray(intervals, dtype=float).T,
            color=colors[mode],
            capsize=2.5,
            label=mode.upper() if mode == "idbo" else "Fixed",
        )
    axes[0, 0].set_xticks(x, [c.replace("case", "Case ") for c in cases])
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_ylabel("Mission-success rate")
    axes[0, 0].grid(True, axis="y")
    assignment_handles, assignment_labels = (
        axes[0, 0].get_legend_handles_labels()
    )
    fig.legend(
        assignment_handles,
        assignment_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=2,
        columnspacing=1.0,
        handlelength=1.6,
    )
    panel_label(axes[0, 0], "(a)")

    idbo = data[data["assignment_mode"] == "idbo"]
    _boxplot(
        axes[0, 1],
        [idbo[idbo["case"] == c]["idbo_runtime_ms"] for c in cases],
        [c.replace("case", "Case ") for c in cases],
        ["#56B4E9"] * len(cases),
        "IDBO runtime (ms)",
    )
    panel_label(axes[0, 1], "(b)")

    for ax, metric, ylabel, label in [
        (axes[1, 0], "E_co_time_s", r"$E_{\mathrm{co\mathrm{-}time}}$ (s)", "(c)"),
        (axes[1, 1], "E_miss_m", r"$E_{\mathrm{miss}}$ (m)", "(d)"),
    ]:
        groups, labels, group_colors = [], [], []
        for case in cases:
            for mode in modes:
                groups.append(
                    data[
                        (data["case"] == case)
                        & (data["assignment_mode"] == mode)
                    ][metric]
                )
                labels.append(
                    f"{case.replace('case', 'C')}-{mode.upper() if mode == 'idbo' else 'Fix'}"
                )
                group_colors.append(colors[mode])
        _boxplot(ax, groups, labels, group_colors, ylabel)
        ax.tick_params(axis="x", rotation=15)
        panel_label(ax, label)
    stem = "end_to_end_assignment_guidance_v10"
    save_figure(fig, outdir, stem)
    return [stem]


def plot_case3_end_to_end(supp_root: Path, outdir: Path):
    episodes = load_episode_files(supp_root)
    if episodes.empty:
        return []
    data = episodes[
        episodes["condition"].astype(str)
        == "end_to_end_case3_idbo_hybrid"
    ].copy()
    if data.empty:
        return []

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COL_WIDTH, 4.4),
        constrained_layout=True,
    )
    rate_definitions = [
        ("target_coverage_success", "Coverage"),
        ("all_defenders_hit", "All defenders"),
        ("cooperative_success", "Cooperative"),
        ("mission_success", "Mission"),
    ]
    x = np.arange(len(rate_definitions))
    means, intervals = [], []
    for metric, _ in rate_definitions:
        mean, low, high = _rate_ci(data[metric])
        means.append(mean)
        intervals.append(
            (max(0.0, mean - low), max(0.0, high - mean))
        )
    axes[0, 0].bar(
        x,
        means,
        yerr=np.asarray(intervals, dtype=float).T,
        color=["#56B4E9", "#009E73", "#E69F00", "#0072B2"],
        capsize=2.5,
        width=0.66,
    )
    axes[0, 0].set_xticks(
        x,
        [label for _, label in rate_definitions],
        rotation=12,
    )
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_ylabel("Success rate")
    axes[0, 0].grid(True, axis="y")
    panel_label(axes[0, 0], "(a)")

    _boxplot(
        axes[0, 1],
        [data["idbo_runtime_ms"]],
        ["Case 3"],
        ["#56B4E9"],
        "IDBO runtime (ms)",
    )
    panel_label(axes[0, 1], "(b)")

    _boxplot(
        axes[1, 0],
        [data["E_co_time_s"]],
        ["Case 3"],
        ["#009E73"],
        r"$E_{\mathrm{co\mathrm{-}time}}$ (s)",
    )
    panel_label(axes[1, 0], "(c)")

    _boxplot(
        axes[1, 1],
        [data["E_miss_m"]],
        ["Case 3"],
        ["#D55E00"],
        r"$E_{\mathrm{miss}}$ (m)",
    )
    panel_label(axes[1, 1], "(d)")

    stem = "case3_idbo_art_mappo_end_to_end_v10"
    save_figure(fig, outdir, stem)
    return [stem]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_root", type=Path)
    parser.add_argument("--evaluation_root", type=Path)
    parser.add_argument("--supplementary_root", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    apply_v10_style()
    generated = []
    if args.training_root:
        generated += plot_training(args.training_root, args.outdir)
    if args.evaluation_root:
        generated += plot_ablation_evaluation(
            args.evaluation_root, args.outdir
        )
    if args.supplementary_root:
        generated += plot_failure_cases(args.supplementary_root, args.outdir)
        generated += plot_generalization(args.supplementary_root, args.outdir)
        generated += plot_end_to_end(args.supplementary_root, args.outdir)
        generated += plot_case3_end_to_end(
            args.supplementary_root, args.outdir
        )
    manifest = {
        "style": "IEEE TASE V10",
        "style_reference": (
            "new_sim_fig/fig_and_data figures_v10; "
            "ieee_plot_v10_tase.py -> ieee_plot_v9_tase.py rcParams"
        ),
        "single_column_width_in": SINGLE_COL_WIDTH,
        "double_column_width_in": DOUBLE_COL_WIDTH,
        "png_dpi": 600,
        "generated_stems": generated,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
