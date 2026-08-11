#!/usr/bin/env python3
"""Generate the complete V10 figure suite for one failure run per case.

The revised selection uses Case 1 seed 76048 (incomplete group A6) and Case 2
seed 77008 (incomplete group A2).  The script consumes
the evaluator-native NPZ and hit-event CSV files and never changes trajectory
samples or policy weights.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path("/home/a2rl/reviewer_failure_cases_20260730")
INPUT = ROOT / "results/selected_six_failure_cases"
OUTPUT = ROOT / "results/selected_representative_full_v10_v2"
EXPORT = OUTPUT / "v10_export"
FIGURES = OUTPUT / "figures"
SCRIPTS = ROOT / "code/onpolicy/scripts"
PRESET = ROOT / "presets/paper_case_presets_original_assignment_verified.npz"
DT = 0.05
HIT_RADIUS_M = 3.0
SYNC_TOL_S = 0.5
SELECTED = {"case1": 76048, "case2": 77008}
PREFIX = {"case1": "nopn", "case2": "sin"}
FOLDER = {"case1": "mappo_success_nopn", "case2": "mappo_success_sin"}
MISS_COLOR = "#D55E00"
HIT_COLOR = "#009E73"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_episode(case: str, seed: int, assignment: np.ndarray) -> dict[str, Any]:
    folder = INPUT / case / f"seed_{seed}" / case
    npz_path = folder / f"{case}_selected_episode.npz"
    event_path = folder / f"{case}_hit_events.csv"
    z = np.load(npz_path)
    ep = {
        "case": case,
        "seed": seed,
        "rep_att": np.asarray(z["rep_att"], dtype=float),
        "rep_def": np.asarray(z["rep_def"], dtype=float),
        "rep_ctrl": np.asarray(z["rep_ctrl"], dtype=float),
        "rep_tgo": np.asarray(z["rep_tgo"], dtype=float),
        "events": read_csv(event_path),
        "npz_path": str(npz_path),
        "event_path": str(event_path),
    }
    for key in ("rep_att", "rep_def", "rep_ctrl", "rep_tgo"):
        if not np.isfinite(ep[key]).all():
            raise ValueError(f"NaN/Inf in {key}: {npz_path}")
    ep["event_by_def"] = {
        int(row["defender_id"]): row for row in ep["events"]
    }
    missed = sorted(set(range(20)) - set(ep["event_by_def"]))
    if len(ep["events"]) != 19 or len(missed) != 1:
        raise ValueError(f"{case}/{seed} is not a 19/20 episode")
    ep["missed_id"] = missed[0]
    ep["mapping"] = {i: int(assignment[i] - 20) for i in range(20)}
    return ep


def build_v10_arrays(ep: dict[str, Any]) -> dict[str, np.ndarray]:
    deff = ep["rep_def"]
    att = ep["rep_att"]
    ctrl = ep["rep_ctrl"]
    tgo = ep["rep_tgo"]
    steps = min(len(deff), len(att), len(ctrl), len(tgo))
    deff, att, ctrl, tgo = (
        deff[:steps],
        att[:steps],
        ctrl[:steps],
        tgo[:steps],
    )
    agentspos = np.zeros((steps, 56), dtype=float)
    agentsall = np.zeros((steps, 40), dtype=float)
    agentsnz = np.zeros((steps, 20), dtype=float)
    agentsvel = np.zeros((steps, 40), dtype=float)
    agentstimetgo = np.zeros((steps, 40), dtype=float)

    velocity = np.gradient(deff, DT, axis=0)
    horizontal_speed = np.hypot(velocity[:, :, 0], velocity[:, :, 1])
    yaw = np.arctan2(velocity[:, :, 1], velocity[:, :, 0])
    for defender_id in range(20):
        agentspos[:, 2 * defender_id] = deff[:, defender_id, 0]
        agentspos[:, 2 * defender_id + 1] = deff[:, defender_id, 1]
        agentsall[:, 2 * defender_id] = ctrl[:, defender_id, 0]
        agentsall[:, 2 * defender_id + 1] = ctrl[:, defender_id, 1]
        agentsnz[:, defender_id] = ctrl[:, defender_id, 2]
        agentsvel[:, 2 * defender_id] = horizontal_speed[:, defender_id]
        agentsvel[:, 2 * defender_id + 1] = yaw[:, defender_id]
        target_index = ep["mapping"][defender_id]
        distance = np.linalg.norm(
            att[:, target_index, :] - deff[:, defender_id, :], axis=1
        )
        agentstimetgo[:, 2 * defender_id] = tgo[:, defender_id]
        agentstimetgo[:, 2 * defender_id + 1] = distance
    for attacker_id in range(8):
        agentspos[:, 40 + 2 * attacker_id] = att[:, attacker_id, 0]
        agentspos[:, 40 + 2 * attacker_id + 1] = att[:, attacker_id, 1]
    return {
        "agentspos": agentspos,
        "agentsall": agentsall,
        "agentsnz": agentsnz,
        "agentsvel": agentsvel,
        "agentstimetgo": agentstimetgo,
    }


def save_export(case: str, arrays: dict[str, np.ndarray]) -> Path:
    folder = EXPORT / FOLDER[case]
    folder.mkdir(parents=True, exist_ok=True)
    for key, array in arrays.items():
        np.savetxt(folder / f"{key}.txt", array, fmt="%.9f")
    return folder


def make_res(ep: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    n_steps = arrays["agentspos"].shape[0]
    hit_end = np.full(20, n_steps - 1, dtype=int)
    for defender_id, row in ep["event_by_def"].items():
        hit_end[defender_id] = int(row["step"])
    plot_end = np.maximum(hit_end - 2, 1)
    return {
        "data": arrays,
        "repeat_start_rows": hit_end,
        "plot_end_rows": plot_end,
        "episode_end": n_steps - 1,
        "mapping": ep["mapping"],
        "impact_angles": {},
        "stats": {},
    }


def install_three_format_save(v9) -> None:
    def save(fig, out: Path, stem: str) -> None:
        replacements = {
            "mappo_nopn": "artmappo_case1_seed76048",
            "mappo_sin": "artmappo_case2_seed77008",
            "standalone_duav_legend": "artmappo_duav_legend",
        }
        for old, new in replacements.items():
            stem = stem.replace(old, new)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{stem}.pdf")
        fig.savefig(out / f"{stem}.svg")
        fig.savefig(out / f"{stem}.png", dpi=600)
        plt.close(fig)
        print(f"[V10] {stem}", flush=True)

    v9._save_fig = save


def save_custom(fig, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf")
    fig.savefig(FIGURES / f"{stem}.svg")
    fig.savefig(FIGURES / f"{stem}.png", dpi=600)
    plt.close(fig)
    print(f"[V10 failure] {stem}", flush=True)


def closest_miss(ep: dict[str, Any]) -> tuple[int, float]:
    defender_id = ep["missed_id"]
    target_index = ep["mapping"][defender_id]
    distance = np.linalg.norm(
        ep["rep_def"][:, defender_id, :]
        - ep["rep_att"][:, target_index, :],
        axis=1,
    )
    step = int(np.argmin(distance))
    return step, float(distance[step])


def plot_failure_trajectory_2d(v9, ep: dict[str, Any]) -> None:
    fig, ax = plt.subplots(
        figsize=(v9.SINGLE_COL_WIDTH, v9.SINGLE_COL_WIDTH * 1.20)
    )
    colors = v9.ACADEMIC_COLORS_8
    deff, att = ep["rep_def"], ep["rep_att"]
    for defender_id in range(20):
        target_index = ep["mapping"][defender_id]
        p = deff[:, defender_id, :]
        event = ep["event_by_def"].get(defender_id)
        stop = int(event["step"]) + 1 if event else len(p)
        if defender_id == ep["missed_id"]:
            ax.plot(
                p[:stop, 0], p[:stop, 1],
                color=MISS_COLOR, ls="--", lw=2.0, zorder=6,
            )
        else:
            ax.plot(
                p[:stop, 0], p[:stop, 1],
                color=colors[target_index], lw=1.15, alpha=0.82,
            )
        ax.plot(
            p[0, 0], p[0, 1], "o", color=colors[target_index],
            ms=3, mfc="white", mew=0.6, zorder=7,
        )
    for target_index in range(8):
        p = att[:, target_index, :]
        ax.plot(
            p[:, 0], p[:, 1], "--", color=colors[target_index],
            lw=1.25, alpha=0.9,
        )
        ax.plot(p[0, 0], p[0, 1], "s", color=colors[target_index], ms=3.5)
    events = ep["events"]
    ax.scatter(
        [float(row["defender_pos_x"]) for row in events],
        [float(row["defender_pos_y"]) for row in events],
        s=13, marker="o", c=HIT_COLOR, edgecolors="white",
        linewidths=0.35, zorder=8,
    )
    miss_step, _ = closest_miss(ep)
    miss_p = deff[miss_step, ep["missed_id"], :]
    ax.plot(
        miss_p[0], miss_p[1], marker="X", ms=6.5, color=MISS_COLOR,
        mec="black", mew=0.45, zorder=9,
    )
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls="--", lw=0.3, color="#CCCCCC", alpha=0.5)
    handles, labels = [], []
    for target_index in range(8):
        handles.append(Line2D([0], [0], color=colors[target_index], lw=1.8))
        labels.append(rf"Group $A_{{{target_index + 1}}}$")
    handles.extend(
        [
            Line2D([0], [0], color="gray", lw=1.2, label="Successful"),
            Line2D([0], [0], color=MISS_COLOR, lw=2.0, ls="--",
                   label="Missed"),
            Line2D([0], [0], color="gray", lw=1.2, ls="--",
                   label="Attacker"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=HIT_COLOR, label="Hit event"),
            Line2D([0], [0], marker="X", color="none",
                   markerfacecolor=MISS_COLOR, markeredgecolor="black",
                   label="Closest approach"),
        ]
    )
    labels.extend(
        ["Successful", "Missed", "Attacker", "Hit event", "Closest approach"]
    )
    ax.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.01),
        ncol=4, fontsize=5.5, frameon=True, edgecolor="0.7",
        fancybox=False, handlelength=1.3, columnspacing=0.55,
    )
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.10, top=0.74)
    save_custom(
        fig, f"artmappo_{ep['case']}_seed{ep['seed']}_trajectory"
    )


def plot_failure_trajectory_3d(v9, ep: dict[str, Any]) -> None:
    colors = v9.ACADEMIC_COLORS_8
    fig = plt.figure(
        figsize=(v9.SINGLE_COL_WIDTH, v9.SINGLE_COL_WIDTH * 0.95)
    )
    ax = fig.add_subplot(111, projection="3d")
    for defender_id in range(20):
        target_index = ep["mapping"][defender_id]
        p = ep["rep_def"][:, defender_id, :]
        event = ep["event_by_def"].get(defender_id)
        stop = int(event["step"]) + 1 if event else len(p)
        if defender_id == ep["missed_id"]:
            ax.plot(
                p[:stop, 0], p[:stop, 1], p[:stop, 2],
                color=MISS_COLOR, ls="--", lw=1.8, zorder=6,
            )
        else:
            ax.plot(
                p[:stop, 0], p[:stop, 1], p[:stop, 2],
                color=colors[target_index], lw=0.85, alpha=0.8,
            )
    for target_index in range(8):
        p = ep["rep_att"][:, target_index, :]
        ax.plot(
            p[:, 0], p[:, 1], p[:, 2], "--",
            color=colors[target_index], lw=1.0, alpha=0.9,
        )
        ax.plot(
            [p[0, 0]], [p[0, 1]], [p[0, 2]], "s",
            color=colors[target_index], ms=3.0,
        )
    event_pos = np.array(
        [
            [float(r["defender_pos_x"]), float(r["defender_pos_y"]),
             float(r["defender_pos_z"])]
            for r in ep["events"]
        ]
    )
    ax.scatter(
        event_pos[:, 0], event_pos[:, 1], event_pos[:, 2],
        s=8, marker="o", c=HIT_COLOR, edgecolors="white",
        linewidths=0.25, depthshade=False,
    )
    miss_step, _ = closest_miss(ep)
    p = ep["rep_def"][miss_step, ep["missed_id"], :]
    ax.scatter(
        [p[0]], [p[1]], [p[2]], marker="X", s=28, c=MISS_COLOR,
        edgecolors="black", linewidths=0.45, depthshade=False,
    )
    ax.set_xlabel(r"$x$ (m)", labelpad=2)
    ax.set_ylabel(r"$y$ (m)", labelpad=2)
    ax.set_zlabel(r"$z$ (m)", labelpad=2)
    ax.tick_params(labelsize=7, pad=0)
    ax.grid(True, ls="--", lw=0.3, color="#CCCCCC", alpha=0.5)
    ax.view_init(elev=23, azim=-58)
    fig.subplots_adjust(left=0.01, right=0.98, bottom=0.01, top=0.99)
    save_custom(
        fig, f"artmappo_{ep['case']}_seed{ep['seed']}_trajectory_3d"
    )


def plot_failure_time_sync(v9, ep: dict[str, Any]) -> None:
    fig, ax = plt.subplots(
        figsize=(v9.SINGLE_COL_WIDTH, v9.SINGLE_COL_WIDTH * 0.55)
    )
    values, labels, colors, hatches, annotations = [], [], [], [], []
    incomplete_indices = []
    horizon = (len(ep["rep_def"]) - 1) * DT
    for target_index in range(8):
        members = [
            d for d, target in ep["mapping"].items() if target == target_index
        ]
        times = [
            float(ep["event_by_def"][d]["time"])
            for d in members if d in ep["event_by_def"]
        ]
        complete = len(times) == len(members)
        if complete:
            value = max(times) - min(times) if len(times) > 1 else 0.0
            annotations.append(f"{value:.2f}")
            colors.append(v9.ACADEMIC_COLORS_8[target_index])
            hatches.append("")
        else:
            value = horizon - min(times) if times else horizon
            annotations.append(f">{value:.1f}")
            colors.append(MISS_COLOR)
            hatches.append("///")
            incomplete_indices.append(target_index)
        values.append(value)
        labels.append(rf"$A_{{{target_index + 1}}}$")
    bars = ax.bar(
        labels, values, color=colors, edgecolor="black",
        linewidth=0.5, alpha=0.85,
    )
    for bar, hatch, text in zip(bars, hatches, annotations):
        bar.set_hatch(hatch)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.018,
            text, ha="center", va="bottom", fontsize=6, fontweight="bold",
        )
    ax.axhline(
        SYNC_TOL_S, color="black", ls="--", lw=0.9,
        label=r"Synchronization tolerance ($0.5$ s)",
    )
    ax.set_xlabel("Target")
    ax.set_ylabel(r"$\Delta t$ (s)")
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(True, axis="y", ls="--", lw=0.2, color="#CCC", alpha=0.5)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.02),
        fontsize=6.2, frameon=True, fancybox=False,
        borderaxespad=0.0,
    )
    fig.tight_layout()
    save_custom(
        fig, f"artmappo_{ep['case']}_seed{ep['seed']}_time_sync"
    )


def write_metadata(episodes: list[dict[str, Any]]) -> None:
    rows = []
    for ep in episodes:
        miss_step, miss_distance = closest_miss(ep)
        selection_text = (
            "earliest exactly-one-miss qualifying Case-1 seed whose "
            "incomplete group differs from the Case-2 A2 group"
            if ep["case"] == "case1"
            else "earliest qualifying Case-2 seed"
        )
        rows.append(
            {
                "case": ep["case"],
                "seed": ep["seed"],
                "selection": selection_text,
                "interceptor_hits": 19,
                "target_coverage": 8,
                "missed_defender_id": ep["missed_id"],
                "assigned_target_id": ep["mapping"][ep["missed_id"]] + 20,
                "missed_min_distance_m": miss_distance,
                "miss_margin_m": miss_distance - HIT_RADIUS_M,
                "closest_approach_time_s": miss_step * DT,
                "training_performed": False,
            }
        )
    with (OUTPUT / "selected_representative_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "selected_runs": SELECTED,
        "figure_types_per_case": [
            "trajectory",
            "trajectory_3d",
            "nx",
            "ny",
            "nz",
            "velocity",
            "heading",
            "yaw",
            "pitch",
            "tgo",
            "tgo_error",
            "distance",
            "time_sync",
        ],
        "selection_rule": (
            "Both runs satisfy 8/8 target coverage, 19/20 interceptor hits, "
            "seven complete groups, and observed hit spread <=0.5 s. Case 2 "
            "uses the earliest qualifying seed. Case 1 uses the earliest "
            "exactly-one-miss qualifying seed whose incomplete group differs "
            "from Case 2's A2 group; this yields A6 without altering any "
            "trajectory samples."
        ),
        "all_figures_derived_from_native_trajectories": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "rows": rows,
    }
    (OUTPUT / "full_v10_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    preset = np.load(PRESET)
    assignment = np.asarray(preset["assignment"], dtype=int)
    import sys

    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    v9 = load_module("ieee_plot_v9_tase", SCRIPTS / "ieee_plot_v9_tase.py")
    # ieee_plot_v10_tase imports ieee_plot_v9_tase by module name.
    sys.modules["ieee_plot_v9_tase"] = v9
    v10 = load_module("ieee_plot_v10_tase", SCRIPTS / "ieee_plot_v10_tase.py")
    install_three_format_save(v9)

    episodes = []
    for case, seed in SELECTED.items():
        ep = load_episode(case, seed, assignment)
        arrays = build_v10_arrays(ep)
        save_export(case, arrays)
        res = make_res(ep, arrays)
        prefix = PREFIX[case]
        method = "MAPPO"
        episode_for_v10 = {
            "rep_def": ep["rep_def"],
            "rep_att": ep["rep_att"],
            "mapping": [ep["mapping"][i] for i in range(20)],
        }

        plot_failure_trajectory_2d(v9, ep)
        plot_failure_trajectory_3d(v9, ep)
        v10._plot_overload(v9.DataLoader(str(EXPORT)), res, prefix, method,
                           FIGURES, "ny")
        v10._plot_overload(v9.DataLoader(str(EXPORT)), res, prefix, method,
                           FIGURES, "nz")
        loader = v9.DataLoader(str(EXPORT))
        v9.plot_nx_single(loader, res, prefix, method, FIGURES)
        v9.plot_velocity_single(loader, res, prefix, method, FIGURES)
        v9.plot_heading_single(loader, res, prefix, method, FIGURES)
        v10._plot_attitude_angles(
            loader, res, episode_for_v10, prefix, method, FIGURES
        )
        v9.plot_tgo_single(loader, res, prefix, method, FIGURES)
        v9.plot_tgo_error_single(loader, res, prefix, method, FIGURES)
        v9.plot_distance_single(loader, res, prefix, method, FIGURES)
        plot_failure_time_sync(v9, ep)
        episodes.append(ep)

    v9.plot_standalone_duav_legend(FIGURES)
    write_metadata(episodes)
    expected = 2 * 13 + 1
    for suffix in ("png", "pdf", "svg"):
        count = len(list(FIGURES.glob(f"*.{suffix}")))
        if count != expected:
            raise RuntimeError(f"expected {expected} {suffix} figures, got {count}")
    print(
        json.dumps(
            {
                "status": "complete",
                "selected": SELECTED,
                "figure_types_per_case": 13,
                "figure_triplets": expected,
                "output": str(OUTPUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
