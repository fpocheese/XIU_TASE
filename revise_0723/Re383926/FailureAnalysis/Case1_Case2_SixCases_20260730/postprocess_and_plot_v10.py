#!/usr/bin/env python3
"""Quantify and plot six frozen-policy partial-interception failure cases.

The script reads the evaluator's native NPZ and hit-event CSV files.  It does
not alter trajectories, replay the policy, train a network, or synthesize
observations.  All derived tables and V10-style figures are deterministic
functions of the six selected remote evaluation episodes.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path("/home/a2rl/reviewer_failure_cases_20260730")
INPUT = ROOT / "results/selected_six_failure_cases"
OUTPUT = ROOT / "results/failure_case_analysis_v10"
PRESET = ROOT / "presets/paper_case_presets_original_assignment_verified.npz"
DT = 0.05
HIT_RADIUS_M = 3.0
SYNC_TOL_S = 0.5
SELECTED = {
    "case1": [76014, 76048, 76052],
    "case2": [77008, 77020, 77023],
}

# Okabe-Ito/V10-compatible colorblind-safe palette.
COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#000000",
]
MISS_COLOR = "#D55E00"
HIT_COLOR = "#009E73"
GRAY = "#8A8A8A"


def set_v10_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.15,
            "lines.markersize": 3.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields=None) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    if fields is None:
        fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def input_paths(case: str, seed: int) -> tuple[Path, Path, Path]:
    folder = INPUT / case / f"seed_{seed}" / case
    return (
        folder / f"{case}_selected_episode.npz",
        folder / f"{case}_hit_events.csv",
        folder / f"{case}_episode_summary.csv",
    )


def load_episode(case: str, seed: int, assignment: np.ndarray) -> dict[str, Any]:
    npz_path, events_path, summary_path = input_paths(case, seed)
    z = np.load(npz_path)
    att = np.asarray(z["rep_att"], dtype=float)
    deff = np.asarray(z["rep_def"], dtype=float)
    ctrl = np.asarray(z["rep_ctrl"], dtype=float)
    tgo = np.asarray(z["rep_tgo"], dtype=float)
    if att.shape != (1501, 8, 3) or deff.shape != (1501, 20, 3):
        raise ValueError(f"unexpected trajectory shape in {npz_path}")
    if ctrl.shape != (1501, 20, 3) or tgo.shape != (1501, 20):
        raise ValueError(f"unexpected controller array shape in {npz_path}")
    for name, arr in {"att": att, "def": deff, "ctrl": ctrl, "tgo": tgo}.items():
        if not np.isfinite(arr).all():
            raise ValueError(f"NaN/Inf in {name}: {npz_path}")

    events = read_csv(events_path)
    summary_rows = read_csv(summary_path)
    if len(summary_rows) != 1:
        raise ValueError(f"expected one summary row: {summary_path}")
    summary = summary_rows[0]
    event_by_def = {int(row["defender_id"]): row for row in events}
    hit_ids = set(event_by_def)
    missed_ids = sorted(set(range(20)) - hit_ids)
    if len(events) != 19 or len(missed_ids) != 1:
        raise ValueError(f"{case}/{seed}: expected exactly 19 hits and one miss")
    if int(summary["target_hit_count"]) != 8:
        raise ValueError(f"{case}/{seed}: expected 8/8 target coverage")

    time = np.arange(att.shape[0], dtype=float) * DT
    distance = np.empty((att.shape[0], 20), dtype=float)
    for defender_id in range(20):
        target_index = int(assignment[defender_id] - 20)
        distance[:, defender_id] = np.linalg.norm(
            deff[:, defender_id, :] - att[:, target_index, :], axis=1
        )

    return {
        "case": case,
        "seed": seed,
        "att": att,
        "def": deff,
        "ctrl": ctrl,
        "tgo": tgo,
        "time": time,
        "distance": distance,
        "events": events,
        "event_by_def": event_by_def,
        "missed_id": missed_ids[0],
        "summary": summary,
        "npz_path": npz_path,
        "events_path": events_path,
    }


def derive_tables(
    episodes: list[dict[str, Any]], assignment: np.ndarray
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    episode_rows: list[dict[str, Any]] = []
    defender_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []

    for ep in episodes:
        case, seed = ep["case"], ep["seed"]
        missed_id = ep["missed_id"]
        missed_dist = ep["distance"][:, missed_id]
        miss_step = int(np.argmin(missed_dist))
        miss_min = float(missed_dist[miss_step])

        group_spreads = []
        complete_groups = 0
        covered_groups = 0
        for target_id in range(20, 28):
            assigned = np.flatnonzero(assignment == target_id).astype(int).tolist()
            hit_members = [idx for idx in assigned if idx in ep["event_by_def"]]
            times = [
                float(ep["event_by_def"][idx]["time"])
                for idx in hit_members
            ]
            spread = max(times) - min(times) if len(times) >= 2 else 0.0
            group_spreads.append(spread)
            complete = len(hit_members) == len(assigned)
            covered = len(hit_members) > 0
            complete_groups += int(complete)
            covered_groups += int(covered)
            group_rows.append(
                {
                    "case": case,
                    "seed": seed,
                    "target_id": target_id,
                    "assigned_defender_count": len(assigned),
                    "hit_defender_count": len(hit_members),
                    "target_intercepted": covered,
                    "assigned_group_complete": complete,
                    "observed_hit_time_spread_s": spread,
                    "within_sync_tolerance_among_hits": spread <= SYNC_TOL_S,
                    "assigned_defender_ids": ";".join(map(str, assigned)),
                    "hit_defender_ids": ";".join(map(str, hit_members)),
                }
            )

        episode_rows.append(
            {
                "case": case,
                "seed": seed,
                "defender_hit_count": len(ep["events"]),
                "defender_miss_count": 1,
                "target_coverage_count": covered_groups,
                "assigned_group_complete_count": complete_groups,
                "missed_defender_id": missed_id,
                "missed_defender_target_id": int(assignment[missed_id]),
                "missed_min_distance_m": miss_min,
                "miss_margin_above_hit_radius_m": miss_min - HIT_RADIUS_M,
                "miss_closest_approach_time_s": float(ep["time"][miss_step]),
                "hit_radius_m": HIT_RADIUS_M,
                "sync_tolerance_s": SYNC_TOL_S,
                "max_observed_hit_group_spread_s": max(group_spreads),
                "mean_observed_hit_group_spread_s": float(np.mean(group_spreads)),
                "episode_horizon_s": float(ep["time"][-1]),
                "all_attackers_intercepted": covered_groups == 8,
                "strict_all_assigned_interceptors_hit": False,
                "frozen_policy": True,
                "training_performed": False,
            }
        )

        for defender_id in range(20):
            target_id = int(assignment[defender_id])
            hit = defender_id in ep["event_by_def"]
            dist = ep["distance"][:, defender_id]
            min_step = int(np.argmin(dist))
            event = ep["event_by_def"].get(defender_id)
            defender_rows.append(
                {
                    "case": case,
                    "seed": seed,
                    "defender_id": defender_id,
                    "assigned_target_id": target_id,
                    "hit": hit,
                    "hit_time_s": float(event["time"]) if event else "",
                    "recorded_hit_distance_m": (
                        float(event["dist_to_target"]) if event else ""
                    ),
                    "trajectory_min_distance_m": float(dist[min_step]),
                    "trajectory_min_distance_time_s": float(ep["time"][min_step]),
                    "min_distance_margin_from_hit_radius_m": (
                        float(dist[min_step]) - HIT_RADIUS_M
                    ),
                }
            )

    return episode_rows, defender_rows, group_rows


def write_trajectory_tables(
    episodes: list[dict[str, Any]], assignment: np.ndarray
) -> None:
    attacker_path = OUTPUT / "attacker_trajectories_long.csv"
    defender_path = OUTPUT / "defender_trajectories_long.csv"
    with attacker_path.open("w", newline="", encoding="utf-8") as ah, \
            defender_path.open("w", newline="", encoding="utf-8") as dh:
        aw = csv.writer(ah)
        dw = csv.writer(dh)
        aw.writerow(["case", "seed", "step", "time_s", "attacker_id",
                     "x_m", "y_m", "z_m"])
        dw.writerow(
            [
                "case", "seed", "step", "time_s", "defender_id",
                "assigned_target_id", "hit", "x_m", "y_m", "z_m",
                "control_0", "control_1", "control_2", "tgo_s",
                "distance_to_assigned_target_m",
            ]
        )
        for ep in episodes:
            hit_ids = set(ep["event_by_def"])
            for step, time_s in enumerate(ep["time"]):
                for attacker_id in range(8):
                    p = ep["att"][step, attacker_id]
                    aw.writerow(
                        [ep["case"], ep["seed"], step, time_s, attacker_id,
                         p[0], p[1], p[2]]
                    )
                for defender_id in range(20):
                    p = ep["def"][step, defender_id]
                    c = ep["ctrl"][step, defender_id]
                    dw.writerow(
                        [
                            ep["case"], ep["seed"], step, time_s, defender_id,
                            int(assignment[defender_id]),
                            defender_id in hit_ids, p[0], p[1], p[2],
                            c[0], c[1], c[2],
                            ep["tgo"][step, defender_id],
                            ep["distance"][step, defender_id],
                        ]
                    )


def equalize_3d(ax, xyz: np.ndarray) -> None:
    mins = np.nanmin(xyz, axis=0)
    maxs = np.nanmax(xyz, axis=0)
    centers = (mins + maxs) / 2.0
    radius = max(maxs - mins) * 0.52
    if not math.isfinite(radius) or radius <= 0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    # Preserve useful altitude resolution instead of forcing a cubic z range.
    zpad = max((maxs[2] - mins[2]) * 0.08, 5.0)
    ax.set_zlim(mins[2] - zpad, maxs[2] + zpad)
    ax.set_box_aspect((1.0, 1.0, 0.54))


def plot_episode_3d(
    ax, ep: dict[str, Any], assignment: np.ndarray, title: bool = True
) -> None:
    paths = [ep["att"].reshape(-1, 3), ep["def"].reshape(-1, 3)]
    for target_index in range(8):
        p = ep["att"][:, target_index, :]
        ax.plot(
            p[:, 0], p[:, 1], p[:, 2],
            color=COLORS[target_index], linestyle="-", linewidth=1.35,
            alpha=0.9,
        )
        ax.scatter(
            p[0, 0], p[0, 1], p[0, 2],
            marker="^", s=11, color=COLORS[target_index],
            edgecolors="none", depthshade=False,
        )

    for defender_id in range(20):
        target_index = int(assignment[defender_id] - 20)
        p = ep["def"][:, defender_id, :]
        event = ep["event_by_def"].get(defender_id)
        stop = int(event["step"]) + 1 if event else len(p)
        if defender_id == ep["missed_id"]:
            ax.plot(
                p[:stop, 0], p[:stop, 1], p[:stop, 2],
                color=MISS_COLOR, linestyle="--", linewidth=2.0, alpha=1.0,
                zorder=6,
            )
        else:
            ax.plot(
                p[:stop, 0], p[:stop, 1], p[:stop, 2],
                color=COLORS[target_index], linestyle=":", linewidth=0.65,
                alpha=0.52,
            )

    event_pos = np.array(
        [
            [
                float(row["defender_pos_x"]),
                float(row["defender_pos_y"]),
                float(row["defender_pos_z"]),
            ]
            for row in ep["events"]
        ]
    )
    ax.scatter(
        event_pos[:, 0], event_pos[:, 1], event_pos[:, 2],
        marker="o", s=8, c=HIT_COLOR, edgecolors="white", linewidths=0.25,
        depthshade=False, zorder=7,
    )
    miss_id = ep["missed_id"]
    miss_step = int(np.argmin(ep["distance"][:, miss_id]))
    miss_p = ep["def"][miss_step, miss_id]
    ax.scatter(
        miss_p[0], miss_p[1], miss_p[2],
        marker="X", s=25, c=MISS_COLOR, edgecolors="black", linewidths=0.35,
        depthshade=False, zorder=9,
    )
    equalize_3d(ax, np.concatenate(paths, axis=0))
    ax.view_init(elev=24, azim=-57)
    ax.set_xlabel(r"$x$ (m)", labelpad=0)
    ax.set_ylabel(r"$y$ (m)", labelpad=0)
    ax.set_zlabel(r"$z$ (m)", labelpad=0)
    ax.tick_params(pad=0.5)
    ax.grid(False)
    if title:
        ax.set_title(
            f"{ep['case'].capitalize()}, seed {ep['seed']}",
            pad=1.5,
        )


def save_three_formats(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def overview_figure(
    episodes: list[dict[str, Any]], assignment: np.ndarray
) -> None:
    fig = plt.figure(figsize=(7.16, 4.72))
    letters = "abcdef"
    for idx, ep in enumerate(episodes):
        ax = fig.add_subplot(2, 3, idx + 1, projection="3d")
        plot_episode_3d(ax, ep, assignment)
        ax.text2D(
            0.01, 0.96, f"({letters[idx]})",
            transform=ax.transAxes, fontsize=8.5, fontweight="bold",
        )
    legend = [
        Line2D([0], [0], color="#0072B2", lw=1.4,
               label="Attacker trajectory (target color)"),
        Line2D([0], [0], color=GRAY, lw=0.8, ls=":",
               label="Successful interceptor"),
        Line2D([0], [0], color=MISS_COLOR, lw=2.0, ls="--",
               label="Missed interceptor"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=HIT_COLOR,
               markeredgecolor="white", markersize=4.0, label="Hit event"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=MISS_COLOR,
               markeredgecolor="black", markeredgewidth=0.35, markersize=5.0,
               label="Closest approach of missed interceptor"),
    ]
    fig.legend(
        handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.002),
        ncol=5, frameon=False, handlelength=1.9, columnspacing=0.75,
    )
    fig.subplots_adjust(
        left=0.015, right=0.995, bottom=0.075, top=0.98,
        wspace=0.06, hspace=0.12,
    )
    save_three_formats(fig, OUTPUT / "failure_cases_case1_case2_v10")


def case_overview_figure(
    case: str, episodes: list[dict[str, Any]], assignment: np.ndarray
) -> None:
    fig = plt.figure(figsize=(7.16, 2.48))
    for idx, ep in enumerate(episodes):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        plot_episode_3d(ax, ep, assignment)
        ax.text2D(
            0.01, 0.96, f"({chr(ord('a') + idx)})",
            transform=ax.transAxes, fontsize=8.5, fontweight="bold",
        )
    legend = [
        Line2D([0], [0], color=GRAY, lw=0.8, ls=":",
               label="Successful interceptor"),
        Line2D([0], [0], color=MISS_COLOR, lw=2.0, ls="--",
               label="Missed interceptor"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=HIT_COLOR,
               markersize=4, label="Hit event"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=MISS_COLOR,
               markeredgecolor="black", markersize=5, label="Closest approach"),
    ]
    fig.legend(
        handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.005),
        ncol=4, frameon=False, columnspacing=1.0, handlelength=2.0,
    )
    fig.subplots_adjust(
        left=0.01, right=0.995, bottom=0.14, top=0.98, wspace=0.04
    )
    save_three_formats(fig, OUTPUT / f"{case}_three_failure_cases_v10")


def diagnostic_figure(ep: dict[str, Any], assignment: np.ndarray) -> None:
    fig = plt.figure(figsize=(7.16, 2.62))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    plot_episode_3d(ax3, ep, assignment, title=False)
    ax3.set_title(
        f"{ep['case'].capitalize()}, seed {ep['seed']}",
        pad=1.5,
    )
    ax3.text2D(
        0.01, 0.96, "(a)", transform=ax3.transAxes,
        fontsize=8.5, fontweight="bold",
    )

    ax = fig.add_subplot(1, 2, 2)
    for defender_id in range(20):
        distance = ep["distance"][:, defender_id]
        event = ep["event_by_def"].get(defender_id)
        stop = int(event["step"]) + 1 if event else len(distance)
        if defender_id == ep["missed_id"]:
            ax.plot(
                ep["time"][:stop], distance[:stop],
                color=MISS_COLOR, ls="--", lw=1.8,
                label=f"Missed interceptor {defender_id}",
                zorder=5,
            )
            k = int(np.argmin(distance))
            ax.plot(
                ep["time"][k], distance[k], marker="X",
                color=MISS_COLOR, markeredgecolor="black",
                markeredgewidth=0.35, ms=5.5, zorder=7,
            )
        else:
            ax.plot(
                ep["time"][:stop], distance[:stop],
                color=GRAY, lw=0.52, alpha=0.52,
            )
    ax.axhline(
        HIT_RADIUS_M, color="#000000", lw=1.0, ls="-.",
        label=r"Hit radius, $d_{\mathrm{hit}}=3$ m",
    )
    miss_id = ep["missed_id"]
    miss_min = float(np.min(ep["distance"][:, miss_id]))
    k_min = int(np.argmin(ep["distance"][:, miss_id]))
    t_min = float(ep["time"][k_min])
    zoom = (ep["time"] >= t_min - 3.0) & (ep["time"] <= t_min + 3.0)
    red_zoom = ep["distance"][zoom, miss_id]
    ymax = float(np.nanpercentile(red_zoom, 88))
    ax.set_ylim(0, max(12.0, min(ymax, 50.0)))
    ax.set_xlim(max(0.0, t_min - 3.0), min(ep["time"][-1], t_min + 3.0))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance to assigned target (m)")
    ax.set_title(
        f"Closest approach of interceptor {miss_id}: {miss_min:.2f} m",
        pad=2.0,
    )
    ax.tick_params(which="both", top=True, right=True)
    ax.legend(loc="upper right", frameon=False, handlelength=2.5)
    ax.text(
        -0.10, 1.01, "(b)", transform=ax.transAxes,
        fontsize=8.5, fontweight="bold",
    )
    fig.subplots_adjust(
        left=0.04, right=0.99, bottom=0.18, top=0.87, wspace=0.17
    )
    save_three_formats(
        fig, OUTPUT / f"{ep['case']}_seed_{ep['seed']}_diagnostic_v10"
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    set_v10_style()
    preset = np.load(PRESET)
    assignment = np.asarray(preset["assignment"], dtype=int)
    if assignment.shape != (20,) or assignment.min() != 20 or assignment.max() != 27:
        raise ValueError(f"unexpected assignment: {assignment}")

    episodes = [
        load_episode(case, seed, assignment)
        for case, seeds in SELECTED.items()
        for seed in seeds
    ]
    episode_rows, defender_rows, group_rows = derive_tables(episodes, assignment)
    write_csv(OUTPUT / "failure_case_metrics.csv", episode_rows)
    write_csv(OUTPUT / "defender_metrics.csv", defender_rows)
    write_csv(OUTPUT / "target_group_metrics.csv", group_rows)
    write_trajectory_tables(episodes, assignment)

    overview_figure(episodes, assignment)
    for case in SELECTED:
        case_overview_figure(
            case, [ep for ep in episodes if ep["case"] == case], assignment
        )
    for ep in episodes:
        diagnostic_figure(ep, assignment)

    validation = {
        "episode_count": len(episodes),
        "case_counts": {
            case: sum(ep["case"] == case for ep in episodes)
            for case in SELECTED
        },
        "all_have_19_of_20_interceptor_hits": all(
            row["defender_hit_count"] == 19 for row in episode_rows
        ),
        "all_have_8_of_8_target_coverage": all(
            row["target_coverage_count"] == 8 for row in episode_rows
        ),
        "all_have_one_incomplete_assignment_group": all(
            row["assigned_group_complete_count"] == 7 for row in episode_rows
        ),
        "all_missed_interceptors_remain_outside_hit_radius": all(
            row["missed_min_distance_m"] > HIT_RADIUS_M for row in episode_rows
        ),
        "all_arrays_finite": True,
        "hit_radius_m": HIT_RADIUS_M,
        "sync_tolerance_s": SYNC_TOL_S,
        "fixed_assignment": assignment.tolist(),
        "policy_training_performed": False,
        "optimizer_steps": 0,
        "selection_rule": (
            "earliest three seeds per case satisfying 8/8 target coverage, "
            "exactly 19/20 interceptor hits, seven complete assignment groups, "
            "and observed hit-group spread no greater than 0.5 s"
        ),
        "input_npz_files": [str(ep["npz_path"]) for ep in episodes],
    }
    (OUTPUT / "analysis_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    (OUTPUT / "README_data.md").write_text(
        """# Failure-case analysis data

All tables and figures in this directory are derived directly from six native
evaluation NPZ files and their hit-event CSV files.  No trajectory value was
edited or synthesized.

- `failure_case_metrics.csv`: one row per selected episode.
- `defender_metrics.csv`: one row per interceptor and episode.
- `target_group_metrics.csv`: one row per target-assignment group and episode.
- `attacker_trajectories_long.csv`: long-form attacker positions.
- `defender_trajectories_long.csv`: long-form interceptor positions, controls,
  time-to-go estimates, and distance to the assigned target.
- `*_v10.pdf`, `*_v10.svg`, `*_v10.png`: vector and 600-dpi plots.

The native NPZ files remain the authoritative raw data and are copied together
with this analysis package.
""",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2))
    print(json.dumps(episode_rows, indent=2))


if __name__ == "__main__":
    main()
