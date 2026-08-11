#!/usr/bin/env python3
"""Reclassify locally saved MC episodes using protected-asset deadlines.

The previous evaluator did not stop an episode when an attacker reached the
protected asset.  Full time histories were not saved for every MC episode, but
the attacker dynamics, fixed paper preset, and per-interceptor hit-event table
are available locally.  Because the scripted attacker motion is independent of
the observation noise and failed-defender draw until the attacker is stopped,
this script first replays each attacker without defenders to obtain its own
3-m asset-entry deadline.  It then checks the saved hit times target by target.

The plotted Case 1 population contains all asset-safe ISR-successful episodes.
To retain the user's preceding Case 2 synchronization requirement, the plotted
Case 2 population additionally requires the recorded cooperative-success flag.
No selection condition is printed inside the figure; all rules and exclusions
are exported to CSV/JSON alongside it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("E_co_time_s", r"$E_{\mathrm{co\!-\!time}}$ (s)"),
    ("E_n_g", r"$E_n$ (g)"),
    ("E_miss_m", r"$E_{miss}$ (m)"),
    ("E_t_s", r"$E_t$ (s)"),
]
COLORS = ["#0072B2", "#D55E00"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_environment_args(case: str, preset: Path) -> SimpleNamespace:
    case_cfg = {
        "case1": {
            "guidance_base_gain": 2.0,
            "guidance_tau": 0.25,
            "guidance_lead": 1.60,
            "sync_speed_gain": 0.14,
            "speed_gain": 0.016,
            "command_lag_tau": 0.25,
        },
        "case2": {
            "guidance_base_gain": 2.6,
            "guidance_tau": 0.35,
            "guidance_lead": 1.70,
            "sync_speed_gain": 1.40,
            "speed_gain": 0.008,
            "command_lag_tau": 0.40,
        },
    }[case]
    return SimpleNamespace(
        case_3d=case,
        target_assignment_mode="fixed",
        target_assignment_spread_weight=6.0,
        hit_radius_3d=3.0,
        attack_maneuver_gain=1.20,
        attack_maneuver_offset_gain=1.25,
        attack_maneuver_freq=0.17,
        attack_maneuver_fade_range=450.0,
        case1_lateral_base=0.95,
        case1_lateral_tail=0.40,
        case1_vertical_amp=0.35,
        case2_lateral_amp=1.00,
        case2_maneuver_freq=2.0 * np.pi / 50.0,
        case2_vertical_amp=0.25,
        case2_vertical_freq_scale=0.50,
        paper_preset_path=str(preset),
        paper_attacker_replay=1,
        paper_altitude=120.0,
        paper_altitude_step=0.0,
        paper_defender_climb_to_target=0,
        no_tailchase_gate=0.0,
        no_tailchase_rebound=5.0,
        no_tailchase_penalty=0.0,
        no_tailchase_terminate=False,
        attacker_speed_min=12.0,
        attacker_speed_max=65.0,
        attacker_axial_min=-4.0,
        attacker_axial_max=4.0,
        attacker_load_limit=1.75,
        attacker_yaw_scale=1.55,
        attacker_pitch_scale=1.55,
        defender_guidance_base_gain=case_cfg["guidance_base_gain"],
        defender_guidance_tau=case_cfg["guidance_tau"],
        defender_guidance_lead=case_cfg["guidance_lead"],
        defender_residual_scale=0.20,
        defender_load_limit=1.0,
        defender_axial_min=-0.1,
        defender_axial_max=1.0,
        defender_sync_speed_gain=case_cfg["sync_speed_gain"],
        defender_sync_tgo_ref="min",
        defender_speed_target=40.0,
        defender_speed_gain=case_cfg["speed_gain"],
        defender_min_accel_load=0.0,
        defender_speed_min=12.0,
        defender_speed_max=40.0,
        defender_sensor_delay_steps=1,
        defender_sensor_delay_compensate=False,
        defender_obs_pos_noise_std=3.0,
        defender_obs_vel_noise_std=0.3,
        defender_obs_filter_alpha=1.0,
        defender_command_lag_tau=case_cfg["command_lag_tau"],
        reward_w_smooth=0.0,
        reference_control_root="",
        reward_w_ref_control=0.0,
        reward_w_ref_rate=0.0,
        defender_reference_blend=0.0,
    )


def replay_asset_deadlines(
    case: str, code_root: Path, preset: Path, max_steps: int = 1500
) -> tuple[list[dict], dict[int, float]]:
    """Replay the exact scripted attackers and return first 3-m entry times."""
    code_root = code_root.resolve()
    sys.path.insert(0, str(code_root))
    from onpolicy.envs.mpe.scenarios.simple_world_comm_3d import Scenario

    scenario = Scenario()
    world = scenario.make_world(make_environment_args(case, preset.resolve()))
    defenders = [agent for agent in world.agents if agent.adversary]
    attackers = [agent for agent in world.agents if not agent.adversary]
    for defender in defenders:
        defender.doneflag = True
        defender.action.u = np.zeros(3, dtype=float)

    first_entry = np.full(len(attackers), np.nan, dtype=float)
    minimum_distance = np.full(len(attackers), np.inf, dtype=float)
    minimum_time = np.full(len(attackers), np.nan, dtype=float)
    for step in range(max_steps):
        for attacker in attackers:
            attacker.action = attacker.action_callback(attacker, world)
        world.step()
        distances = np.asarray(
            [
                np.linalg.norm(
                    attacker.state.p_pos - world.landmarks[0].state.p_pos
                )
                for attacker in attackers
            ],
            dtype=float,
        )
        time_s = (step + 1) * float(world.dt)
        newly_entered = np.isnan(first_entry) & (distances <= 3.0)
        first_entry[newly_entered] = time_s
        improved = distances < minimum_distance
        minimum_distance[improved] = distances[improved]
        minimum_time[improved] = time_s

    rows = []
    deadlines: dict[int, float] = {}
    for target_id in range(len(attackers)):
        deadline = float(first_entry[target_id])
        deadlines[target_id] = deadline if np.isfinite(deadline) else float("inf")
        rows.append(
            {
                "case": case,
                "target_index": target_id,
                "asset_entry_time_s": deadline,
                "minimum_asset_distance_m_over_75s": float(
                    minimum_distance[target_id]
                ),
                "minimum_distance_time_s": float(minimum_time[target_id]),
                "asset_center_m": "0;0;0",
                "asset_hit_radius_m": 3.0,
            }
        )
    return rows, deadlines


def audit_case(
    case: str,
    episode_rows: list[dict[str, str]],
    target_rows: list[dict[str, str]],
    hit_rows: list[dict[str, str]],
    deadlines: dict[int, float],
) -> list[dict]:
    targets_by_episode = {
        (int(row["episode"]), int(row["target_index"])): row
        for row in target_rows
    }
    hits_by_episode_target: dict[tuple[int, int], list[float]] = {}
    for row in hit_rows:
        key = (int(row["episode"]), int(row["target_index"]))
        hits_by_episode_target.setdefault(key, []).append(float(row["hit_time_s"]))

    audit = []
    for episode_row in episode_rows:
        episode = int(episode_row["episode"])
        coverage_safe = True
        active_group_safe = True
        breach_targets: list[int] = []
        late_active_group_targets: list[int] = []
        first_hit_times: list[float] = []
        last_active_hit_times: list[float] = []
        for target_id in range(8):
            times = sorted(hits_by_episode_target.get((episode, target_id), []))
            deadline = deadlines[target_id]
            target_row = targets_by_episode[(episode, target_id)]
            active_defenders = int(target_row["active_defenders"])
            first_safe = bool(times) and (
                not np.isfinite(deadline) or times[0] < deadline
            )
            if not first_safe:
                coverage_safe = False
                breach_targets.append(target_id)
            else:
                first_hit_times.append(times[0])

            group_safe = bool(times) and len(times) == active_defenders and (
                not np.isfinite(deadline) or times[-1] < deadline
            )
            if not group_safe:
                active_group_safe = False
                late_active_group_targets.append(target_id)
            else:
                last_active_hit_times.append(times[-1])

        original_isr = bool(int(episode_row["interception_success"]))
        original_coop = bool(int(episode_row["cooperative_success"]))
        audit.append(
            {
                "case": case,
                "episode": episode,
                "seed": int(episode_row["seed"]),
                "original_interception_success": int(original_isr),
                "original_cooperative_success": int(original_coop),
                "asset_safe_interception_success": int(
                    original_isr and coverage_safe
                ),
                "asset_safe_active_group_success": int(
                    active_group_safe
                ),
                "asset_safe_cooperative_success": int(
                    original_coop and active_group_safe
                ),
                "breach_before_first_hit_target_indices": ";".join(
                    map(str, breach_targets)
                ),
                "not_completed_before_breach_target_indices": ";".join(
                    map(str, late_active_group_targets)
                ),
                "mission_first_hit_completion_time_s": (
                    max(first_hit_times) if len(first_hit_times) == 8 else np.nan
                ),
                "mission_active_group_completion_time_s": (
                    max(last_active_hit_times)
                    if len(last_active_hit_times) == 8
                    else np.nan
                ),
            }
        )
    return audit


def summarize(rows: list[dict[str, str]], case: str, rule: str) -> list[dict]:
    output = []
    for key, _ in METRICS:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        output.append(
            {
                "case": case,
                "selection_rule": rule,
                "n": int(values.size),
                "metric": key,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "median": float(np.median(values)),
                "q1": float(np.quantile(values, 0.25)),
                "q3": float(np.quantile(values, 0.75)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case1-dir", type=Path, required=True)
    parser.add_argument("--case2-dir", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    all_deadline_rows: list[dict] = []
    deadline_by_case: dict[str, dict[int, float]] = {}
    for case in ("case1", "case2"):
        rows, deadlines = replay_asset_deadlines(
            case, args.code_root, args.preset
        )
        all_deadline_rows.extend(rows)
        deadline_by_case[case] = deadlines
    write_rows(args.outdir / "attacker_asset_entry_deadlines.csv", all_deadline_rows)

    data_dirs = {"case1": args.case1_dir, "case2": args.case2_dir}
    episode_tables: dict[str, list[dict[str, str]]] = {}
    audit_tables: dict[str, list[dict]] = {}
    for case, directory in data_dirs.items():
        episodes = read_rows(directory / "episodes.csv")
        targets = read_rows(directory / "targets.csv")
        hits = read_rows(directory / "hit_events.csv")
        episode_tables[case] = episodes
        audit_tables[case] = audit_case(
            case, episodes, targets, hits, deadline_by_case[case]
        )
    audit_rows = audit_tables["case1"] + audit_tables["case2"]
    write_rows(args.outdir / "episode_asset_safety_audit.csv", audit_rows)

    audit_by_key = {
        (row["case"], int(row["episode"])): row for row in audit_rows
    }
    case1_selected = [
        row
        for row in episode_tables["case1"]
        if audit_by_key[("case1", int(row["episode"]))][
            "asset_safe_interception_success"
        ]
    ]
    case2_selected = [
        row
        for row in episode_tables["case2"]
        if audit_by_key[("case2", int(row["episode"]))][
            "asset_safe_cooperative_success"
        ]
    ]
    if not case1_selected or not case2_selected:
        raise RuntimeError("one of the asset-safe selected populations is empty")

    write_rows(args.outdir / "case1_asset_safe_success_subset.csv", case1_selected)
    write_rows(
        args.outdir / "case2_asset_safe_synchronized_success_subset.csv",
        case2_selected,
    )
    stats = summarize(
        case1_selected,
        "case1",
        "asset_safe_interception_success == 1",
    ) + summarize(
        case2_selected,
        "case2",
        "asset_safe_cooperative_success == 1",
    )
    write_rows(args.outdir / "asset_safe_subset_statistics.csv", stats)

    excluded_case2 = []
    for row in episode_tables["case2"]:
        audit = audit_by_key[("case2", int(row["episode"]))]
        if int(row["cooperative_success"]) and not int(
            audit["asset_safe_cooperative_success"]
        ):
            excluded_case2.append({**row, **audit})
    write_rows(
        args.outdir / "case2_synchronized_episodes_excluded_by_asset_rule.csv",
        excluded_case2,
    )

    case_counts = {}
    for case in ("case1", "case2"):
        audits = audit_tables[case]
        case_counts[case] = {
            "input_episodes": len(audits),
            "original_interception_success_count": sum(
                int(row["original_interception_success"]) for row in audits
            ),
            "asset_safe_interception_success_count": sum(
                int(row["asset_safe_interception_success"]) for row in audits
            ),
            "original_cooperative_success_count": sum(
                int(row["original_cooperative_success"]) for row in audits
            ),
            "asset_safe_cooperative_success_count": sum(
                int(row["asset_safe_cooperative_success"]) for row in audits
            ),
        }

    manifest = {
        "purpose": (
            "Offline reclassification of locally saved MC event data under "
            "the rule that a non-intercepted attacker entering the 3-m "
            "protected-asset radius terminates the episode as a failure."
        ),
        "asset_center_m": [0.0, 0.0, 0.0],
        "asset_hit_radius_m": 3.0,
        "attacker_deadline_method": (
            "Exact local replay of the stable scripted-attacker dynamics and "
            "paper preset without defenders; evaluated separately for each "
            "attacker/target."
        ),
        "case1_figure_selection": "asset_safe_interception_success == 1",
        "case2_figure_selection": "asset_safe_cooperative_success == 1",
        "case2_extra_condition_reason": (
            "Retains the user's preceding request to show only the strongly "
            "synchronized Case 2 trials."
        ),
        "metric_values": (
            "The four stored episode metrics are unchanged; only the episode "
            "population is reclassified."
        ),
        "figure_labels": "Case 1 and Case 2 only, as requested",
        "case_counts": case_counts,
        "limitations": (
            "This is an event-table reclassification, not a rerun with an "
            "in-loop asset-breach terminal condition. Full per-step histories "
            "were saved only for one representative episode per case."
        ),
    }
    (args.outdir / "asset_safe_subset_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=True), encoding="utf-8"
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.25))
    for panel, (axis, (key, ylabel)) in enumerate(zip(axes.flat, METRICS)):
        values = [
            np.asarray([float(row[key]) for row in case1_selected]),
            np.asarray([float(row[key]) for row in case2_selected]),
        ]
        boxes = axis.boxplot(
            values,
            labels=["Case 1", "Case 2"],
            widths=0.48,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 3.2,
            },
            medianprops={"color": "black", "linewidth": 1.15},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
            flierprops={
                "marker": "o",
                "markersize": 2.2,
                "markerfacecolor": "none",
                "markeredgecolor": "#666666",
                "alpha": 0.55,
            },
        )
        for patch, color in zip(boxes["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
            patch.set_alpha(0.65)
            patch.set_linewidth(0.8)
        axis.set_ylabel(ylabel)
        axis.grid(
            True,
            axis="y",
            linestyle="--",
            linewidth=0.35,
            color="#B8B8B8",
            alpha=0.65,
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(
            -0.17,
            1.03,
            f"({chr(97 + panel)})",
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
        )
    fig.subplots_adjust(
        left=0.075, right=0.995, bottom=0.22, top=0.94, wspace=0.52
    )
    stem = args.outdir / "two_defender_failure_mc_boxplots_asset_safe_success"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
