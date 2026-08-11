#!/usr/bin/env python3
"""Quantify and plot the mechanisms behind two representative boundary failures.

All quantities are computed from evaluator-native trajectories.  The script is
intended to run on the remote server where the evaluation data were generated.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path("/home/a2rl/reviewer_failure_cases_20260730")
CF_ROOT = ROOT / "results/failure_cause_counterfactuals_v2"
OUT = ROOT / "results/failure_mechanism_analysis_v2"
DT = 0.05
HORIZON_S = 75.0
HIT_RADIUS_M = 3.0
CASES = {"case1": 76048, "case2": 77008}
MISSED_DEFENDER = {"case1": 5, "case2": 1}
ASSIGNMENT = np.array(
    [20, 21, 22, 23, 24, 25, 26, 27, 20, 21,
     22, 23, 24, 25, 26, 27, 20, 21, 22, 23],
    dtype=int,
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
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
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "text.usetex": False,
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def vec_norm(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=-1)


def scalar_at(values: np.ndarray, step: int) -> float:
    return float(values[int(np.clip(step, 0, len(values) - 1))])


def load_observed(case: str) -> dict:
    folder = CF_ROOT / case / "observed_boundary" / case
    z = np.load(folder / f"{case}_selected_episode.npz")
    events = read_csv(folder / f"{case}_hit_events.csv")
    data = {
        "defender": np.asarray(z["rep_def"], dtype=float),
        "attacker": np.asarray(z["rep_att"], dtype=float),
        "control": np.asarray(z["rep_ctrl"], dtype=float),
        "tgo": np.asarray(z["rep_tgo"], dtype=float),
        "events": events,
    }
    for key in ("defender", "attacker", "control", "tgo"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"NaN/Inf in {case} {key}")
    return data


def kinematics(case: str, data: dict) -> dict[str, np.ndarray]:
    missed_defender = MISSED_DEFENDER[case]
    target_index = int(ASSIGNMENT[missed_defender] - 20)
    p_d = data["defender"][:, missed_defender]
    p_a = data["attacker"][:, target_index]
    v_d = np.gradient(p_d, DT, axis=0)
    v_a = np.gradient(p_a, DT, axis=0)
    a_a = np.gradient(v_a, DT, axis=0)
    rel = p_a - p_d
    rel_vel = v_a - v_d
    distance = vec_norm(rel)
    safe_d = np.maximum(distance, 1e-12)
    range_rate = np.einsum("ij,ij->i", rel, rel_vel) / safe_d
    closing_speed = -range_rate
    los_rate = vec_norm(np.cross(rel, rel_vel)) / (safe_d ** 2)
    defender_speed = vec_norm(v_d)
    cos_heading = np.einsum("ij,ij->i", v_d, rel) / (
        np.maximum(defender_speed, 1e-12) * safe_d
    )
    heading_error = np.arccos(np.clip(cos_heading, -1.0, 1.0))
    control = data["control"][:, missed_defender]
    control_norm = vec_norm(control)
    lateral_control = np.hypot(control[:, 1], control[:, 2])
    return {
        "distance": distance,
        "range_rate": range_rate,
        "closing_speed": closing_speed,
        "los_rate": los_rate,
        "defender_speed": defender_speed,
        "heading_error": heading_error,
        "target_acceleration": vec_norm(a_a),
        "control_norm": control_norm,
        "lateral_control": lateral_control,
        "nx": control[:, 0],
        "ny": control[:, 1],
        "nz": control[:, 2],
        "tgo": data["tgo"][:, missed_defender],
    }


def zero_crossing_near(values: np.ndarray, center: int, half_window: int) -> int:
    lo = max(center - half_window, 0)
    hi = min(center + half_window + 1, len(values))
    candidates = []
    for idx in range(lo, hi - 1):
        if values[idx] <= 0.0 < values[idx + 1]:
            candidates.append(idx + 1)
    if not candidates:
        return center
    return min(candidates, key=lambda idx: abs(idx - center))


def analyze_case(case: str, data: dict) -> tuple[dict, list[dict], list[dict], dict]:
    missed_defender = MISSED_DEFENDER[case]
    kin = kinematics(case, data)
    distance = kin["distance"]
    closest = int(np.argmin(distance))
    crossing = zero_crossing_near(kin["range_rate"], closest, int(2.0 / DT))
    time = np.arange(len(distance)) * DT
    target_id = int(ASSIGNMENT[missed_defender])
    group = np.flatnonzero(ASSIGNMENT == target_id).tolist()
    event_by_def = {int(row["defender_id"]): row for row in data["events"]}
    peer_hit_times = [
        float(event_by_def[defender_id]["time"])
        for defender_id in group
        if defender_id in event_by_def
    ]
    first_peer_hit = min(peer_hit_times)
    last_peer_hit = max(peer_hit_times)

    terminal_mask = (
        (time >= time[closest] - 1.0) & (time <= time[closest] + 1.0)
    )
    approach_mask = (
        (time >= time[closest] - 2.0) & (time <= time[closest])
    )
    summary = {
        "case": case,
        "seed": CASES[case],
        "missed_defender_internal_id": missed_defender,
        "missed_defender_paper_label": f"D-UAV {missed_defender + 1}",
        "assigned_target_internal_id": target_id,
        "assigned_target_paper_label": f"A{target_id - 19}",
        "closest_time_s": float(time[closest]),
        "minimum_distance_m": float(distance[closest]),
        "kill_radius_m": HIT_RADIUS_M,
        "miss_margin_m": float(distance[closest] - HIT_RADIUS_M),
        "range_rate_sign_change_time_s": float(time[crossing]),
        "sign_change_minus_closest_s": float(time[crossing] - time[closest]),
        "closing_speed_1s_before_mps": scalar_at(
            kin["closing_speed"], closest - round(1.0 / DT)
        ),
        "closing_speed_0p5s_before_mps": scalar_at(
            kin["closing_speed"], closest - round(0.5 / DT)
        ),
        "range_rate_0p5s_after_mps": scalar_at(
            kin["range_rate"], closest + round(0.5 / DT)
        ),
        "los_rate_at_closest_radps": scalar_at(kin["los_rate"], closest),
        "max_los_rate_terminal_2s_radps": float(
            np.max(kin["los_rate"][terminal_mask])
        ),
        "heading_error_at_closest_deg": float(
            np.degrees(kin["heading_error"][closest])
        ),
        "tgo_at_closest_s": scalar_at(kin["tgo"], closest),
        "lateral_control_at_closest_g": scalar_at(
            kin["lateral_control"], closest
        ),
        "max_lateral_control_last_2s_g": float(
            np.max(kin["lateral_control"][approach_mask])
        ),
        "control_norm_at_closest_g": scalar_at(kin["control_norm"], closest),
        "max_control_norm_last_2s_g": float(
            np.max(kin["control_norm"][approach_mask])
        ),
        "fraction_abs_ny_ge_0p95_last_2s": float(
            np.mean(np.abs(kin["ny"][approach_mask]) >= 0.95)
        ),
        "fraction_abs_nz_ge_0p95_last_2s": float(
            np.mean(np.abs(kin["nz"][approach_mask]) >= 0.95)
        ),
        "max_target_acceleration_last_2s_mps2": float(
            np.max(kin["target_acceleration"][approach_mask])
        ),
        "first_peer_hit_time_s": first_peer_hit,
        "last_peer_hit_time_s": last_peer_hit,
        "closest_minus_first_peer_hit_s": float(time[closest] - first_peer_hit),
        "group_observed_hit_spread_s": float(last_peer_hit - first_peer_hit),
        "incomplete_group_delay_lower_bound_s": float(
            HORIZON_S - first_peer_hit
        ),
        "defender_hit_count": len(data["events"]),
        "target_coverage_count": len(
            {int(row["target_id"]) for row in data["events"]}
        ),
    }

    window_rows = []
    for offset_s in (-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0):
        step = closest + int(round(offset_s / DT))
        step = int(np.clip(step, 0, len(distance) - 1))
        window_rows.append(
            {
                "case": case,
                "time_relative_to_closest_s": float(time[step] - time[closest]),
                "absolute_time_s": float(time[step]),
                "distance_m": scalar_at(kin["distance"], step),
                "closing_speed_mps": scalar_at(kin["closing_speed"], step),
                "range_rate_mps": scalar_at(kin["range_rate"], step),
                "los_rate_radps": scalar_at(kin["los_rate"], step),
                "heading_error_deg": float(
                    np.degrees(scalar_at(kin["heading_error"], step))
                ),
                "tgo_s": scalar_at(kin["tgo"], step),
                "nx_g": scalar_at(kin["nx"], step),
                "ny_g": scalar_at(kin["ny"], step),
                "nz_g": scalar_at(kin["nz"], step),
                "lateral_control_g": scalar_at(kin["lateral_control"], step),
            }
        )

    peer_rows = []
    for defender_id in group:
        row = event_by_def.get(defender_id)
        peer_rows.append(
            {
                "case": case,
                "target": f"A{target_id - 19}",
                "defender_internal_id": defender_id,
                "defender_paper_label": f"D-UAV {defender_id + 1}",
                "hit": row is not None,
                "hit_time_s": float(row["time"]) if row else "",
                "distance_at_event_m": float(row["dist_to_target"]) if row else "",
                "status": "hit" if row else "right-censored at 75 s",
            }
        )
    plot_data = {"time": time, "closest": closest, **kin}
    return summary, window_rows, peer_rows, plot_data


def plot_diagnostics(plot_data: dict[str, dict], summaries: list[dict]) -> None:
    colors = {"case1": "#0072B2", "case2": "#D55E00"}
    fig, axes = plt.subplots(4, 2, figsize=(7.16, 7.0), sharex="col")
    for col, case in enumerate(("case1", "case2")):
        data = plot_data[case]
        center = data["closest"]
        rel_t = data["time"] - data["time"][center]
        mask = (rel_t >= -0.5) & (rel_t <= 0.5)
        color = colors[case]

        axes[0, col].plot(rel_t[mask], data["distance"][mask], color=color)
        axes[0, col].axhline(
            HIT_RADIUS_M, color="#000000", ls="--", lw=1.0,
            label=r"$d_{\rm kill}=3$ m",
        )
        axes[0, col].scatter(
            [0.0], [data["distance"][center]], color="#D55E00",
            marker="X", s=35, zorder=5, label="Closest approach",
        )
        axes[0, col].set_ylabel(r"$d$ (m)")
        axes[0, col].set_title(f"Case {col + 1}", pad=2)

        axes[1, col].plot(
            rel_t[mask], data["range_rate"][mask], color=color
        )
        axes[1, col].axhline(0.0, color="#000000", ls="--", lw=1.0)
        axes[1, col].set_ylabel(r"$\dot d$ (m/s)")

        axes[2, col].plot(
            rel_t[mask], data["los_rate"][mask], color="#009E73"
        )
        axes[2, col].set_ylabel(r"$\|\dot{\boldsymbol{\lambda}}\|$ (rad/s)")

        axes[2, col].plot(
            [], [], alpha=0.0
        )
        axes[3, col].plot(
            rel_t[mask], np.abs(data["ny"][mask]), color=color,
            label=r"$|n_y|$",
        )
        axes[3, col].plot(
            rel_t[mask], np.abs(data["nz"][mask]), color="#009E73",
            ls="-.", label=r"$|n_z|$",
        )
        axes[3, col].axhline(
            1.0, color="#000000", ls="--", lw=1.0,
            label="Component limit",
        )
        axes[3, col].set_ylabel("Control command (g)")
        axes[3, col].set_xlabel("Time relative to closest approach (s)")

        for row in range(4):
            axes[row, col].axvline(0.0, color="#777777", ls=":", lw=0.8)
            axes[row, col].grid(True)

        axes[0, 0].legend(loc="upper right", fontsize=7)
    axes[3, 0].legend(loc="upper center", ncol=3, fontsize=7)
    labels = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)")
    for index, (label, ax) in enumerate(zip(labels, axes.flat)):
        y_position = 0.08 if index in (0, 6) else 0.96
        ax.text(
            0.02, y_position, label, transform=ax.transAxes,
            ha="left", va="top", fontweight="bold"
        )
    fig.align_ylabels()
    fig.subplots_adjust(hspace=0.12, wspace=0.28)
    for suffix, kwargs in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 600}),
    ):
        fig.savefig(OUT / f"failure_mechanism_diagnostics_v10.{suffix}", **kwargs)
    plt.close(fig)


def build_group_timing(case: str, data: dict) -> list[dict]:
    event_by_target: dict[int, list[float]] = {}
    for row in data["events"]:
        event_by_target.setdefault(int(row["target_id"]), []).append(
            float(row["time"])
        )
    rows = []
    for target_id in range(20, 28):
        assigned_count = int(np.sum(ASSIGNMENT == target_id))
        times = sorted(event_by_target.get(target_id, []))
        complete = len(times) == assigned_count
        rows.append(
            {
                "case": case,
                "target_internal_id": target_id,
                "target_paper_label": f"A{target_id - 19}",
                "assigned_interceptor_count": assigned_count,
                "observed_hit_count": len(times),
                "complete_group": complete,
                "first_hit_time_s": min(times) if times else "",
                "last_observed_hit_time_s": max(times) if times else "",
                "observed_within_group_spread_s": (
                    max(times) - min(times) if len(times) >= 2 else 0.0
                ),
                "strict_completion_delay_lower_bound_s": (
                    "" if complete or not times else HORIZON_S - min(times)
                ),
            }
        )
    return rows


def plot_group_timing(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.85), sharey=True)
    for col, case in enumerate(("case1", "case2")):
        ax = axes[col]
        case_rows = [row for row in rows if row["case"] == case]
        for x, row in enumerate(case_rows, start=1):
            first = float(row["first_hit_time_s"])
            last = float(row["last_observed_hit_time_s"])
            if row["complete_group"]:
                ax.plot([x, x], [first, last], color="#0072B2", lw=2.2)
                ax.scatter(
                    [x], [first], color="#0072B2", marker="o", s=24,
                    zorder=4,
                )
                ax.scatter(
                    [x], [last], color="#009E73", marker="D", s=22,
                    zorder=4,
                )
            else:
                ax.plot(
                    [x, x], [first, HORIZON_S], color="#D55E00",
                    ls="--", lw=1.8,
                )
                ax.scatter(
                    [x], [first], color="#D55E00", marker="o", s=27,
                    zorder=4,
                )
                ax.scatter(
                    [x], [HORIZON_S], facecolors="none",
                    edgecolors="#D55E00", marker="^", s=38, zorder=4,
                )
        ax.set_xticks(range(1, 9))
        ax.set_xticklabels([rf"$A_{{{i}}}$" for i in range(1, 9)])
        ax.set_xlabel("Target group")
        ax.set_title(f"Case {col + 1}", pad=2)
        ax.set_ylim(20.0, 78.0)
        ax.grid(True)
        ax.text(
            0.02, 0.08, f"({chr(ord('a') + col)})",
            transform=ax.transAxes, ha="left", va="top", fontweight="bold",
        )
    axes[0].set_ylabel("Absolute interception time (s)")
    legend_handles = [
        Line2D(
            [0], [0], color="#0072B2", marker="o", lw=2,
            label="First group hit",
        ),
        Line2D(
            [0], [0], color="#009E73", marker="D", lw=0,
            label="Last group hit",
        ),
        Line2D(
            [0], [0], color="#D55E00", marker="^", ls="--",
            markerfacecolor="none", label="Incomplete at 75 s",
        ),
    ]
    axes[0].legend(
        handles=legend_handles, loc="upper left", fontsize=7, ncol=1
    )
    fig.subplots_adjust(wspace=0.18)
    for suffix, kwargs in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 600}),
    ):
        fig.savefig(OUT / f"group_absolute_timing_v10.{suffix}", **kwargs)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    window_rows: list[dict] = []
    peer_rows: list[dict] = []
    group_timing_rows: list[dict] = []
    plot_data: dict[str, dict] = {}
    for case in CASES:
        observed = load_observed(case)
        summary, terminal, peers, curves = analyze_case(case, observed)
        summaries.append(summary)
        window_rows.extend(terminal)
        peer_rows.extend(peers)
        group_timing_rows.extend(build_group_timing(case, observed))
        plot_data[case] = curves

    cf_rows = read_csv(CF_ROOT / "counterfactual_results.csv")
    save_csv(OUT / "failure_mechanism_summary.csv", summaries)
    save_csv(OUT / "failure_terminal_window.csv", window_rows)
    save_csv(OUT / "group_peer_comparison.csv", peer_rows)
    save_csv(OUT / "group_absolute_timing.csv", group_timing_rows)
    save_csv(OUT / "counterfactual_results.csv", cf_rows)
    report = {
        "data_source": "evaluator-native frozen-policy trajectories",
        "dt_s": DT,
        "hit_radius_m": HIT_RADIUS_M,
        "summaries": summaries,
        "counterfactual_results": cf_rows,
        "interpretation_rule": (
            "Range-rate reversal at d>d_kill is the direct geometric miss "
            "mechanism. Counterfactual removals identify contributing "
            "observation/actuation impairments but are not treated as an "
            "additive causal decomposition in this nonlinear closed loop."
        ),
    }
    (OUT / "failure_cause_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    plot_diagnostics(plot_data, summaries)
    plot_group_timing(group_timing_rows)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
