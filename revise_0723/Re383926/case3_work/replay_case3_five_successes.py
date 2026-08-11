#!/usr/bin/env python3
"""Deterministically replay and plot five preselected successful Case-3 trials."""
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
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from run_case3_end_to_end import (
    apply_assignment,
    build_policy,
    configure_case3_geometry,
    make_case3_attacker_callback,
    solve_idbo,
)


COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#332288",
]


def _snapshot(env, assignment):
    defenders = [a for a in env.world.agents if a.adversary]
    attackers = [a for a in env.world.agents if not a.adversary]
    def_pos = np.asarray([a.state.p_pos for a in defenders], dtype=np.float32)
    atk_pos = np.asarray([a.state.p_pos for a in attackers], dtype=np.float32)
    def_vel = np.asarray([a.state.p_vel for a in defenders], dtype=np.float32)
    atk_vel = np.asarray([a.state.p_vel for a in attackers], dtype=np.float32)
    loads = np.asarray([a.state.load for a in defenders], dtype=np.float32)
    dists = np.asarray(
        [
            np.linalg.norm(env.world.agents[int(20 + assignment[i])].state.p_pos - a.state.p_pos)
            for i, a in enumerate(defenders)
        ],
        dtype=np.float32,
    )
    return def_pos, atk_pos, def_vel, atk_vel, loads, dists


def replay_seed(args, cfg, env, policy, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    obs = np.asarray(env.reset(), dtype=np.float32)
    geometry = configure_case3_geometry(env, seed)
    assignment, idbo = solve_idbo(env, seed, args.idbo_population, args.idbo_iterations)
    apply_assignment(env, assignment)
    callback = make_case3_attacker_callback()
    for attacker in [a for a in env.world.agents if not a.adversary]:
        attacker.action_callback = callback
    defenders = [a for a in env.world.agents if a.adversary]
    obs = np.asarray([env._get_obs(a) for a in defenders], dtype=np.float32)
    rnn = np.zeros((20, cfg.recurrent_N, cfg.hidden_size), dtype=np.float32)
    masks = np.ones((20, 1), dtype=np.float32)
    hit_times = np.full(20, np.nan, dtype=float)
    miss_at_arrival = np.full(20, np.nan, dtype=float)
    min_distance = np.full(20, np.inf, dtype=float)
    histories = [[] for _ in defenders]
    snapshots = [_snapshot(env, assignment)]
    rewards_by_step = []

    for step in range(args.max_steps):
        with torch.no_grad():
            actions, rnn = policy.act(obs, rnn, masks, deterministic=True)
        obs, rewards, dones, _ = env.step(actions.detach().cpu().numpy(), step)
        obs = np.asarray(obs, dtype=np.float32)
        snapshots.append(_snapshot(env, assignment))
        rewards_by_step.append(np.asarray(rewards, dtype=np.float32).reshape(-1))
        for i, defender in enumerate(defenders):
            target = env.world.agents[int(20 + assignment[i])]
            distance = float(np.linalg.norm(target.state.p_pos - defender.state.p_pos))
            min_distance[i] = min(min_distance[i], distance)
            histories[i].append(float(np.linalg.norm(np.asarray(defender.state.load)[1:3])))
            if getattr(defender.state, "actual_hit", False) and not np.isfinite(hit_times[i]):
                hit_times[i] = (step + 1) * env.dt
                miss_at_arrival[i] = distance
        done = np.asarray(dones, dtype=bool)
        if done.all():
            break
        masks[:] = 1.0
        masks[done, 0] = 0.0

    arrays = [np.stack([s[i] for s in snapshots]) for i in range(6)]
    terminal_steps = max(1, int(round(args.terminal_window_s / env.dt)))
    group_rows = []
    for target_index in range(8):
        members = np.where(assignment == target_index)[0]
        times = hit_times[members]
        if not np.isfinite(times).all():
            raise RuntimeError(f"seed {seed}: incomplete group {target_index}, times={times}")
        mean_time = float(np.mean(times))
        group_rows.append(
            dict(
                seed=seed,
                target_index=target_index,
                target_id=20 + target_index,
                group_size=len(members),
                defender_ids=";".join(map(str, members.tolist())),
                mean_arrival_time_s=mean_time,
                min_arrival_time_s=float(np.min(times)),
                max_arrival_time_s=float(np.max(times)),
                arrival_spread_s=float(np.max(times) - np.min(times)),
                E_co_time_s=float(np.mean(np.abs(times - mean_time))),
                E_n_g=float(
                    np.mean(
                        [
                            np.mean(histories[i][-terminal_steps:])
                            for i in members
                        ]
                    )
                ),
                E_miss_m=float(np.mean(miss_at_arrival[members])),
                E_t_s=float(np.max(times)),
            )
        )
    episode = dict(
        seed=seed,
        interceptor_hit_count=int(np.isfinite(hit_times).sum()),
        target_coverage_count=8,
        all_targets_hit=True,
        all_targets_coordinated=bool(
            max(r["arrival_spread_s"] for r in group_rows) <= args.sync_tol
        ),
        mission_success=bool(
            np.isfinite(hit_times).all()
            and max(r["arrival_spread_s"] for r in group_rows) <= args.sync_tol
        ),
        E_co_time_s=float(np.mean([r["E_co_time_s"] for r in group_rows])),
        E_n_g=float(np.mean([r["E_n_g"] for r in group_rows])),
        E_miss_m=float(np.mean([r["E_miss_m"] for r in group_rows])),
        E_t_s=float(np.mean([r["E_t_s"] for r in group_rows])),
        max_group_spread_s=float(max(r["arrival_spread_s"] for r in group_rows)),
        mean_interceptor_min_distance_m=float(np.mean(min_distance)),
        worst_interceptor_min_distance_m=float(np.max(min_distance)),
        elapsed_steps=int(arrays[0].shape[0] - 1),
        **idbo,
    )
    if not episode["mission_success"]:
        raise RuntimeError(f"seed {seed} did not reproduce strict success: {episode}")
    return dict(
        episode=episode,
        assignment=assignment,
        geometry=geometry,
        hit_times=hit_times,
        miss_at_arrival=miss_at_arrival,
        min_distance=min_distance,
        group_rows=group_rows,
        def_pos=arrays[0],
        atk_pos=arrays[1],
        def_vel=arrays[2],
        atk_vel=arrays[3],
        loads=arrays[4],
        distances=arrays[5],
        rewards=np.asarray(rewards_by_step, dtype=np.float32),
    )


def save_trial(outdir, data, dt):
    seed = int(data["episode"]["seed"])
    trial = outdir / f"seed_{seed}"
    trial.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        trial / f"seed_{seed}_trajectory_raw.npz",
        def_pos=data["def_pos"],
        atk_pos=data["atk_pos"],
        def_vel=data["def_vel"],
        atk_vel=data["atk_vel"],
        loads=data["loads"],
        distances=data["distances"],
        rewards=data["rewards"],
        assignment=data["assignment"],
        hit_times=data["hit_times"],
        miss_at_arrival=data["miss_at_arrival"],
        min_distance=data["min_distance"],
        dt=dt,
        seed=seed,
    )
    with (trial / f"seed_{seed}_episode_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data["episode"].keys()))
        w.writeheader()
        w.writerow(data["episode"])
    with (trial / f"seed_{seed}_group_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data["group_rows"][0].keys()))
        w.writeheader()
        w.writerows(data["group_rows"])
    assignment_rows = []
    for i, target_index in enumerate(data["assignment"]):
        assignment_rows.append(
            dict(
                seed=seed,
                defender_id=i,
                target_index=int(target_index),
                target_id=int(20 + target_index),
                hit_time_s=float(data["hit_times"][i]),
                miss_at_arrival_m=float(data["miss_at_arrival"][i]),
                min_distance_m=float(data["min_distance"][i]),
            )
        )
    with (trial / f"seed_{seed}_assignment_arrivals.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(assignment_rows[0].keys()))
        w.writeheader()
        w.writerows(assignment_rows)
    with (trial / f"seed_{seed}_trajectory_long.csv").open("w", newline="", encoding="utf-8") as f:
        names = [
            "seed", "time_s", "entity_type", "entity_id", "assigned_target_id",
            "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps",
            "nx_g", "ny_g", "nz_g", "distance_to_assigned_target_m",
        ]
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        for step in range(data["def_pos"].shape[0]):
            t = step * dt
            for i in range(20):
                w.writerow(
                    dict(
                        seed=seed, time_s=t, entity_type="defender", entity_id=i,
                        assigned_target_id=20 + int(data["assignment"][i]),
                        x_m=float(data["def_pos"][step, i, 0]),
                        y_m=float(data["def_pos"][step, i, 1]),
                        z_m=float(data["def_pos"][step, i, 2]),
                        vx_mps=float(data["def_vel"][step, i, 0]),
                        vy_mps=float(data["def_vel"][step, i, 1]),
                        vz_mps=float(data["def_vel"][step, i, 2]),
                        nx_g=float(data["loads"][step, i, 0]),
                        ny_g=float(data["loads"][step, i, 1]),
                        nz_g=float(data["loads"][step, i, 2]),
                        distance_to_assigned_target_m=float(data["distances"][step, i]),
                    )
                )
            for j in range(8):
                w.writerow(
                    dict(
                        seed=seed, time_s=t, entity_type="attacker", entity_id=20 + j,
                        assigned_target_id="", x_m=float(data["atk_pos"][step, j, 0]),
                        y_m=float(data["atk_pos"][step, j, 1]),
                        z_m=float(data["atk_pos"][step, j, 2]),
                        vx_mps=float(data["atk_vel"][step, j, 0]),
                        vy_mps=float(data["atk_vel"][step, j, 1]),
                        vz_mps=float(data["atk_vel"][step, j, 2]),
                        nx_g="", ny_g="", nz_g="", distance_to_assigned_target_m="",
                    )
                )
    plot_trial(trial, data, dt)


def _style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save(fig, stem):
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_trial(trial, data, dt):
    _style()
    seed = int(data["episode"]["seed"])
    t = np.arange(data["def_pos"].shape[0]) * dt

    fig = plt.figure(figsize=(3.5, 2.7))
    ax = fig.add_subplot(111, projection="3d")
    for j in range(8):
        ax.plot(
            data["atk_pos"][:, j, 0],
            data["atk_pos"][:, j, 1],
            data["atk_pos"][:, j, 2],
            color=COLORS[j],
            ls="--",
            lw=1.3,
            label=f"Target {j + 1}",
        )
        ax.scatter(*data["atk_pos"][0, j], marker="^", color=COLORS[j], s=22)
    for i in range(20):
        j = int(data["assignment"][i])
        ax.plot(
            data["def_pos"][:, i, 0],
            data["def_pos"][:, i, 1],
            data["def_pos"][:, i, 2],
            color=COLORS[j],
            lw=0.72,
            alpha=0.82,
        )
        ax.scatter(*data["def_pos"][0, i], marker="o", color=COLORS[j], s=6)
    ax.scatter([0], [0], [0], marker="*", color="black", s=42, label="Protected asset")
    ax.set_xlabel("$x$ (m)", labelpad=1)
    ax.set_ylabel("$y$ (m)", labelpad=1)
    ax.set_zlabel("$z$ (m)", labelpad=1)
    ax.view_init(elev=24, azim=-53)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    ax.text2D(0.02, 0.96, "(a)", transform=ax.transAxes, fontweight="bold")
    _save(fig, trial / f"seed_{seed}_trajectory_3d")

    fig, ax = plt.subplots(figsize=(3.5, 2.45))
    for j in range(8):
        members = np.where(data["assignment"] == j)[0]
        times = data["hit_times"][members]
        y = np.full(len(members), j + 1, dtype=float)
        offsets = np.linspace(-0.13, 0.13, len(members)) if len(members) > 1 else np.array([0.0])
        ax.scatter(times, y + offsets, color=COLORS[j], marker="o", s=22, edgecolor="black", linewidth=0.35)
        ax.plot([np.min(times), np.max(times)], [j + 1, j + 1], color=COLORS[j], lw=1.35)
        ax.text(np.max(times) + 0.012, j + 1, f"$m={len(members)}$", va="center", fontsize=6.2)
    ax.set_yticks(range(1, 9), [f"$T_{j}$" for j in range(1, 9)])
    ax.set_xlabel("Interception time (s)")
    ax.set_ylabel("Assigned target group")
    ax.grid(axis="x", color="0.88", lw=0.55)
    ax.tick_params(direction="in", length=3.0)
    ax.text(0.02, 0.96, "(b)", transform=ax.transAxes, fontweight="bold")
    _save(fig, trial / f"seed_{seed}_assignment_timing")

    fig = plt.figure(figsize=(7.16, 2.7), constrained_layout=True)
    ax3 = fig.add_subplot(121, projection="3d")
    for j in range(8):
        ax3.plot(data["atk_pos"][:, j, 0], data["atk_pos"][:, j, 1], data["atk_pos"][:, j, 2], color=COLORS[j], ls="--", lw=1.1)
        ax3.scatter(*data["atk_pos"][0, j], marker="^", color=COLORS[j], s=18)
    for i in range(20):
        j = int(data["assignment"][i])
        ax3.plot(data["def_pos"][:, i, 0], data["def_pos"][:, i, 1], data["def_pos"][:, i, 2], color=COLORS[j], lw=0.62, alpha=0.8)
    ax3.scatter([0], [0], [0], marker="*", color="black", s=38)
    ax3.set_xlabel("$x$ (m)", labelpad=1)
    ax3.set_ylabel("$y$ (m)", labelpad=1)
    ax3.set_zlabel("$z$ (m)", labelpad=1)
    ax3.view_init(elev=24, azim=-53)
    ax3.text2D(0.01, 0.97, "(a)", transform=ax3.transAxes, fontweight="bold")
    ax2 = fig.add_subplot(122)
    for j in range(8):
        members = np.where(data["assignment"] == j)[0]
        times = data["hit_times"][members]
        offsets = np.linspace(-0.13, 0.13, len(members)) if len(members) > 1 else np.array([0.0])
        ax2.scatter(times, j + 1 + offsets, color=COLORS[j], s=24, edgecolor="black", linewidth=0.35)
        ax2.plot([np.min(times), np.max(times)], [j + 1, j + 1], color=COLORS[j], lw=1.4)
        ax2.text(np.max(times) + 0.012, j + 1, f"$m={len(members)}$", va="center", fontsize=6.2)
    ax2.set_yticks(range(1, 9), [f"$T_{j}$" for j in range(1, 9)])
    ax2.set_xlabel("Interception time (s)")
    ax2.set_ylabel("Assigned target group")
    ax2.grid(axis="x", color="0.88", lw=0.55)
    ax2.tick_params(direction="in", length=3.0)
    ax2.text(0.01, 0.97, "(b)", transform=ax2.transAxes, fontweight="bold")
    _save(fig, trial / f"seed_{seed}_combined")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--hit-radius", type=float, default=3.0)
    parser.add_argument("--sync-tol", type=float, default=0.5)
    parser.add_argument("--terminal-window-s", type=float, default=1.0)
    parser.add_argument("--idbo-population", type=int, default=24)
    parser.add_argument("--idbo-iterations", type=int, default=80)
    parser.add_argument("--guide-gain", type=float, default=2.4)
    parser.add_argument("--guide-tau", type=float, default=0.25)
    parser.add_argument("--guide-lead", type=float, default=1.40)
    parser.add_argument("--residual-scale", type=float, default=0.20)
    parser.add_argument("--sync-speed-gain", type=float, default=0.01)
    parser.add_argument("--sync-tgo-ref", choices=["mean", "max", "min"], default="mean")
    args = parser.parse_args()
    args.seed = int(args.seeds[0])
    args.episodes = len(args.seeds)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    args.outdir.mkdir(parents=True, exist_ok=True)
    cfg, env, policy, _, model_dir = build_policy(args)
    rows = []
    for seed in args.seeds:
        data = replay_seed(args, cfg, env, policy, seed)
        save_trial(args.outdir, data, env.dt)
        rows.append(data["episode"])
        print(
            f"[replay] seed={seed} spread={data['episode']['max_group_spread_s']:.4f} "
            f"E_co={data['episode']['E_co_time_s']:.5f} E_miss={data['episode']['E_miss_m']:.4f}",
            flush=True,
        )
    ranked = sorted(
        rows,
        key=lambda r: (
            round(r["max_group_spread_s"], 10),
            round(r["E_co_time_s"], 10),
            round(r["E_miss_m"], 10),
            round(r["E_n_g"], 10),
        ),
    )
    ranking_rows = []
    for rank, row in enumerate(ranked, 1):
        ranking_rows.append(
            dict(
                rank=rank,
                recommended=rank == 1,
                seed=row["seed"],
                max_group_spread_s=row["max_group_spread_s"],
                E_co_time_s=row["E_co_time_s"],
                E_miss_m=row["E_miss_m"],
                E_n_g=row["E_n_g"],
                E_t_s=row["E_t_s"],
                ranking_rule="max spread, then E_co-time, E_miss, E_n (ascending)",
            )
        )
    with (args.outdir / "five_trial_objective_ranking.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ranking_rows[0].keys()))
        w.writeheader()
        w.writerows(ranking_rows)
    (args.outdir / "five_trial_manifest.json").write_text(
        json.dumps(
            {
                "seeds": args.seeds,
                "model_dir": str(model_dir),
                "policy_updated": False,
                "selection": "first five strict successes by ascending seed in formal 100-episode CSV",
                "ranking_rule": "max_group_spread_s, E_co_time_s, E_miss_m, E_n_g; all ascending",
                "recommended_seed": ranking_rows[0]["seed"],
                "configuration": {
                    "guide_gain": args.guide_gain,
                    "guide_tau": args.guide_tau,
                    "guide_lead": args.guide_lead,
                    "residual_scale": args.residual_scale,
                    "sync_speed_gain": args.sync_speed_gain,
                    "sync_tgo_ref": args.sync_tgo_ref,
                    "hit_radius_m": args.hit_radius,
                    "sync_tolerance_s": args.sync_tol,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
