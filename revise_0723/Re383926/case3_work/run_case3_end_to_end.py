#!/usr/bin/env python3
"""Frozen-policy Case-3 end-to-end evaluation on the stable_V2 code line.

The script deliberately does not train or update the policy.  For each seeded
episode it:
  1. resets the archived Case-2 state template;
  2. applies the preregistered unseen Case-3 geometry;
  3. solves target assignment with paper-faithful IDBO;
  4. executes a distinct hybrid attacker maneuver;
  5. evaluates the archived Case-2 recurrent policy and saves raw metrics.

All stochastic quantities are generated from the episode seed.  The output
contains enough raw information to recompute the paper's four terminal metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_3d_guidance import build_args, collect_model
from idbo_paper import IDBO_paper
from scenario_paper import Scenario as IDBOScenario


def _unit(x, fallback):
    x = np.asarray(x, dtype=float)
    norm = float(np.linalg.norm(x))
    if norm < 1e-10:
        return np.asarray(fallback, dtype=float)
    return x / norm


def _rotate_xy(v, angle):
    c, s = math.cos(angle), math.sin(angle)
    out = np.asarray(v, dtype=float).copy()
    out[:2] = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
    return out


def _vel_to_flight_state(v):
    speed = max(float(np.linalg.norm(v)), 1e-8)
    pitch = math.asin(float(np.clip(v[2] / speed, -1.0, 1.0)))
    yaw = math.atan2(float(v[1]), float(v[0]))
    return np.array([speed, pitch, yaw], dtype=float)


def configure_case3_geometry(env, seed):
    """Wider shell, staggered altitude, heterogeneous speed and heading bias."""
    rng = np.random.default_rng(seed + 300_003)
    attackers = [a for a in env.world.agents if not a.adversary]
    defenders = [a for a in env.world.agents if a.adversary]
    rows = []
    for j, agent in enumerate(attackers):
        old = np.asarray(agent.state.p_pos, dtype=float)
        azimuth = math.atan2(float(old[1]), float(old[0]))
        azimuth += 0.095 * math.sin(0.9 * j + 0.35) + rng.normal(0.0, 0.010)
        radius = 2075.0 + 32.0 * j + rng.normal(0.0, 15.0)
        altitude = 90.0 + 22.0 * (j % 4) + rng.normal(0.0, 2.0)
        position = np.array(
            [radius * math.cos(azimuth), radius * math.sin(azimuth), altitude],
            dtype=float,
        )
        direction = _unit(-position, np.array([-1.0, 0.0, 0.0]))
        heading_bias = (1.0 if j % 2 == 0 else -1.0) * (0.040 + 0.007 * (j % 3))
        direction = _unit(_rotate_xy(direction, heading_bias), direction)
        speed = float(np.clip(34.0 + 1.45 * j + rng.normal(0.0, 0.35), 34.0, 45.0))
        agent.state.p_pos = position
        agent.state.p_vel = speed * direction
        agent.state.v_vel = _vel_to_flight_state(agent.state.p_vel)
        agent.state.phase = float(2.0 * np.pi * j / len(attackers))
        rows.append(
            dict(
                attacker_id=int(agent.namenumber),
                radius_m=float(radius),
                altitude_m=float(altitude),
                speed_mps=speed,
                heading_bias_rad=float(heading_bias),
                x_m=float(position[0]),
                y_m=float(position[1]),
                z_m=float(position[2]),
            )
        )
    for agent in defenders:
        agent.state.defender_sensor_history = None
        agent.state.defender_obs_history = None
        agent.state.defender_sensor_filtered = None
        agent.state.defender_obs_filtered = None
    return rows


def _repair_assignment(assignment, scn):
    assign = np.asarray(assignment, dtype=int).copy()
    before = assign.copy()
    moves = 0
    for _ in range(10 * scn.N_D):
        counts = np.bincount(assign, minlength=scn.N_A)
        uncovered = np.where(counts == 0)[0]
        overflow = np.where(counts > scn.L_max)[0]
        if uncovered.size == 0 and overflow.size == 0:
            break
        target = int(uncovered[0]) if uncovered.size else int(np.argmin(counts))
        donors = [
            i
            for i, old in enumerate(assign)
            if counts[int(old)] > (1 if uncovered.size else scn.L_max)
        ]
        if not donors:
            raise RuntimeError("IDBO assignment repair has no feasible donor")

        def loss(i):
            old = int(assign[i])
            old_v = -math.log(max(1.0 - scn.p_int[i, old], 1e-8))
            new_v = -math.log(max(1.0 - scn.p_int[i, target], 1e-8))
            return old_v - new_v

        donor = min(donors, key=lambda i: (loss(i), i))
        assign[donor] = target
        moves += 1
    counts = np.bincount(assign, minlength=scn.N_A)
    if not (np.all(counts >= 1) and np.all(counts <= scn.L_max)):
        raise RuntimeError(f"infeasible repaired assignment: {counts.tolist()}")
    return assign, before, moves


def solve_idbo(env, seed, population, iterations):
    defenders = [a for a in env.world.agents if a.adversary]
    attackers = [a for a in env.world.agents if not a.adversary]
    scn = IDBOScenario(len(defenders), len(attackers), L_max=3, seed=seed)
    scn.pD = np.asarray([a.state.p_pos for a in defenders], dtype=float)
    scn.vD = np.asarray([a.state.p_vel for a in defenders], dtype=float)
    scn.pT = np.asarray([a.state.p_pos for a in attackers], dtype=float)
    scn.vT = np.asarray([a.state.p_vel for a in attackers], dtype=float)
    scn._precompute()
    start = time.perf_counter()
    best_cost, raw, convergence, history = IDBO_paper(
        population,
        iterations,
        scn,
        schedule="linear",
        seed=seed,
        return_history=True,
    )
    runtime_ms = 1000.0 * (time.perf_counter() - start)
    assignment, raw_before, repair_moves = _repair_assignment(raw, scn)
    return assignment, dict(
        idbo_best_cost=float(best_cost),
        idbo_repaired_cost=float(scn.assignment_cost(assignment)),
        idbo_runtime_ms=float(runtime_ms),
        idbo_final_population_cost=float(convergence[-1]),
        idbo_final_disagreement=float(history["gamma"][-1]),
        repair_moves=int(repair_moves),
        raw_counts=np.bincount(raw_before, minlength=scn.N_A).tolist(),
        final_counts=np.bincount(assignment, minlength=scn.N_A).tolist(),
    )


def apply_assignment(env, assignment):
    defenders = [a for a in env.world.agents if a.adversary]
    for i, agent in enumerate(defenders):
        agent.target = int(20 + assignment[i])
        target = env.world.agents[agent.target]
        distance = max(float(np.linalg.norm(target.state.p_pos - agent.state.p_pos)), 0.01)
        agent.state.dist0 = distance
        agent.state.dist1 = distance
        agent.state.dist_target = distance


def make_case3_attacker_callback():
    """Hybrid multi-sine, bang-bang and chirp maneuver, unseen in Cases 1/2."""

    def callback(agent, world):
        asset = world.landmarks[0].state.p_pos
        rel = asset - agent.state.p_pos
        horizontal_dist = max(float(np.linalg.norm(rel[:2])), 1.0)
        speed = max(float(agent.state.v_vel[0]), 1.0)
        pitch = float(agent.state.v_vel[1])
        yaw = float(agent.state.v_vel[2])
        desired_pitch = float(np.clip(math.atan2(float(rel[2]), horizontal_dist), -0.45, 0.05))
        desired_speed = 38.0 + 0.8 * ((int(agent.namenumber) - 20) % 5)
        load_x = (desired_speed - speed) / (9.81 * max(float(world.dt), 1e-6))

        q = math.atan2(float(rel[1]), float(rel[0]))
        q_dot = (
            (-agent.state.p_vel[1]) * math.cos(q)
            - (-agent.state.p_vel[0]) * math.sin(q)
        ) / horizontal_dist
        j = int(agent.namenumber) - 20
        phase = float(getattr(agent.state, "phase", 2.0 * np.pi * j / 8.0))
        t = float(agent.state.timestep)
        pn = 3.0 * q_dot * speed / 9.81
        if t < 14.0:
            extra_y = 0.30 * math.sin(0.23 * t + phase) + 0.16 * math.sin(0.51 * t + 0.4 * phase)
        elif t < 32.0:
            extra_y = (0.38 if int((t - 14.0) / 3.0) % 2 == 0 else -0.38) * (1.0 if j % 2 == 0 else -1.0)
        else:
            chirp_phase = 0.17 * (t - 32.0) + 0.009 * (t - 32.0) ** 2 + phase
            extra_y = 0.28 * math.sin(chirp_phase)
        fade = float(np.clip(horizontal_dist / 500.0, 0.12, 1.0))
        load_y = pn + fade * extra_y
        vertical_extra = fade * 0.16 * math.sin(0.19 * t + 0.65 * phase)
        load_z = (desired_pitch - pitch) * speed / (9.81 * max(float(world.dt), 1e-6)) + vertical_extra
        yaw_scale = max(float(getattr(world, "attacker_yaw_scale", 1.0)), 1e-6)
        pitch_scale = max(float(getattr(world, "attacker_pitch_scale", 1.0)), 1e-6)
        agent.action.u = np.array([load_x, load_y / yaw_scale, load_z / pitch_scale], dtype=np.float32)
        return agent.action

    return callback


def build_policy(args):
    raw = SimpleNamespace(
        seed=args.seed,
        eval_episodes=args.episodes,
        max_steps=args.max_steps,
        sync_tol=args.sync_tol,
        sync_min_hits=0,
        hit_radius_3d=args.hit_radius,
        defender_guidance_base_gain=args.guide_gain,
        defender_guidance_tau=args.guide_tau,
        defender_guidance_lead=args.guide_lead,
        defender_residual_scale=args.residual_scale,
        defender_load_limit=1.0,
        defender_axial_min=-0.1,
        defender_axial_max=1.0,
        defender_sync_speed_gain=args.sync_speed_gain,
        defender_sync_tgo_ref=args.sync_tgo_ref,
        defender_speed_target=0.0,
        defender_speed_gain=0.0,
        defender_min_accel_load=0.0,
        defender_speed_min=12.0,
        defender_speed_max=40.0,
        defender_sensor_delay_steps=0,
        defender_obs_pos_noise_std=0.0,
        defender_obs_vel_noise_std=0.0,
        defender_obs_filter_alpha=1.0,
        defender_command_lag_tau=0.0,
        reward_w_smooth=0.0,
        reference_control_root="",
        reward_w_ref_control=0.0,
        reward_w_ref_rate=0.0,
        defender_reference_blend=0.0,
        target_assignment_mode="fixed",
        target_assignment_spread_weight=6.0,
        attack_maneuver_gain=1.2,
        attack_maneuver_offset_gain=1.25,
        attack_maneuver_freq=0.17,
        attack_maneuver_fade_range=450.0,
        case1_lateral_base=0.95,
        case1_lateral_tail=0.40,
        case1_vertical_amp=0.35,
        case2_lateral_amp=1.0,
        case2_maneuver_freq=2.0 * np.pi / 50.0,
        case2_vertical_amp=0.25,
        case2_vertical_freq_scale=0.50,
        stochastic_eval=False,
        eval_different_seed=True,
        require_success_plot=False,
        require_all_hit=False,
        paper_preset_path=str(args.preset),
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
        attacker_load_limit=200.0,
        attacker_yaw_scale=1.0,
        attacker_pitch_scale=1.0,
        outdir=str(args.outdir),
    )
    cfg = build_args("case2", raw)
    env, policy, device, model_dir = collect_model(cfg, args.model_dir)
    return cfg, env, policy, device, model_dir


def run(args):
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cfg, env, policy, device, model_dir = build_policy(args)
    callback = make_case3_attacker_callback()
    rows = []
    assignments = []
    geometry_rows = []
    terminal_window_steps = max(1, int(round(args.terminal_window_s / env.dt)))

    for ep in range(args.episodes):
        seed = args.seed + ep
        np.random.seed(seed)
        torch.manual_seed(seed)
        obs = np.asarray(env.reset(), dtype=np.float32)
        geometry = configure_case3_geometry(env, seed)
        assignment, idbo = solve_idbo(env, seed, args.idbo_population, args.idbo_iterations)
        apply_assignment(env, assignment)
        for attacker in [a for a in env.world.agents if not a.adversary]:
            attacker.action_callback = callback
        defenders = [a for a in env.world.agents if a.adversary]
        obs = np.asarray([env._get_obs(a) for a in defenders], dtype=np.float32)
        rnn = np.zeros((len(defenders), cfg.recurrent_N, cfg.hidden_size), dtype=np.float32)
        masks = np.ones((len(defenders), 1), dtype=np.float32)
        hit_times = np.full(len(defenders), np.nan, dtype=float)
        miss_at_arrival = np.full(len(defenders), np.nan, dtype=float)
        min_dist = np.full(len(defenders), np.inf, dtype=float)
        load_history = [[] for _ in defenders]
        elapsed_steps = 0

        for step in range(args.max_steps):
            with torch.no_grad():
                actions, rnn = policy.act(obs, rnn, masks, deterministic=True)
            obs, _, dones, _ = env.step(actions.detach().cpu().numpy(), step)
            obs = np.asarray(obs, dtype=np.float32)
            elapsed_steps = step + 1
            for i, defender in enumerate(defenders):
                target = env.world.agents[defender.target]
                distance = float(np.linalg.norm(target.state.p_pos - defender.state.p_pos))
                min_dist[i] = min(min_dist[i], distance)
                load = np.asarray(defender.state.load, dtype=float)
                load_history[i].append(float(np.linalg.norm(load[1:3])))
                if getattr(defender.state, "actual_hit", False) and not np.isfinite(hit_times[i]):
                    hit_times[i] = elapsed_steps * env.dt
                    miss_at_arrival[i] = distance
            done_arr = np.asarray(dones, dtype=bool)
            if done_arr.all():
                break
            masks[:] = 1.0
            masks[done_arr, 0] = 0.0

        groups = []
        all_group_hit = True
        all_group_sync = True
        for target_index in range(8):
            members = np.where(assignment == target_index)[0]
            times = hit_times[members]
            group_hit = bool(len(members) > 0 and np.isfinite(times).all())
            all_group_hit &= group_hit
            if group_hit:
                mean_time = float(np.mean(times))
                eco = float(np.mean(np.abs(times - mean_time)))
                spread = float(np.max(times) - np.min(times))
                en = float(
                    np.mean(
                        [
                            np.mean(load_history[i][-terminal_window_steps:])
                            for i in members
                        ]
                    )
                )
                emiss = float(np.mean(miss_at_arrival[members]))
                et = float(np.max(times))
            else:
                eco = en = emiss = et = spread = float("nan")
            group_sync = bool(group_hit and spread <= args.sync_tol)
            all_group_sync &= group_sync
            groups.append((eco, en, emiss, et, spread))

        successful_groups = [g for g in groups if np.isfinite(g[0])]
        mission_success = bool(all_group_hit and all_group_sync)
        row = dict(
            episode=ep + 1,
            seed=seed,
            interceptor_hit_count=int(np.isfinite(hit_times).sum()),
            target_coverage_count=int(sum(np.isfinite(g[0]) for g in groups)),
            all_targets_hit=bool(all_group_hit),
            all_targets_coordinated=bool(all_group_sync),
            mission_success=mission_success,
            E_co_time_s=float(np.mean([g[0] for g in successful_groups])) if successful_groups else float("nan"),
            E_n_g=float(np.mean([g[1] for g in successful_groups])) if successful_groups else float("nan"),
            E_miss_m=float(np.mean([g[2] for g in successful_groups])) if successful_groups else float("nan"),
            E_t_s=float(np.mean([g[3] for g in successful_groups])) if successful_groups else float("nan"),
            max_group_spread_s=float(np.nanmax([g[4] for g in groups])) if successful_groups else float("nan"),
            mean_interceptor_min_distance_m=float(np.mean(min_dist)),
            worst_interceptor_min_distance_m=float(np.max(min_dist)),
            elapsed_steps=elapsed_steps,
            **idbo,
        )
        rows.append(row)
        for i, target_index in enumerate(assignment):
            assignments.append(
                dict(
                    episode=ep + 1,
                    seed=seed,
                    defender_id=i,
                    target_index=int(target_index),
                    target_id=int(20 + target_index),
                    hit_time_s=float(hit_times[i]),
                    min_distance_m=float(min_dist[i]),
                    miss_at_arrival_m=float(miss_at_arrival[i]),
                )
            )
        for geom in geometry:
            geometry_rows.append(dict(episode=ep + 1, seed=seed, **geom))
        print(
            f"[case3] {ep + 1}/{args.episodes} hits={row['interceptor_hit_count']}/20 "
            f"covered={row['target_coverage_count']}/8 mission={int(mission_success)} "
            f"idbo={row['idbo_runtime_ms']:.1f} ms",
            flush=True,
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.outdir / "case3_episode_metrics.csv", rows)
    _write_csv(args.outdir / "case3_assignment_and_arrivals.csv", assignments)
    _write_csv(args.outdir / "case3_initial_geometry.csv", geometry_rows)
    successful = [r for r in rows if r["mission_success"]]
    summary = dict(
        protocol="unseen Case-3 geometry + IDBO assignment + frozen stable_V2 Case-2 policy",
        episodes=args.episodes,
        seed_start=args.seed,
        model_dir=str(model_dir),
        policy_updated=False,
        idbo_population=args.idbo_population,
        idbo_iterations=args.idbo_iterations,
        hit_radius_m=args.hit_radius,
        sync_tolerance_s=args.sync_tol,
        terminal_window_s=args.terminal_window_s,
        interceptor_hit_rate=float(np.mean([r["interceptor_hit_count"] / 20.0 for r in rows])),
        target_coverage_rate=float(np.mean([r["target_coverage_count"] / 8.0 for r in rows])),
        all_targets_hit_rate=float(np.mean([r["all_targets_hit"] for r in rows])),
        cooperative_mission_success_rate=float(np.mean([r["mission_success"] for r in rows])),
        successful_trials=len(successful),
        successful_trial_metrics={
            key: float(np.mean([r[key] for r in successful])) if successful else None
            for key in ["E_co_time_s", "E_n_g", "E_miss_m", "E_t_s"]
        },
    )
    (args.outdir / "case3_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = dict(
        command=" ".join(sys.argv),
        generated_unix=time.time(),
        files={},
        configuration=vars(args).copy(),
    )
    manifest["configuration"]["model_dir"] = str(args.model_dir)
    manifest["configuration"]["preset"] = str(args.preset)
    manifest["configuration"]["outdir"] = str(args.outdir)
    for path in sorted(args.outdir.glob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=73001)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--hit-radius", type=float, default=3.0)
    parser.add_argument("--sync-tol", type=float, default=0.5)
    parser.add_argument("--terminal-window-s", type=float, default=1.0)
    parser.add_argument("--idbo-population", type=int, default=24)
    parser.add_argument("--idbo-iterations", type=int, default=80)
    parser.add_argument("--guide-gain", type=float, default=2.4)
    parser.add_argument("--guide-tau", type=float, default=0.40)
    parser.add_argument("--guide-lead", type=float, default=1.70)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--sync-speed-gain", type=float, default=0.0)
    parser.add_argument(
        "--sync-tgo-ref",
        choices=["mean", "max", "min"],
        default="mean",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
