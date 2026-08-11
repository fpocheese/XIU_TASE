#!/usr/bin/env python3
"""Frozen-policy reviewer experiments for Sections 3.8, 3.9, and 1.6.

The script never performs backpropagation or an optimizer step.  It evaluates
one already-trained ART-MAPPO checkpoint on a named condition and writes
episode-, target-, assignment-, and summary-level CSV files.  Independent
processes can be launched for different conditions without sharing RNG state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_art_mappo_ablation_3d import (  # noqa: E402
    build_args,
    collect_model,
)


PAPER_ASSIGNMENT = np.array(
    [
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3,
    ],
    dtype=int,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _wilson_interval(successes: int, total: int, z: float = 1.9599639845):
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return center - radius, center + radius


def _mean_ci(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(np.mean(x))
    if x.size == 1:
        return mean, mean, mean, 1
    half = 1.9599639845 * float(np.std(x, ddof=1)) / math.sqrt(x.size)
    return mean, mean - half, mean + half, int(x.size)


def _rotate_xy(vector: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    out = np.asarray(vector, dtype=float).copy()
    out[:2] = np.array(
        [c * out[0] - s * out[1], s * out[0] + c * out[1]],
        dtype=float,
    )
    return out


def _velocity_to_flight_state(velocity: np.ndarray) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=float)
    speed = max(float(np.linalg.norm(velocity)), 1e-8)
    pitch = math.asin(float(np.clip(velocity[2] / speed, -1.0, 1.0)))
    yaw = math.atan2(float(velocity[1]), float(velocity[0]))
    return np.array([speed, pitch, yaw], dtype=float)


def perturb_initial_state(env, seed: int, scale: float) -> dict:
    """Apply a reproducible, physically small perturbation after paper reset."""
    rng = np.random.default_rng(seed)
    defenders = [a for a in env.world.agents if a.adversary]
    attackers = [a for a in env.world.agents if not a.adversary]
    if scale <= 0.0:
        return {
            "defender_position_rms_m": 0.0,
            "attacker_radial_scale_std": 0.0,
        }

    def_offsets = []
    for agent in defenders:
        offset = rng.normal(0.0, 8.0 * scale, size=2)
        agent.state.p_pos[:2] += offset
        def_offsets.append(float(np.linalg.norm(offset)))
        yaw_delta = float(rng.normal(0.0, 0.045 * scale))
        speed_scale = float(np.clip(rng.normal(1.0, 0.025 * scale), 0.9, 1.1))
        agent.state.p_vel = _rotate_xy(agent.state.p_vel, yaw_delta) * speed_scale
        agent.state.v_vel = _velocity_to_flight_state(agent.state.p_vel)

    attacker_scales = []
    for agent in attackers:
        radial_scale = float(
            np.clip(rng.normal(1.0, 0.018 * scale), 0.94, 1.06)
        )
        angle_delta = float(rng.normal(0.0, 0.025 * scale))
        agent.state.p_pos = _rotate_xy(agent.state.p_pos, angle_delta)
        agent.state.p_pos[:2] *= radial_scale
        agent.state.p_pos[2] += float(rng.normal(0.0, 3.0 * scale))
        attacker_scales.append(radial_scale)
        speed = float(np.linalg.norm(agent.state.p_vel))
        direction = -agent.state.p_pos / max(
            float(np.linalg.norm(agent.state.p_pos)), 1e-8
        )
        speed *= float(np.clip(rng.normal(1.0, 0.02 * scale), 0.92, 1.08))
        agent.state.p_vel = speed * direction
        agent.state.v_vel = _velocity_to_flight_state(agent.state.p_vel)

    return {
        "defender_position_rms_m": float(
            np.sqrt(np.mean(np.square(def_offsets)))
        ),
        "attacker_radial_scale_std": float(np.std(attacker_scales)),
    }


def configure_case3_initial_state(env, seed: int) -> dict:
    """Apply the preregistered unseen Case-3 engagement geometry.

    Case 3 uses the Case-2 preset only as a reproducible template.  Before
    target assignment, attackers are moved to a wider, staggered-altitude
    shell and receive biased inbound headings and heterogeneous speeds.  IDBO
    therefore solves the assignment from the actual transformed snapshot.
    """
    rng = np.random.default_rng(seed + 300_003)
    attackers = [a for a in env.world.agents if not a.adversary]
    defenders = [a for a in env.world.agents if a.adversary]
    radii = []
    altitudes = []
    speeds = []
    heading_biases = []
    for j, agent in enumerate(attackers):
        position = np.asarray(agent.state.p_pos, dtype=float).copy()
        azimuth = math.atan2(float(position[1]), float(position[0]))
        azimuth += (
            0.11 * math.sin(0.9 * j + 0.35)
            + float(rng.normal(0.0, 0.012))
        )
        radius = (
            2260.0
            + 42.0 * j
            + float(rng.normal(0.0, 18.0))
        )
        altitude = (
            85.0
            + 30.0 * (j % 4)
            + float(rng.normal(0.0, 2.5))
        )
        agent.state.p_pos = np.array(
            [
                radius * math.cos(azimuth),
                radius * math.sin(azimuth),
                altitude,
            ],
            dtype=float,
        )

        direction = -agent.state.p_pos / max(
            float(np.linalg.norm(agent.state.p_pos)), 1e-8
        )
        heading_bias = (
            (1.0 if j % 2 == 0 else -1.0)
            * (0.055 + 0.010 * (j % 3))
        )
        direction = _rotate_xy(direction, heading_bias)
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        speed = (
            34.0
            + 1.65 * j
            + float(rng.normal(0.0, 0.45))
        )
        speed = float(np.clip(speed, 34.0, 47.0))
        agent.state.p_vel = speed * direction
        agent.state.v_vel = _velocity_to_flight_state(agent.state.p_vel)
        agent.state.phase = float(2.0 * np.pi * j / max(len(attackers), 1))
        radii.append(radius)
        altitudes.append(altitude)
        speeds.append(speed)
        heading_biases.append(heading_bias)

    # Discard delayed/filter state computed from the template geometry.  The
    # first Case-3 observation then starts a fresh, physically consistent
    # one-step sensing history.
    for agent in defenders:
        agent.state.defender_obs_history = None
        agent.state.defender_obs_filtered = None
        agent.state.defender_last_observed_target_pos = None
        agent.state.defender_last_observed_target_vel = None

    return {
        "case3_radius_min_m": float(np.min(radii)),
        "case3_radius_max_m": float(np.max(radii)),
        "case3_altitude_min_m": float(np.min(altitudes)),
        "case3_altitude_max_m": float(np.max(altitudes)),
        "case3_speed_min_mps": float(np.min(speeds)),
        "case3_speed_max_mps": float(np.max(speeds)),
        "case3_abs_heading_bias_mean_rad": float(
            np.mean(np.abs(heading_biases))
        ),
    }


def _repair_assignment(assignment, scenario):
    """Deterministic feasibility repair required by the manuscript algorithm."""
    assign = np.asarray(assignment, dtype=int).copy()
    before = assign.copy()
    moves = 0
    for _ in range(10 * scenario.N_D):
        counts = np.bincount(assign, minlength=scenario.N_A)
        uncovered = np.where(counts == 0)[0]
        overflow = np.where(counts > scenario.L_max)[0]
        if uncovered.size == 0 and overflow.size == 0:
            break

        if uncovered.size:
            target = int(uncovered[0])
            donors = [
                i
                for i, old in enumerate(assign)
                if counts[int(old)] > 1
            ]
        else:
            target = int(np.argmin(counts))
            donors = [
                i
                for i, old in enumerate(assign)
                if counts[int(old)] > scenario.L_max
            ]
        if not donors:
            raise RuntimeError("no feasible donor during assignment repair")

        def move_delta(i):
            old = int(assign[i])
            old_value = -math.log(max(1.0 - scenario.p_int[i, old], 1e-8))
            new_value = -math.log(
                max(1.0 - scenario.p_int[i, target], 1e-8)
            )
            return old_value - new_value

        donor = min(donors, key=lambda i: (move_delta(i), i))
        assign[donor] = target
        moves += 1
    counts = np.bincount(assign, minlength=scenario.N_A)
    feasible = bool(
        np.all(counts >= 1) and np.all(counts <= scenario.L_max)
    )
    if not feasible:
        raise RuntimeError(
            f"assignment repair failed, counts={counts.tolist()}"
        )
    return assign, {
        "repair_moves": moves,
        "raw_counts": np.bincount(
            before, minlength=scenario.N_A
        ).tolist(),
        "final_counts": counts.tolist(),
    }


def solve_idbo_assignment(env, seed: int, population: int, iterations: int):
    from idbo_paper import IDBO_paper
    from scenario_paper import Scenario

    defenders = [a for a in env.world.agents if a.adversary]
    attackers = [a for a in env.world.agents if not a.adversary]
    scn = Scenario(
        n_def=len(defenders),
        n_att=len(attackers),
        L_max=3,
        seed=seed,
    )
    scn.pD = np.asarray([a.state.p_pos for a in defenders], dtype=float)
    scn.vD = np.asarray([a.state.p_vel for a in defenders], dtype=float)
    scn.pT = np.asarray([a.state.p_pos for a in attackers], dtype=float)
    scn.vT = np.asarray([a.state.p_vel for a in attackers], dtype=float)
    scn._precompute()
    start = time.perf_counter()
    best_cost, raw_assignment, convergence, history = IDBO_paper(
        population,
        iterations,
        scn,
        schedule="linear",
        seed=seed,
        return_history=True,
    )
    runtime_ms = 1000.0 * (time.perf_counter() - start)
    assignment, repair = _repair_assignment(raw_assignment, scn)
    repaired_cost = float(scn.assignment_cost(assignment))
    return assignment, {
        "idbo_best_cost": float(best_cost),
        "idbo_repaired_cost": repaired_cost,
        "idbo_runtime_ms": runtime_ms,
        "idbo_final_population_cost": float(convergence[-1]),
        "idbo_final_disagreement": float(history["gamma"][-1]),
        **repair,
    }


def apply_assignment(env, assignment: np.ndarray):
    defenders = [a for a in env.world.agents if a.adversary]
    for i, agent in enumerate(defenders):
        agent.target = int(20 + assignment[i])
        target = env.world.agents[agent.target]
        distance = max(
            float(np.linalg.norm(target.state.p_pos - agent.state.p_pos)),
            0.01,
        )
        agent.state.dist0 = distance
        agent.state.dist1 = distance
        agent.state.dist_target = distance
    return np.asarray(
        [env._get_obs(agent) for agent in defenders],
        dtype=np.float32,
    )


def _build_raw_args(args):
    return SimpleNamespace(
        variant=args.variant,
        seed=args.seed,
        eval_episodes=args.episodes,
        max_steps=args.max_steps,
        sync_tol=args.sync_tol,
        stochastic_eval=args.stochastic_eval,
        eval_different_seed=True,
        require_success_plot=False,
        require_all_hit=False,
        attack_pattern=args.attack_pattern,
        unseen_chirp_rate=args.unseen_chirp_rate,
        unseen_frequency_scale=args.unseen_frequency_scale,
        unseen_vertical_load_amp=args.unseen_vertical_load_amp,
    )


_TORCH_CPU_THREADS_CONFIGURED = False


def build_environment_and_policy(args):
    global _TORCH_CPU_THREADS_CONFIGURED
    if args.cpu_eval and not _TORCH_CPU_THREADS_CONFIGURED:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        _TORCH_CPU_THREADS_CONFIGURED = True
    raw = _build_raw_args(args)
    source_case = "case2" if args.case == "case3" else args.case
    cfg = build_args(source_case, raw)
    cfg.case_3d = args.case
    cfg.defender_sensor_delay_steps = args.sensor_delay_steps
    cfg.defender_obs_pos_noise_std = args.position_noise_std
    cfg.defender_obs_vel_noise_std = args.velocity_noise_std
    cfg.defender_command_lag_tau = args.command_lag_tau
    cfg.trust_pn_navigation_constant = float(
        getattr(args, "pn_navigation_constant", 3.0)
    )
    cfg.paper_attacker_replay = 1
    cfg.attack_pattern = args.attack_pattern
    cfg.unseen_chirp_rate = args.unseen_chirp_rate
    cfg.unseen_frequency_scale = args.unseen_frequency_scale
    cfg.unseen_vertical_load_amp = args.unseen_vertical_load_amp
    if args.cpu_eval:
        cfg.cuda = False
    env, policy, device, selected_model = collect_model(
        cfg, Path(args.model_dir)
    )
    return cfg, env, policy, device, selected_model


def evaluate(args):
    cfg, env, policy, device, selected_model = (
        build_environment_and_policy(args)
    )
    defenders = [a for a in env.world.agents if a.adversary]
    n_agents = len(defenders)
    if n_agents != 20:
        raise RuntimeError(f"expected 20 defenders, got {n_agents}")
    dt = float(env.dt)
    terminal_steps = max(1, int(round(args.terminal_window_s / dt)))
    episode_rows: list[dict] = []
    target_rows: list[dict] = []
    assignment_rows: list[dict] = []

    for episode in range(args.episodes):
        episode_seed = int(args.seed + episode)
        episode_id = int(args.episode_offset + episode + 1)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        env.seed(episode_seed)
        obs = np.asarray(env.reset(), dtype=np.float32)
        perturb = perturb_initial_state(
            env, episode_seed, args.initial_perturbation_scale
        )
        case3_geometry = (
            configure_case3_initial_state(env, episode_seed)
            if args.case == "case3"
            else {}
        )

        if args.assignment_mode == "idbo":
            assignment, assignment_info = solve_idbo_assignment(
                env,
                episode_seed,
                args.idbo_population,
                args.idbo_iterations,
            )
        elif args.assignment_mode == "fixed":
            assignment = PAPER_ASSIGNMENT.copy()
            assignment_info = {
                "idbo_best_cost": float("nan"),
                "idbo_repaired_cost": float("nan"),
                "idbo_runtime_ms": 0.0,
                "idbo_final_population_cost": float("nan"),
                "idbo_final_disagreement": float("nan"),
                "repair_moves": 0,
                "raw_counts": np.bincount(
                    assignment, minlength=8
                ).tolist(),
                "final_counts": np.bincount(
                    assignment, minlength=8
                ).tolist(),
            }
        else:
            raise ValueError(args.assignment_mode)
        obs = apply_assignment(env, assignment)

        for i, target_index in enumerate(assignment):
            assignment_rows.append(
                {
                    "condition": args.condition,
                    "case": args.case,
                    "variant": args.variant,
                    "episode": episode_id,
                    "seed": episode_seed,
                    "assignment_mode": args.assignment_mode,
                    "defender_id": i,
                    "target_index": int(target_index),
                    "target_id": int(20 + target_index),
                    "idbo_best_cost": assignment_info["idbo_best_cost"],
                    "idbo_repaired_cost": assignment_info[
                        "idbo_repaired_cost"
                    ],
                    "idbo_runtime_ms": assignment_info["idbo_runtime_ms"],
                    "idbo_final_disagreement": assignment_info[
                        "idbo_final_disagreement"
                    ],
                    "repair_moves": assignment_info["repair_moves"],
                }
            )

        rnn_states = np.zeros(
            (n_agents, cfg.recurrent_N, cfg.hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((n_agents, 1), dtype=np.float32)
        hit_time = np.full(n_agents, np.nan, dtype=float)
        hit_step = np.full(n_agents, -1, dtype=int)
        hit_distance = np.full(n_agents, np.nan, dtype=float)
        min_distance = np.full(n_agents, np.inf, dtype=float)
        load_history: list[np.ndarray] = []
        reward_sum = np.zeros(n_agents, dtype=float)
        simulated_steps = 0

        for step in range(args.max_steps):
            with torch.no_grad():
                actions, rnn_states_t = policy.act(
                    obs,
                    rnn_states,
                    masks,
                    deterministic=not args.stochastic_eval,
                )
            rnn_states = rnn_states_t.detach().cpu().numpy()
            actions_np = actions.detach().cpu().numpy()
            actions_np[..., 0] = np.clip(actions_np[..., 0], -0.1, 1.0)
            actions_np[..., 1:] = np.clip(actions_np[..., 1:], -1.0, 1.0)
            obs_next, rewards, dones, infos = env.step(actions_np, step)
            obs = np.asarray(obs_next, dtype=np.float32)
            reward_array = np.asarray(rewards, dtype=float).reshape(
                n_agents, -1
            )
            reward_sum += reward_array.sum(axis=1)
            dones_array = np.asarray(dones, dtype=bool)
            masks = np.ones_like(masks)
            masks[dones_array, 0] = 0.0
            rnn_states[dones_array] = 0.0
            load_history.append(
                np.asarray([a.state.load.copy() for a in defenders])
            )
            simulated_steps = step + 1

            for i, agent in enumerate(defenders):
                target = env.world.agents[agent.target]
                distance = float(
                    np.linalg.norm(
                        target.state.p_pos - agent.state.p_pos
                    )
                )
                min_distance[i] = min(min_distance[i], distance)
                if (
                    getattr(agent.state, "actual_hit", False)
                    and not np.isfinite(hit_time[i])
                ):
                    hit_time[i] = float(agent.state.hit_time)
                    hit_step[i] = step
                    hit_distance[i] = distance
            if np.all(dones_array):
                break

        loads = np.asarray(load_history, dtype=float)
        target_covered = []
        group_complete = []
        group_coordinated = []
        per_target_metrics = []
        for target_index in range(8):
            indices = np.where(assignment == target_index)[0]
            hits = np.isfinite(hit_time[indices])
            covered = bool(np.any(hits))
            complete = bool(np.all(hits))
            target_covered.append(covered)
            group_complete.append(complete)
            if complete:
                times = hit_time[indices]
                mean_time = float(np.mean(times))
                co_time = float(np.mean(np.abs(times - mean_time)))
                spread = float(np.max(times) - np.min(times))
                e_miss = float(np.mean(hit_distance[indices]))
                e_t = float(np.max(times))
                terminal_load_values = []
                for defender_index in indices:
                    stop = int(hit_step[defender_index]) + 1
                    start = max(0, stop - terminal_steps)
                    normal_load = np.sqrt(
                        np.square(loads[start:stop, defender_index, 1])
                        + np.square(loads[start:stop, defender_index, 2])
                    )
                    terminal_load_values.append(float(np.mean(normal_load)))
                e_n = float(np.mean(terminal_load_values))
                coordinated = spread <= args.sync_tol
            else:
                co_time = float("nan")
                spread = float("nan")
                e_miss = float("nan")
                e_t = float("nan")
                e_n = float("nan")
                coordinated = False
            group_coordinated.append(bool(complete and coordinated))
            target_row = {
                "condition": args.condition,
                "case": args.case,
                "variant": args.variant,
                "episode": episode_id,
                "seed": episode_seed,
                "target_index": target_index,
                "group_size": int(indices.size),
                "hit_count": int(np.count_nonzero(hits)),
                "target_covered": int(covered),
                "group_complete": int(complete),
                "group_coordinated": int(complete and coordinated),
                "arrival_spread_s": spread,
                "E_co_time_s": co_time,
                "E_n_g": e_n,
                "E_miss_m": e_miss,
                "E_t_s": e_t,
                "closest_approach_m": float(
                    np.mean(min_distance[indices])
                ),
            }
            target_rows.append(target_row)
            per_target_metrics.append(target_row)

        coverage_success = bool(np.all(target_covered))
        all_defenders_hit = bool(np.all(group_complete))
        cooperative_success = bool(np.all(group_coordinated))
        mission_success = bool(coverage_success and cooperative_success)
        if not coverage_success:
            failure_class = "unsuccessful_interception"
        elif not all_defenders_hit:
            failure_class = "incomplete_cooperative_group"
        elif not cooperative_success:
            failure_class = "delayed_cooperative_engagement"
        else:
            failure_class = "mission_success"

        successful_metrics = (
            per_target_metrics if all_defenders_hit else []
        )
        row = {
            "condition": args.condition,
            "case": args.case,
            "variant": args.variant,
            "episode": episode_id,
            "seed": episode_seed,
            "attack_pattern": args.attack_pattern,
            "assignment_mode": args.assignment_mode,
            "target_coverage_success": int(coverage_success),
            "all_defenders_hit": int(all_defenders_hit),
            "cooperative_success": int(cooperative_success),
            "mission_success": int(mission_success),
            "failure_class": failure_class,
            "targets_covered": int(np.count_nonzero(target_covered)),
            "complete_groups": int(np.count_nonzero(group_complete)),
            "coordinated_groups": int(
                np.count_nonzero(group_coordinated)
            ),
            "E_co_time_s": (
                float(
                    np.mean(
                        [m["E_co_time_s"] for m in successful_metrics]
                    )
                )
                if successful_metrics
                else float("nan")
            ),
            "E_n_g": (
                float(np.mean([m["E_n_g"] for m in successful_metrics]))
                if successful_metrics
                else float("nan")
            ),
            "E_miss_m": (
                float(
                    np.mean([m["E_miss_m"] for m in successful_metrics])
                )
                if successful_metrics
                else float("nan")
            ),
            "E_t_s": (
                float(np.mean([m["E_t_s"] for m in successful_metrics]))
                if successful_metrics
                else float("nan")
            ),
            "mean_closest_approach_m": float(np.mean(min_distance)),
            "worst_closest_approach_m": float(np.max(min_distance)),
            "mean_agent_return": float(np.mean(reward_sum)),
            "simulated_time_s": simulated_steps * dt,
            "sensor_delay_ms": args.sensor_delay_steps * dt * 1000.0,
            "position_noise_std_m": args.position_noise_std,
            "velocity_noise_std_mps": args.velocity_noise_std,
            "command_lag_ms": args.command_lag_tau * 1000.0,
            "initial_perturbation_scale": args.initial_perturbation_scale,
            **perturb,
            "idbo_runtime_ms": assignment_info["idbo_runtime_ms"],
            "idbo_best_cost": assignment_info["idbo_best_cost"],
            "idbo_repaired_cost": assignment_info[
                "idbo_repaired_cost"
            ],
            "idbo_final_disagreement": assignment_info[
                "idbo_final_disagreement"
            ],
            "assignment_repair_moves": assignment_info["repair_moves"],
            **case3_geometry,
        }
        episode_rows.append(row)
        if (episode + 1) % 10 == 0 or episode == 0:
            print(
                f"[{args.condition}] {args.case} "
                f"episode={episode + 1}/{args.episodes} "
                f"coverage={np.mean([r['target_coverage_success'] for r in episode_rows]):.3f} "
                f"mission={np.mean([r['mission_success'] for r in episode_rows]):.3f}",
                flush=True,
            )

    env.close()
    outdir = Path(args.outdir)
    _write_csv(outdir / "episodes.csv", episode_rows)
    _write_csv(outdir / "targets.csv", target_rows)
    _write_csv(outdir / "assignments.csv", assignment_rows)

    summary = {
        "condition": args.condition,
        "case": args.case,
        "episodes": args.episodes,
        "variant": args.variant,
        "model_dir": str(selected_model),
        "attack_pattern": args.attack_pattern,
        "assignment_mode": args.assignment_mode,
        "case3_source_policy_case": (
            "case2" if args.case == "case3" else None
        ),
        "case3_geometry": (
            "wider staggered-altitude shell with biased inbound headings"
            if args.case == "case3"
            else None
        ),
        "case3_maneuver": (
            "heterogeneous multisine-to-jink-to-chirp switching with axial pulses"
            if args.case == "case3"
            else None
        ),
        "training_performed": False,
        "optimizer_steps": 0,
        "backpropagation_performed": False,
        "evaluation_device": str(device),
    }
    for metric in [
        "target_coverage_success",
        "all_defenders_hit",
        "cooperative_success",
        "mission_success",
    ]:
        count = int(sum(int(row[metric]) for row in episode_rows))
        low, high = _wilson_interval(count, args.episodes)
        summary[f"{metric}_count"] = count
        summary[f"{metric}_rate"] = count / args.episodes
        summary[f"{metric}_ci95_low"] = low
        summary[f"{metric}_ci95_high"] = high
    for metric in [
        "E_co_time_s",
        "E_n_g",
        "E_miss_m",
        "E_t_s",
        "mean_closest_approach_m",
        "worst_closest_approach_m",
        "mean_agent_return",
        "idbo_runtime_ms",
        "idbo_repaired_cost",
    ]:
        mean, low, high, n = _mean_ci([row[metric] for row in episode_rows])
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_ci95_low"] = low
        summary[f"{metric}_ci95_high"] = high
        summary[f"{metric}_n"] = n
    class_counts = {}
    for row in episode_rows:
        name = row["failure_class"]
        class_counts[name] = class_counts.get(name, 0) + 1
    summary["failure_class_counts"] = class_counts

    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    _write_csv(
        outdir / "summary.csv",
        [
            {
                key: (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in summary.items()
            }
        ],
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument(
        "--case",
        choices=["case1", "case2", "case3"],
        required=True,
    )
    parser.add_argument(
        "--variant",
        choices=[
            "full",
            "no_trust",
            "no_gru",
            "no_attention_residual",
        ],
        default="full",
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--episode_offset",
        type=int,
        default=0,
        help="Global episode-number offset used by the parallel evaluator.",
    )
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=91000)
    parser.add_argument("--sync_tol", type=float, default=0.5)
    parser.add_argument("--terminal_window_s", type=float, default=1.0)
    parser.add_argument("--stochastic_eval", action="store_true")
    parser.add_argument(
        "--cpu_eval",
        action="store_true",
        help="Use CPU inference so independent evaluation episodes can run in parallel.",
    )
    parser.add_argument(
        "--attack_pattern",
        choices=[
            "nominal",
            "chirp",
            "multisine",
            "jink",
            "case3_hybrid",
        ],
        default="nominal",
    )
    parser.add_argument("--unseen_chirp_rate", type=float, default=0.004)
    parser.add_argument("--unseen_frequency_scale", type=float, default=1.35)
    parser.add_argument("--unseen_vertical_load_amp", type=float, default=0.18)
    parser.add_argument("--sensor_delay_steps", type=int, default=1)
    parser.add_argument("--position_noise_std", type=float, default=3.0)
    parser.add_argument("--velocity_noise_std", type=float, default=0.3)
    parser.add_argument("--command_lag_tau", type=float, default=0.30)
    parser.add_argument(
        "--assignment_mode", choices=["fixed", "idbo"], default="fixed"
    )
    parser.add_argument("--initial_perturbation_scale", type=float, default=0.0)
    parser.add_argument("--idbo_population", type=int, default=30)
    parser.add_argument("--idbo_iterations", type=int, default=80)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.episode_offset < 0:
        parser.error("--episode_offset must be nonnegative")
    if args.sensor_delay_steps < 0:
        parser.error("--sensor_delay_steps must be nonnegative")
    if args.case == "case3" and args.attack_pattern != "case3_hybrid":
        parser.error(
            "Case 3 requires --attack_pattern case3_hybrid"
        )
    if args.case == "case3" and args.assignment_mode != "idbo":
        parser.error(
            "Case 3 is an end-to-end test and requires --assignment_mode idbo"
        )
    return args


if __name__ == "__main__":
    evaluate(parse_args())
