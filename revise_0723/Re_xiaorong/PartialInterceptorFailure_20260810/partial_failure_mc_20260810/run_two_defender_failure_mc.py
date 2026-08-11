#!/usr/bin/env python3
"""Frozen-policy Monte Carlo evaluation with two unavailable defenders.

The script imports the stable evaluation stack, never updates network weights,
and writes episode-, target-, hit-, and summary-level results.  Two defenders
are sampled uniformly without replacement after the fixed paper assignment is
loaded and before the first guidance command is executed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR / "code"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "onpolicy" / "scripts"))

from eval_3d_guidance import build_args, collect_model  # noqa: E402


CASE_CONFIG = {
    "case1": {
        "seed": 89001,
        "guidance_base_gain": 2.0,
        "guidance_tau": 0.25,
        "guidance_lead": 1.60,
        "sync_speed_gain": 0.14,
        "speed_gain": 0.016,
        "command_lag_tau": 0.25,
    },
    "case2": {
        "seed": 90001,
        "guidance_base_gain": 2.6,
        "guidance_tau": 0.35,
        "guidance_lead": 1.70,
        "sync_speed_gain": 1.40,
        "speed_gain": 0.008,
        "command_lag_tau": 0.40,
    },
}


def write_csv(path: Path, rows: list[dict]) -> None:
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


def wilson_interval(successes: int, total: int, z: float = 1.9599639845):
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


def mean_ci(values):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    half = 1.9599639845 * std / math.sqrt(array.size) if array.size > 1 else 0.0
    return mean, std, mean - half, mean + half, int(array.size)


def make_raw_args(args, case_cfg):
    return SimpleNamespace(
        seed=args.seed,
        eval_episodes=args.episodes,
        max_steps=args.max_steps,
        sync_tol=args.sync_tol,
        stochastic_eval=False,
        eval_different_seed=True,
        require_success_plot=False,
        require_all_hit=False,
        hit_radius_3d=args.hit_radius,
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
        defender_sensor_delay_steps=args.sensor_delay_steps,
        defender_sensor_delay_compensate=False,
        defender_obs_pos_noise_std=args.position_noise_std,
        defender_obs_vel_noise_std=args.velocity_noise_std,
        defender_obs_filter_alpha=1.0,
        defender_command_lag_tau=case_cfg["command_lag_tau"],
        reward_w_smooth=0.0,
        reference_control_root="",
        reward_w_ref_control=0.0,
        reward_w_ref_rate=0.0,
        defender_reference_blend=0.0,
        target_assignment_mode="fixed",
        target_assignment_spread_weight=6.0,
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
        attacker_load_limit=1.75,
        attacker_yaw_scale=1.55,
        attacker_pitch_scale=1.55,
    )


def disable_defenders(env, failed_ids: np.ndarray) -> None:
    defenders = [agent for agent in env.world.agents if agent.adversary]
    for defender_id in failed_ids:
        agent = defenders[int(defender_id)]
        agent.doneflag = True
        agent.state.done = True
        agent.state.actual_hit = False
        agent.state.load = np.zeros(3, dtype=float)
        agent.action.u = np.zeros(3, dtype=float)
        agent.state.defender_command_lag_load = np.zeros(3, dtype=float)


def recompute_observations(env) -> np.ndarray:
    return np.asarray(
        [env._get_obs(agent) for agent in env.world.policy_agents],
        dtype=np.float32,
    )


def target_index(agent) -> int:
    # Paper target IDs are 20--27; retain a defensive fallback for other layouts.
    value = int(agent.target)
    return value - 20 if value >= 20 else value


def evaluate(args) -> None:
    case_cfg = CASE_CONFIG[args.case]
    raw_args = make_raw_args(args, case_cfg)
    cfg = build_args(args.case, raw_args)
    cfg.cuda = bool(torch.cuda.is_available() and not args.cpu)
    env, policy, device, selected_model = collect_model(cfg, args.model_dir)
    defenders = [agent for agent in env.world.agents if agent.adversary]
    if len(defenders) != 20:
        raise RuntimeError(f"expected 20 defenders, found {len(defenders)}")

    dt = float(env.dt)
    episode_rows: list[dict] = []
    target_rows: list[dict] = []
    hit_rows: list[dict] = []
    representative_saved = False
    args.outdir.mkdir(parents=True, exist_ok=True)

    for episode in range(args.episodes):
        episode_seed = int(args.seed + episode)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        env.seed(episode_seed)
        obs = np.asarray(env.reset(), dtype=np.float32)

        assignment = np.asarray([target_index(agent) for agent in defenders], dtype=int)
        failure_rng = np.random.default_rng(episode_seed + 1_000_003)
        failed_ids = np.sort(
            failure_rng.choice(len(defenders), size=args.failed_count, replace=False)
        ).astype(int)
        active_mask = np.ones(len(defenders), dtype=bool)
        active_mask[failed_ids] = False
        disable_defenders(env, failed_ids)
        obs = recompute_observations(env)

        n_agents = len(defenders)
        rnn_states = np.zeros(
            (n_agents, cfg.recurrent_N, cfg.hidden_size), dtype=np.float32
        )
        masks = np.ones((n_agents, 1), dtype=np.float32)
        masks[failed_ids, 0] = 0.0
        hit_time = np.full(n_agents, np.nan, dtype=float)
        hit_distance = np.full(n_agents, np.nan, dtype=float)
        terminal_ny = np.full(n_agents, np.nan, dtype=float)
        closest_distance = np.full(n_agents, np.inf, dtype=float)
        simulated_steps = 0

        trajectory_def: list[np.ndarray] = []
        trajectory_att: list[np.ndarray] = []
        trajectory_load: list[np.ndarray] = []

        for step in range(args.max_steps):
            with torch.no_grad():
                actions, next_rnn = policy.act(
                    obs, rnn_states, masks, deterministic=True
                )
            rnn_states = next_rnn.detach().cpu().numpy()
            actions_np = actions.detach().cpu().numpy()
            actions_np[..., 0] = np.clip(actions_np[..., 0], -0.1, 1.0)
            actions_np[..., 1:] = np.clip(actions_np[..., 1:], -1.0, 1.0)
            actions_np[failed_ids] = 0.0

            obs_next, _, dones, _ = env.step(actions_np, step)
            obs = np.asarray(obs_next, dtype=np.float32)
            dones_array = np.asarray(dones, dtype=bool)
            masks = np.ones_like(masks)
            masks[dones_array, 0] = 0.0
            rnn_states[dones_array] = 0.0
            simulated_steps = step + 1

            attackers = [agent for agent in env.world.agents if not agent.adversary]
            trajectory_def.append(
                np.asarray([agent.state.p_pos.copy() for agent in defenders])
            )
            trajectory_att.append(
                np.asarray([agent.state.p_pos.copy() for agent in attackers])
            )
            trajectory_load.append(
                np.asarray([agent.state.load.copy() for agent in defenders])
            )

            for defender_id, agent in enumerate(defenders):
                if not active_mask[defender_id]:
                    continue
                target = env.world.agents[agent.target]
                distance = float(
                    np.linalg.norm(target.state.p_pos - agent.state.p_pos)
                )
                closest_distance[defender_id] = min(
                    closest_distance[defender_id], distance
                )
                if (
                    getattr(agent.state, "actual_hit", False)
                    and not np.isfinite(hit_time[defender_id])
                ):
                    event_time = float(
                        agent.state.hit_time
                        if np.isfinite(agent.state.hit_time)
                        else (step + 1) * dt
                    )
                    hit_time[defender_id] = event_time
                    hit_distance[defender_id] = distance
                    terminal_ny[defender_id] = abs(float(agent.state.load[1]))
                    hit_rows.append(
                        {
                            "case": args.case,
                            "episode": episode + 1,
                            "seed": episode_seed,
                            "defender_id": defender_id,
                            "target_index": int(assignment[defender_id]),
                            "hit_time_s": event_time,
                            "terminal_abs_ny_g": terminal_ny[defender_id],
                            "hit_distance_m": distance,
                        }
                    )

            if np.all(dones_array):
                break

        target_covered = []
        active_group_complete = []
        target_coordinated = []
        target_metric_values: list[dict] = []
        for target_id in range(8):
            group = np.where((assignment == target_id) & active_mask)[0]
            hits = group[np.isfinite(hit_time[group])]
            covered = bool(hits.size > 0)
            complete = bool(group.size > 0 and hits.size == group.size)
            times = hit_time[hits]
            if times.size:
                mean_time = float(np.mean(times))
                e_co = float(np.mean(np.abs(times - mean_time)))
                spread = float(np.max(times) - np.min(times))
                e_n = float(np.mean(terminal_ny[hits]))
                e_miss = float(np.mean(hit_distance[hits]))
                e_t = float(np.max(times))
            else:
                e_co = spread = e_n = e_miss = e_t = float("nan")
            coordinated = bool(complete and spread <= args.sync_tol)
            target_covered.append(covered)
            active_group_complete.append(complete)
            target_coordinated.append(coordinated)
            target_row = {
                "case": args.case,
                "episode": episode + 1,
                "seed": episode_seed,
                "target_index": target_id,
                "assigned_defenders": int(np.count_nonzero(assignment == target_id)),
                "active_defenders": int(group.size),
                "active_hits": int(hits.size),
                "target_covered": int(covered),
                "active_group_complete": int(complete),
                "coordinated": int(coordinated),
                "arrival_spread_s": spread,
                "E_co_time_s": e_co,
                "E_n_g": e_n,
                "E_miss_m": e_miss,
                "E_t_s": e_t,
            }
            target_rows.append(target_row)
            target_metric_values.append(target_row)

        interception_success = bool(np.all(target_covered))
        all_active_hit = bool(np.all(active_group_complete))
        cooperative_success = bool(np.all(target_coordinated))

        # Match the manuscript convention: terminal metric distributions use
        # only successful interception trials.  Failed defenders are not used
        # in the four continuous metrics, but their loss can still make ISR fail.
        valid_targets = [row for row in target_metric_values if row["target_covered"]]
        if interception_success:
            e_co_episode = float(
                np.mean([row["E_co_time_s"] for row in valid_targets])
            )
            e_n_episode = float(np.mean([row["E_n_g"] for row in valid_targets]))
            e_miss_episode = float(
                np.mean([row["E_miss_m"] for row in valid_targets])
            )
            e_t_episode = float(
                np.max([row["E_t_s"] for row in valid_targets])
            )
        else:
            e_co_episode = e_n_episode = e_miss_episode = e_t_episode = float("nan")

        episode_row = {
            "case": args.case,
            "episode": episode + 1,
            "seed": episode_seed,
            "failed_count": args.failed_count,
            "failed_defender_ids": ";".join(str(int(i)) for i in failed_ids),
            "failed_target_indices": ";".join(
                str(int(assignment[i])) for i in failed_ids
            ),
            "interception_success": int(interception_success),
            "all_active_defenders_hit": int(all_active_hit),
            "cooperative_success": int(cooperative_success),
            "targets_covered": int(np.count_nonzero(target_covered)),
            "active_defenders_hit": int(np.count_nonzero(np.isfinite(hit_time) & active_mask)),
            "E_co_time_s": e_co_episode,
            "E_n_g": e_n_episode,
            "E_miss_m": e_miss_episode,
            "E_t_s": e_t_episode,
            "simulated_time_s": simulated_steps * dt,
            "sensor_delay_ms": args.sensor_delay_steps * dt * 1000.0,
            "position_noise_std_m": args.position_noise_std,
            "velocity_noise_std_mps": args.velocity_noise_std,
            "command_lag_ms": case_cfg["command_lag_tau"] * 1000.0,
        }
        episode_rows.append(episode_row)

        if interception_success and not representative_saved:
            np.savez_compressed(
                args.outdir / f"{args.case}_representative_success.npz",
                defender_positions=np.asarray(trajectory_def, dtype=np.float32),
                attacker_positions=np.asarray(trajectory_att, dtype=np.float32),
                defender_loads=np.asarray(trajectory_load, dtype=np.float32),
                assignment=assignment,
                failed_ids=failed_ids,
                hit_time=hit_time,
                seed=np.asarray([episode_seed], dtype=np.int64),
            )
            representative_saved = True

        if (episode + 1) % 10 == 0 or episode == 0:
            current_isr = float(
                np.mean([row["interception_success"] for row in episode_rows])
            )
            print(
                f"[{args.case}] episode={episode + 1}/{args.episodes} "
                f"ISR={current_isr:.3f}",
                flush=True,
            )

    env.close()
    successful_rows = [row for row in episode_rows if row["interception_success"]]
    write_csv(args.outdir / "episodes.csv", episode_rows)
    write_csv(args.outdir / "successful_metrics.csv", successful_rows)
    write_csv(args.outdir / "targets.csv", target_rows)
    write_csv(args.outdir / "hit_events.csv", hit_rows)

    isr_count = int(sum(row["interception_success"] for row in episode_rows))
    active_count = int(sum(row["all_active_defenders_hit"] for row in episode_rows))
    coop_count = int(sum(row["cooperative_success"] for row in episode_rows))
    isr_ci = wilson_interval(isr_count, args.episodes)
    active_ci = wilson_interval(active_count, args.episodes)
    coop_ci = wilson_interval(coop_count, args.episodes)
    summary = {
        "case": args.case,
        "episodes": args.episodes,
        "failed_defenders_per_episode": args.failed_count,
        "failure_selection": "uniform_without_replacement_after_assignment_before_guidance",
        "interception_success_definition": "all_eight_attackers_covered_by_operational_defenders",
        "continuous_metric_population": "interception_successful_episodes_only_operational_hits",
        "model_dir": str(selected_model),
        "policy_training_performed": False,
        "backpropagation_performed": False,
        "optimizer_steps": 0,
        "device": str(device),
        "interception_success_count": isr_count,
        "interception_success_rate": isr_count / args.episodes,
        "interception_success_ci95_low": isr_ci[0],
        "interception_success_ci95_high": isr_ci[1],
        "all_active_defenders_hit_count": active_count,
        "all_active_defenders_hit_rate": active_count / args.episodes,
        "all_active_defenders_hit_ci95_low": active_ci[0],
        "all_active_defenders_hit_ci95_high": active_ci[1],
        "cooperative_success_count": coop_count,
        "cooperative_success_rate": coop_count / args.episodes,
        "cooperative_success_ci95_low": coop_ci[0],
        "cooperative_success_ci95_high": coop_ci[1],
        "parameters": {
            "dt_s": dt,
            "hit_radius_m": args.hit_radius,
            "sync_tolerance_s": args.sync_tol,
            "sensor_delay_steps": args.sensor_delay_steps,
            "sensor_delay_ms": args.sensor_delay_steps * dt * 1000.0,
            "position_noise_std_m_per_axis": args.position_noise_std,
            "velocity_noise_std_mps_per_axis": args.velocity_noise_std,
            "command_lag_ms": case_cfg["command_lag_tau"] * 1000.0,
            **case_cfg,
        },
    }
    for metric in ["E_co_time_s", "E_n_g", "E_miss_m", "E_t_s"]:
        mean, std, low, high, count = mean_ci(
            [row[metric] for row in successful_rows]
        )
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_std"] = std
        summary[f"{metric}_ci95_low"] = low
        summary[f"{metric}_ci95_high"] = high
        summary[f"{metric}_n"] = count

    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    write_csv(
        args.outdir / "summary.csv",
        [
            {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for key, value in summary.items()
            }
        ],
    )
    manifest = {
        "script": str(Path(__file__).resolve()),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "stable_policy_sha256_note": "verified separately against stable_V2 before evaluation",
        "training_performed": False,
    }
    (args.outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASE_CONFIG), required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--failed_count", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--hit_radius", type=float, default=3.0)
    parser.add_argument("--sync_tol", type=float, default=0.5)
    parser.add_argument("--sensor_delay_steps", type=int, default=1)
    parser.add_argument("--position_noise_std", type=float, default=3.0)
    parser.add_argument("--velocity_noise_std", type=float, default=0.3)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.seed is None:
        args.seed = CASE_CONFIG[args.case]["seed"]
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if not 1 <= args.failed_count < 20:
        parser.error("--failed_count must be in [1, 19]")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
