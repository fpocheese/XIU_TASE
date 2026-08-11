#!/usr/bin/env python3
"""Read-only diagnostic for policy/guide alignment.

This utility never constructs an optimizer and never changes a checkpoint.  It
compares the frozen deterministic policy with the current paper-defined PN
guide on identical reset seeds and can execute either command source to verify
whether a training failure originates in the guide or in policy transfer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_reviewer_supplementary_experiments import (
    build_environment_and_policy,
)


def run_episode(args, episode_seed: int, source: str):
    args.seed = episode_seed
    cfg, env, policy, _, selected_model = build_environment_and_policy(args)
    np.random.seed(episode_seed)
    torch.manual_seed(episode_seed)
    env.seed(episode_seed)
    obs = np.asarray(env.reset(), dtype=np.float32)
    defenders = [agent for agent in env.world.agents if agent.adversary]
    n_agents = len(defenders)
    rnn_states = np.zeros(
        (n_agents, cfg.recurrent_N, cfg.hidden_size), dtype=np.float32
    )
    masks = np.ones((n_agents, 1), dtype=np.float32)
    min_distance = np.full(n_agents, np.inf, dtype=float)
    policy_commands = []
    pn_commands = []
    probe_commands = []
    mix_rng = np.random.RandomState(episode_seed + 4817)
    mix_hold_remaining = np.zeros(n_agents, dtype=np.int32)
    mix_choice = np.zeros(n_agents, dtype=np.int8)

    for step in range(args.max_steps):
        with torch.no_grad():
            actions, next_rnn = policy.act(
                obs, rnn_states, masks, deterministic=True
            )
        policy_action = actions.detach().cpu().numpy()
        policy_action[..., 0] = np.clip(
            policy_action[..., 0], -0.1, 1.0
        )
        policy_action[..., 1:] = np.clip(
            policy_action[..., 1:], -1.0, 1.0
        )
        if np.isfinite(args.policy_axial_override):
            policy_action[..., 0] = np.clip(
                float(args.policy_axial_override), -0.1, 1.0
            )
        info = [env._get_info(agent) for agent in defenders]
        guides = np.asarray(
            [item["pn_action"] for item in info], dtype=np.float32
        )
        probes = np.asarray(
            [item["probe_action"] for item in info], dtype=np.float32
        )
        base = np.asarray(
            [env.world.defender_base_load(agent) for agent in defenders],
            dtype=np.float32,
        )
        base[:, 0] = np.clip(base[:, 0], -0.1, 1.0)
        base[:, 1:] = np.clip(base[:, 1:], -1.0, 1.0)
        policy_commands.append(policy_action.copy())
        pn_commands.append(guides.copy())
        probe_commands.append(probes.copy())
        commands = {
            "policy": policy_action,
            "pn": guides,
            "probe": probes,
            "base": base,
        }
        if source == "mix":
            refresh = mix_hold_remaining <= 0
            if np.any(refresh):
                sampled = mix_rng.choice(
                    3,
                    size=n_agents,
                    p=[
                        args.mix_omega_pn,
                        args.mix_omega_probe,
                        args.mix_omega_random,
                    ],
                )
                mix_choice[refresh] = sampled[refresh]
                mix_hold_remaining[refresh] = args.mix_hold_steps
            uniform = np.empty_like(policy_action)
            uniform[:, 0] = mix_rng.uniform(-0.1, 1.0, size=n_agents)
            uniform[:, 1:] = mix_rng.uniform(
                -1.0, 1.0, size=(n_agents, 2)
            )
            executed = policy_action.copy()
            for choice, command in (
                (0, guides),
                (1, probes),
                (2, uniform),
            ):
                selected = mix_choice == choice
                executed[selected] = command[selected]
            mix_hold_remaining -= 1
        else:
            executed = commands[source]
        obs_next, _, dones, _ = env.step(executed, step)
        obs = np.asarray(obs_next, dtype=np.float32)
        rnn_states = next_rnn.detach().cpu().numpy()
        dones = np.asarray(dones, dtype=bool)
        masks = np.ones_like(masks)
        masks[dones, 0] = 0.0
        rnn_states[dones] = 0.0
        for index, agent in enumerate(defenders):
            target = env.world.agents[agent.target]
            distance = np.linalg.norm(target.state.p_pos - agent.state.p_pos)
            min_distance[index] = min(min_distance[index], float(distance))
        if np.all(dones):
            break

    hits = np.asarray(
        [bool(getattr(agent.state, "actual_hit", False)) for agent in defenders]
    )
    policy_commands = np.asarray(policy_commands, dtype=float)
    pn_commands = np.asarray(pn_commands, dtype=float)
    probe_commands = np.asarray(probe_commands, dtype=float)
    flat_policy = policy_commands.reshape(-1, 3)
    def alignment(reference_commands):
        flat_reference = reference_commands.reshape(-1, 3)
        correlations = []
        for dimension in range(3):
            if (
                np.std(flat_policy[:, dimension]) > 0
                and np.std(flat_reference[:, dimension]) > 0
            ):
                correlations.append(
                    float(
                        np.corrcoef(
                            flat_policy[:, dimension],
                            flat_reference[:, dimension],
                        )[0, 1]
                    )
                )
            else:
                correlations.append(float("nan"))
        return {
            "mae": np.mean(
                np.abs(policy_commands - reference_commands), axis=(0, 1)
            ).tolist(),
            "correlation": correlations,
            "reference_command_mean": np.mean(
                reference_commands, axis=(0, 1)
            ).tolist(),
            "reference_command_std": np.std(
                reference_commands, axis=(0, 1)
            ).tolist(),
        }

    pn_alignment = alignment(pn_commands)
    probe_alignment = alignment(probe_commands)
    return {
        "seed": episode_seed,
        "source": source,
        "selected_model": str(selected_model),
        "steps": int(len(policy_commands)),
        "hit_count": int(hits.sum()),
        "all_defenders_hit": bool(np.all(hits)),
        "mean_closest_approach_m": float(np.mean(min_distance)),
        "worst_closest_approach_m": float(np.max(min_distance)),
        "policy_pn_mae": pn_alignment["mae"],
        "policy_pn_correlation": pn_alignment["correlation"],
        "policy_probe_mae": probe_alignment["mae"],
        "policy_probe_correlation": probe_alignment["correlation"],
        "policy_command_mean": np.mean(
            policy_commands, axis=(0, 1)
        ).tolist(),
        "policy_command_std": np.std(
            policy_commands, axis=(0, 1)
        ).tolist(),
        "pn_command_mean": pn_alignment["reference_command_mean"],
        "pn_command_std": pn_alignment["reference_command_std"],
        "probe_command_mean": probe_alignment["reference_command_mean"],
        "probe_command_std": probe_alignment["reference_command_std"],
        "training_performed": False,
        "optimizer_steps": 0,
        "backpropagation_performed": False,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["case1", "case2"], required=True)
    parser.add_argument("--variant", default="full")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=99001)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument(
        "--source",
        choices=[
            "policy",
            "pn",
            "probe",
            "base",
            "mix",
            "all",
            "both",
        ],
        default="both",
    )
    parser.add_argument("--mix_omega_pn", type=float, default=0.60)
    parser.add_argument("--mix_omega_probe", type=float, default=0.39)
    parser.add_argument("--mix_omega_random", type=float, default=0.01)
    parser.add_argument("--mix_hold_steps", type=int, default=20)
    parser.add_argument("--sync_tol", type=float, default=0.5)
    parser.add_argument("--terminal_window_s", type=float, default=1.0)
    parser.add_argument("--stochastic_eval", action="store_false")
    parser.add_argument("--cpu_eval", action="store_true", default=False)
    parser.add_argument("--attack_pattern", default="nominal")
    parser.add_argument("--unseen_chirp_rate", type=float, default=0.004)
    parser.add_argument("--unseen_frequency_scale", type=float, default=1.35)
    parser.add_argument("--unseen_vertical_load_amp", type=float, default=0.18)
    parser.add_argument("--sensor_delay_steps", type=int, default=1)
    parser.add_argument("--position_noise_std", type=float, default=3.0)
    parser.add_argument("--velocity_noise_std", type=float, default=0.3)
    parser.add_argument("--command_lag_tau", type=float, default=0.30)
    parser.add_argument(
        "--pn_navigation_constant",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--trust_los_rate_filter_alpha",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--trust_los_rate_window_steps",
        type=int,
        default=21,
    )
    parser.add_argument(
        "--policy_axial_override",
        type=float,
        default=float("nan"),
        help=(
            "Read-only diagnostic override for the frozen policy axial command; "
            "NaN leaves the learned command unchanged."
        ),
    )
    parser.add_argument("--assignment_mode", default="fixed")
    parser.add_argument("--initial_perturbation_scale", type=float, default=0.0)
    parser.add_argument("--idbo_population", type=int, default=30)
    parser.add_argument("--idbo_iterations", type=int, default=80)
    parser.add_argument("--condition", default="guide_alignment_diagnostic")
    parser.add_argument("--episode_offset", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    mix_weights = np.asarray(
        [
            args.mix_omega_pn,
            args.mix_omega_probe,
            args.mix_omega_random,
        ],
        dtype=float,
    )
    if (
        np.any(mix_weights < 0.0)
        or not np.isclose(np.sum(mix_weights), 1.0)
    ):
        raise ValueError("mix guide weights must be nonnegative and sum to one")
    if args.mix_hold_steps <= 0:
        raise ValueError("mix_hold_steps must be positive")
    first_seed = int(args.seed)
    if args.source == "all":
        sources = ["policy", "pn", "probe", "base"]
    elif args.source == "both":
        sources = ["policy", "pn"]
    else:
        sources = [args.source]
    rows = []
    for episode in range(args.episodes):
        for source in sources:
            rows.append(run_episode(args, first_seed + episode, source))
    result = {
        "rows": rows,
        "training_performed": False,
        "optimizer_steps": 0,
        "backpropagation_performed": False,
    }
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "guide_alignment.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
