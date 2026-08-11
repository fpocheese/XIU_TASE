#!/usr/bin/env python3
"""Audited actor initialization from the manuscript's tactical guide.

This is an explicit, separately logged behavior-cloning initialization
experiment.  It does not compute the paper reward, train a critic, or perform
PPO updates.  The resulting actor can optionally initialize a subsequent
paper-aligned ART-MAPPO run; every sample and optimizer step is recorded so the
initialization cannot be mistaken for ordinary RL experience.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch

from eval_art_mappo_ablation_3d import build_args
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy_art import (
    R_MAPPOPolicy_ART,
)
from onpolicy.envs.mpe.MPE_env import MPEEnv


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_config(args):
    raw = SimpleNamespace(
        variant=args.variant,
        seed=args.seed,
        eval_episodes=1,
        max_steps=args.max_steps,
        sync_tol=0.5,
        stochastic_eval=False,
        eval_different_seed=True,
        require_success_plot=False,
        require_all_hit=False,
        attack_pattern="nominal",
        unseen_chirp_rate=0.004,
        unseen_frequency_scale=1.35,
        unseen_vertical_load_amp=0.18,
    )
    cfg = build_args(args.case, raw)
    cfg.cuda = bool(args.cuda and torch.cuda.is_available())
    cfg.lr = args.learning_rate
    cfg.critic_lr = args.learning_rate
    cfg.opti_eps = 1e-5
    cfg.weight_decay = 0.0
    cfg.gain = 0.001
    cfg.defender_sensor_delay_steps = 1
    cfg.defender_obs_pos_noise_std = 3.0
    cfg.defender_obs_vel_noise_std = 0.3
    cfg.defender_command_lag_tau = 0.30
    cfg.paper_attacker_replay = 1
    return cfg


def pretrain(args):
    outdir = Path(args.outdir).resolve()
    model_dir = outdir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)

    cfg = make_config(args)
    device = torch.device("cuda:0" if cfg.cuda else "cpu")
    env = MPEEnv(cfg)
    env.seed(args.seed)
    policy = R_MAPPOPolicy_ART(
        cfg,
        env.observation_space[0],
        env.share_observation_space[0],
        env.action_space[0],
        device=device,
    )
    actor = policy.actor
    actor.train()
    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=args.learning_rate,
        eps=1e-5,
        weight_decay=0.0,
    )
    defenders = [agent for agent in env.world.agents if agent.adversary]
    n_agents = len(defenders)
    if n_agents != 20:
        raise RuntimeError(f"expected 20 defenders, got {n_agents}")

    rows = []
    checkpoint_dir = outdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    optimizer_steps = 0
    total_samples = 0
    execution_rng = np.random.RandomState(args.seed + 47011)
    for episode in range(args.episodes):
        episode_seed = args.seed + episode
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        env.seed(episode_seed)
        obs = np.asarray(env.reset(), dtype=np.float32)
        rnn_states = torch.zeros(
            n_agents,
            cfg.recurrent_N,
            cfg.hidden_size,
            dtype=torch.float32,
            device=device,
        )
        masks = torch.ones(
            n_agents, 1, dtype=torch.float32, device=device
        )
        loss_terms = []
        nll_values = []
        entropy_values = []
        actor_execution_count = 0
        action_execution_count = 0
        steps = 0
        if args.episodes == 1:
            actor_execution_probability = args.actor_execution_end
        else:
            fraction = episode / (args.episodes - 1)
            actor_execution_probability = (
                (1.0 - fraction) * args.actor_execution_start
                + fraction * args.actor_execution_end
            )

        for step in range(args.max_steps):
            guide = np.asarray(
                [
                    env._get_info(agent)["probe_action"]
                    for agent in defenders
                ],
                dtype=np.float32,
            )
            obs_tensor = torch.as_tensor(obs, device=device)
            guide_tensor = torch.as_tensor(guide, device=device)
            features, next_rnn = actor._features(
                obs_tensor, rnn_states, masks
            )
            log_probs, entropy = actor.act.evaluate_actions(
                features,
                guide_tensor,
                available_actions=None,
                active_masks=torch.ones_like(masks),
            )
            nll = -log_probs.mean()
            loss_terms.append(nll - args.entropy_coef * entropy)
            nll_values.append(float(nll.detach().cpu()))
            entropy_values.append(float(entropy.detach().cpu()))

            predicted, _ = actor.act(
                features, available_actions=None, deterministic=True
            )
            predicted_np = predicted.detach().cpu().numpy()
            use_actor = execution_rng.random(n_agents) < (
                actor_execution_probability
            )
            executed = guide.copy()
            executed[use_actor] = predicted_np[use_actor]
            actor_execution_count += int(use_actor.sum())
            action_execution_count += n_agents
            obs_next, _, dones, _ = env.step(executed, step)
            obs = np.asarray(obs_next, dtype=np.float32)
            dones = np.asarray(dones, dtype=bool)
            masks = torch.ones_like(masks)
            masks[
                torch.as_tensor(dones, dtype=torch.bool, device=device)
            ] = 0.0
            rnn_states = next_rnn
            steps = step + 1
            total_samples += n_agents

            boundary = (
                len(loss_terms) == args.chunk_length
                or np.all(dones)
                or step + 1 == args.max_steps
            )
            if boundary:
                loss = torch.stack(loss_terms).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), args.max_grad_norm
                )
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise FloatingPointError(
                        "non-finite behavior-cloning gradient"
                    )
                optimizer.step()
                optimizer_steps += 1
                rnn_states = rnn_states.detach()
                loss_terms.clear()
            if np.all(dones):
                break

        hit_count = sum(
            bool(getattr(agent.state, "actual_hit", False))
            for agent in defenders
        )
        row = {
            "episode": episode + 1,
            "seed": episode_seed,
            "environment_steps": steps,
            "guide_samples": steps * n_agents,
            "optimizer_steps_cumulative": optimizer_steps,
            "mean_negative_log_likelihood": float(np.mean(nll_values)),
            "mean_policy_entropy": float(np.mean(entropy_values)),
            "hit_count_under_probe_guide": (
                int(hit_count) if actor_execution_count == 0 else -1
            ),
            "all_defenders_hit_under_probe_guide": (
                int(hit_count == n_agents)
                if actor_execution_count == 0
                else -1
            ),
            "actor_execution_probability": float(
                actor_execution_probability
            ),
            "actor_execution_fraction": float(
                actor_execution_count / max(action_execution_count, 1)
            ),
            "hit_count_under_executed_mixture": int(hit_count),
            "all_defenders_hit_under_executed_mixture": int(
                hit_count == n_agents
            ),
            "ppo_updates": 0,
            "critic_updates": 0,
            "reward_used_for_initialization": False,
        }
        checkpoint_path = (
            checkpoint_dir / f"actor_episode_{episode + 1:04d}.pt"
        )
        torch.save(actor.state_dict(), checkpoint_path)
        row["actor_checkpoint"] = str(checkpoint_path)
        row["actor_checkpoint_sha256"] = sha256(checkpoint_path)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    actor_path = model_dir / "actor.pt"
    critic_path = model_dir / "critic.pt"
    torch.save(actor.state_dict(), actor_path)
    # The frozen evaluator requires both files, but the critic is not used for
    # action selection. Mark its random, untrained status in the manifest.
    torch.save(policy.critic.state_dict(), critic_path)
    write_csv(outdir / "behavior_cloning_metrics.csv", rows)
    manifest = {
        "case": args.case,
        "variant": args.variant,
        "seed_first": args.seed,
        "seed_last": args.seed + args.episodes - 1,
        "episodes": args.episodes,
        "chunk_length": args.chunk_length,
        "learning_rate": args.learning_rate,
        "entropy_coefficient": args.entropy_coef,
        "guide_source": "paper tactical boundary-probe command",
        "actor_execution_probability_start": args.actor_execution_start,
        "actor_execution_probability_end": args.actor_execution_end,
        "guide_samples": total_samples,
        "actor_optimizer_steps": optimizer_steps,
        "ppo_updates": 0,
        "critic_updates": 0,
        "reward_used_for_initialization": False,
        "critic_file_is_random_untrained": True,
        "actor_path": str(actor_path),
        "actor_sha256": sha256(actor_path),
        "critic_path": str(critic_path),
        "critic_sha256": sha256(critic_path),
    }
    (outdir / "behavior_cloning_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["case1", "case2"], required=True)
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
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--chunk_length", type=int, default=10)
    parser.add_argument("--seed", type=int, default=88020)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--entropy_coef", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument(
        "--actor_execution_start", type=float, default=0.0
    )
    parser.add_argument(
        "--actor_execution_end", type=float, default=0.0
    )
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    if (
        args.episodes < 1
        or args.max_steps < 1
        or args.chunk_length < 1
        or args.learning_rate <= 0
        or not 0.0 <= args.actor_execution_start <= 1.0
        or not 0.0 <= args.actor_execution_end <= 1.0
    ):
        parser.error("invalid behavior-cloning configuration")
    return args


if __name__ == "__main__":
    pretrain(parse_args())
