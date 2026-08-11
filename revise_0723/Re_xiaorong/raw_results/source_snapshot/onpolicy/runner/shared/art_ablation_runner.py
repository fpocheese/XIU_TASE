"""Training runner for the ART-MAPPO component ablation.

The runner keeps the environment, reward, optimizer, rollout budget, and seed
fixed across variants. Trust-guided actions are sampled from the manuscript's
mixture only during training. Following Algorithm 1 and the manuscript's PPO
ratio definition, the executed action and its likelihood under the rollout
policy are stored for every transition. Deployment always uses the learned
policy alone.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy_art import (
    R_MAPPOPolicy_ART,
)
from onpolicy.algorithms.r_mappo.r_mappo_art import R_MAPPO_ART
from onpolicy.utils.shared_buffer import SharedReplayBuffer


def _t2n(value):
    return value.detach().cpu().numpy()


class ARTMAPPOAblationRunner:
    def __init__(self, config):
        self.args = config["all_args"]
        self.envs = config["envs"]
        self.device = config["device"]
        self.num_agents = int(config["num_agents"])
        self.run_dir = Path(config["run_dir"])
        self.variant = str(self.args.ablation_variant)
        self.use_trust = bool(self.args.art_use_trust)
        self.n_threads = int(self.args.n_rollout_threads)
        self.episode_length = int(self.args.episode_length)
        self.hidden_size = int(self.args.hidden_size)
        self.recurrent_N = int(self.args.recurrent_N)
        self.gamma = float(self.args.gamma)
        self.total_steps = int(self.args.num_env_steps)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = self.run_dir / "models"
        self.model_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(str(self.run_dir / "logs"))

        obs_space = self.envs.observation_space[0]
        share_space = self.envs.share_observation_space[0]
        act_space = self.envs.action_space[0]
        self.policy = R_MAPPOPolicy_ART(
            self.args, obs_space, share_space, act_space, self.device
        )
        self.trainer = R_MAPPO_ART(self.args, self.policy, self.device)
        self.buffer = SharedReplayBuffer(
            self.args,
            self.num_agents,
            obs_space,
            share_space,
            act_space,
        )
        # FixedNormal.log_probs sums the three action dimensions and returns
        # one joint log likelihood.  The stock buffer allocates this field
        # using action dimensionality, which would silently broadcast the
        # scalar log likelihood three times.
        self.buffer.action_log_probs = np.zeros(
            (
                self.episode_length,
                self.n_threads,
                self.num_agents,
                1,
            ),
            dtype=np.float32,
        )

        self.num_updates = max(
            1,
            self.total_steps // self.episode_length // self.n_threads,
        )
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.policy.actor_optimizer,
            T_max=self.num_updates,
            eta_min=float(self.args.art_min_lr),
        )
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.policy.critic_optimizer,
            T_max=self.num_updates,
            eta_min=float(self.args.art_min_lr),
        )

        self.rng = np.random.RandomState(int(self.args.seed) + 9173)
        self.trust = np.full(
            self.num_agents, float(self.args.trust_initial), dtype=np.float64
        )
        self.return_mean = 0.0
        self.return_var = 1.0
        self.trust_stats_initialized = False
        self.guided_count = 0
        self.action_count = 0
        self.next_pn = np.zeros(
            (self.n_threads, self.num_agents, 3), dtype=np.float32
        )
        self.next_probe = np.zeros_like(self.next_pn)
        self.start_update = 0
        self.metrics_path = self.run_dir / "training_metrics.csv"
        self._write_manifest()
        if bool(getattr(self.args, "resume", False)):
            self._restore_latest()

    def _write_manifest(self):
        manifest = {
            "variant": self.variant,
            "seed": int(self.args.seed),
            "case": str(self.args.case_3d),
            "components": {
                "trust_aware": bool(self.args.art_use_trust),
                "gru": bool(self.args.art_use_gru),
                "attention_residual": bool(
                    self.args.art_use_attention_residual
                ),
            },
            "parameter_count": {
                "actor": int(
                    sum(p.numel() for p in self.policy.actor.parameters())
                ),
                "critic": int(
                    sum(p.numel() for p in self.policy.critic.parameters())
                ),
            },
            "trust_hyperparameters": {
                "initial": float(self.args.trust_initial),
                "rho": float(self.args.trust_rho),
                "alpha": float(self.args.trust_alpha),
                "tau": float(self.args.trust_tau),
                "guide_weights": [
                    float(self.args.trust_omega_pn),
                    float(self.args.trust_omega_probe),
                    float(self.args.trust_omega_random),
                ],
            },
            "optimization": {
                "num_env_steps": self.total_steps,
                "episode_length": self.episode_length,
                "rollout_threads": self.n_threads,
                "learning_rate": float(self.args.lr),
                "min_learning_rate": float(self.args.art_min_lr),
                "ppo_clip": float(self.args.clip_param),
                "dual_clip": float(self.args.dual_clip_param),
                "log_ratio_clip": float(
                    getattr(self.args, "log_ratio_clip", 20.0)
                ),
                "target_kl": float(self.args.target_kl),
                "gae_lambda": float(self.args.gae_lambda),
                "mini_batches": int(self.args.num_mini_batch),
                "ppo_epochs": int(self.args.ppo_epoch),
            },
            "paper_guidance_profile": {
                "base_gain": float(
                    self.args.defender_guidance_base_gain
                ),
                "tau": float(self.args.defender_guidance_tau),
                "lead": float(self.args.defender_guidance_lead),
                "command_lag_tau": float(
                    self.args.defender_command_lag_tau
                ),
                "residual_scale": float(
                    self.args.defender_residual_scale
                ),
            },
        }
        with open(
            self.run_dir / "run_manifest.json", "w", encoding="utf-8"
        ) as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

    def _share_obs(self, obs):
        joint = obs.reshape(self.n_threads, -1)
        return np.repeat(joint[:, None, :], self.num_agents, axis=1)

    def _warmup(self):
        obs = np.asarray(self.envs.reset(), dtype=np.float32)
        self.buffer.obs[0] = obs
        self.buffer.share_obs[0] = self._share_obs(obs)
        self.buffer.rnn_states[0].fill(0.0)
        self.buffer.rnn_states_critic[0].fill(0.0)
        self.buffer.masks[0].fill(1.0)
        self.next_pn.fill(0.0)
        self.next_probe.fill(0.0)
        return obs

    @torch.no_grad()
    def _collect_policy(self, step):
        self.trainer.prep_rollout()
        value, action, log_prob, rnn_actor, rnn_critic = (
            self.policy.get_actions(
                np.concatenate(self.buffer.share_obs[step]),
                np.concatenate(self.buffer.obs[step]),
                np.concatenate(self.buffer.rnn_states[step]),
                np.concatenate(self.buffer.rnn_states_critic[step]),
                np.concatenate(self.buffer.masks[step]),
            )
        )
        split = lambda x: np.asarray(
            np.split(_t2n(x), self.n_threads), dtype=np.float32
        )
        return (
            split(value),
            split(action),
            split(log_prob),
            split(rnn_actor),
            split(rnn_critic),
        )

    def _apply_trust_mixture(self, policy_actions):
        if not self.use_trust:
            policy_mask = np.ones(
                (self.n_threads, self.num_agents, 1), dtype=np.float32
            )
            return policy_actions.copy(), policy_mask

        beta = np.clip(1.0 - self.trust, 0.0, 1.0)
        guided = self.rng.random((self.n_threads, self.num_agents)) < beta[None, :]
        guide_choice = self.rng.choice(
            3,
            size=(self.n_threads, self.num_agents),
            p=[
                float(self.args.trust_omega_pn),
                float(self.args.trust_omega_probe),
                float(self.args.trust_omega_random),
            ],
        )
        actions_env = policy_actions.copy()
        uniform = np.empty_like(actions_env)
        uniform[..., 0] = self.rng.uniform(
            -0.1, 1.0, size=uniform[..., 0].shape
        )
        uniform[..., 1:] = self.rng.uniform(
            -1.0, 1.0, size=uniform[..., 1:].shape
        )
        for choice, guide in (
            (0, self.next_pn),
            (1, self.next_probe),
            (2, uniform),
        ):
            selected = guided & (guide_choice == choice)
            actions_env[selected] = guide[selected]
        # Algorithm 1 stores every mixture transition, and the PPO equation
        # evaluates its executed action under the rollout policy.
        policy_mask = np.ones(
            (self.n_threads, self.num_agents, 1), dtype=np.float32
        )
        self.guided_count += int(guided.sum())
        self.action_count += int(guided.size)
        return actions_env, policy_mask

    @torch.no_grad()
    def _executed_action_log_probs(self, step, actions_env):
        active_masks = np.ones(
            (self.n_threads, self.num_agents, 1), dtype=np.float32
        )
        action_log_probs, _ = self.policy.actor.evaluate_actions(
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(actions_env),
            np.concatenate(self.buffer.masks[step]),
            None,
            np.concatenate(active_masks),
        )
        return np.asarray(
            np.split(_t2n(action_log_probs), self.n_threads),
            dtype=np.float32,
        )

    def _extract_guides_and_events(self, infos, hit_times):
        for thread in range(self.n_threads):
            for agent in range(self.num_agents):
                try:
                    info = infos[thread][agent]
                except Exception:
                    continue
                if not isinstance(info, dict):
                    continue
                pn = np.asarray(info.get("pn_action", np.zeros(3)), dtype=np.float32)
                probe = np.asarray(
                    info.get("probe_action", np.zeros(3)), dtype=np.float32
                )
                if pn.shape == (3,) and np.isfinite(pn).all():
                    self.next_pn[thread, agent] = pn
                if probe.shape == (3,) and np.isfinite(probe).all():
                    self.next_probe[thread, agent] = probe
                hit_time = float(info.get("hit_time", np.nan))
                if bool(info.get("actual_hit", False)) and np.isfinite(hit_time):
                    hit_times[thread, agent] = hit_time

    def _update_trust(self, discounted_returns):
        if not self.use_trust:
            return
        per_agent_return = np.mean(discounted_returns, axis=0)
        batch_mean = float(np.mean(per_agent_return))
        batch_var = float(np.mean((per_agent_return - batch_mean) ** 2))
        rho = float(self.args.trust_rho)
        if not self.trust_stats_initialized:
            self.return_mean = batch_mean
            self.return_var = max(batch_var, float(self.args.trust_epsilon))
            self.trust_stats_initialized = True
        else:
            self.return_mean = (
                (1.0 - rho) * self.return_mean + rho * batch_mean
            )
            self.return_var = (
                (1.0 - rho) * self.return_var + rho * batch_var
            )
        standardized = (per_agent_return - self.return_mean) / np.sqrt(
            self.return_var + float(self.args.trust_epsilon)
        )
        target = 1.0 / (
            1.0 + np.exp(-float(self.args.trust_tau) * standardized)
        )
        alpha = float(self.args.trust_alpha)
        self.trust = (1.0 - alpha) * self.trust + alpha * target
        self.trust = np.clip(self.trust, 0.0, 1.0)

    def _episode_metrics(self, hit_times):
        assignment = np.array(
            [20, 21, 22, 23, 24, 25, 26, 27,
             20, 21, 22, 23, 24, 25, 26, 27,
             20, 21, 22, 23],
            dtype=int,
        )
        all_hit = []
        all_sync = []
        mean_spreads = []
        for thread in range(self.n_threads):
            hits = np.isfinite(hit_times[thread])
            all_hit.append(float(np.all(hits)))
            spreads = []
            synced = True
            for target in np.unique(assignment):
                idx = np.where(assignment == target)[0]
                if not np.all(hits[idx]):
                    synced = False
                    continue
                spread = float(
                    np.max(hit_times[thread, idx])
                    - np.min(hit_times[thread, idx])
                )
                spreads.append(spread)
                if spread > float(self.args.reward_sync_tol):
                    synced = False
            all_sync.append(float(np.all(hits) and synced))
            mean_spreads.append(
                float(np.mean(spreads)) if spreads else np.nan
            )
        return (
            float(np.mean(all_hit)),
            float(np.mean(all_sync)),
            float(np.nanmean(mean_spreads))
            if np.isfinite(mean_spreads).any()
            else np.nan,
        )

    def _append_metrics(self, row):
        fieldnames = list(row.keys())
        write_header = not self.metrics_path.exists()
        with open(
            self.metrics_path, "a", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    @torch.no_grad()
    def _compute_returns(self):
        self.trainer.prep_rollout()
        next_values = self.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]),
        )
        next_values = np.asarray(
            np.split(_t2n(next_values), self.n_threads),
            dtype=np.float32,
        )
        self.buffer.compute_returns(
            next_values, self.trainer.value_normalizer
        )

    def _checkpoint_payload(self, next_update):
        return {
            "next_update": int(next_update),
            "actor": self.policy.actor.state_dict(),
            "critic": self.policy.critic.state_dict(),
            "actor_optimizer": self.policy.actor_optimizer.state_dict(),
            "critic_optimizer": self.policy.critic_optimizer.state_dict(),
            "actor_scheduler": self.actor_scheduler.state_dict(),
            "critic_scheduler": self.critic_scheduler.state_dict(),
            "value_normalizer": (
                self.trainer.value_normalizer.state_dict()
                if self.trainer.value_normalizer is not None
                else None
            ),
            "trainer_current_episode": int(self.trainer.current_episode),
            "kl_coef": float(self.trainer.kl_coef),
            "trust": self.trust,
            "return_mean": float(self.return_mean),
            "return_var": float(self.return_var),
            "trust_stats_initialized": bool(self.trust_stats_initialized),
            "numpy_rng": self.rng.get_state(),
            "torch_rng": torch.get_rng_state(),
            "torch_cuda_rng": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        }

    def _save_checkpoint(self, next_update, named=False):
        payload = self._checkpoint_payload(next_update)
        tmp_path = self.model_dir / "checkpoint_latest.pt.tmp"
        final_path = self.model_dir / "checkpoint_latest.pt"
        torch.save(payload, str(tmp_path))
        os.replace(str(tmp_path), str(final_path))
        if named:
            torch.save(
                payload,
                str(self.model_dir / f"checkpoint_update_{next_update:04d}.pt"),
            )
        torch.save(
            self.policy.actor.state_dict(), str(self.model_dir / "actor.pt")
        )
        torch.save(
            self.policy.critic.state_dict(), str(self.model_dir / "critic.pt")
        )

    def _restore_latest(self):
        path = self.model_dir / "checkpoint_latest.pt"
        if not path.exists():
            return
        payload = torch.load(str(path), map_location=self.device)
        self.policy.actor.load_state_dict(payload["actor"])
        self.policy.critic.load_state_dict(payload["critic"])
        self.policy.actor_optimizer.load_state_dict(
            payload["actor_optimizer"]
        )
        self.policy.critic_optimizer.load_state_dict(
            payload["critic_optimizer"]
        )
        self.actor_scheduler.load_state_dict(payload["actor_scheduler"])
        self.critic_scheduler.load_state_dict(payload["critic_scheduler"])
        if (
            self.trainer.value_normalizer is not None
            and payload.get("value_normalizer") is not None
        ):
            self.trainer.value_normalizer.load_state_dict(
                payload["value_normalizer"]
            )
        self.trainer.current_episode = int(
            payload.get("trainer_current_episode", 0)
        )
        self.trainer.kl_coef = float(
            payload.get("kl_coef", self.trainer.kl_coef)
        )
        self.trust = np.asarray(payload["trust"], dtype=np.float64)
        self.return_mean = float(payload["return_mean"])
        self.return_var = float(payload["return_var"])
        self.trust_stats_initialized = bool(
            payload.get("trust_stats_initialized", True)
        )
        self.rng.set_state(payload["numpy_rng"])
        # CPU RNG state must remain a CPU ByteTensor even when the rest of the
        # checkpoint is mapped to CUDA.
        torch.set_rng_state(payload["torch_rng"].cpu())
        if (
            torch.cuda.is_available()
            and payload.get("torch_cuda_rng") is not None
        ):
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in payload["torch_cuda_rng"]]
            )
        self.start_update = int(payload["next_update"])

    def run(self):
        start_time = time.time()
        for update in range(self.start_update, self.num_updates):
            self._warmup()
            discounted_returns = np.zeros(
                (self.n_threads, self.num_agents), dtype=np.float64
            )
            hit_times = np.full(
                (self.n_threads, self.num_agents), np.nan, dtype=np.float64
            )
            episode_reward = 0.0
            discount = 1.0
            guided_before = self.guided_count
            actions_before = self.action_count

            for step in range(self.episode_length):
                values, policy_actions, log_probs, rs, rsc = (
                    self._collect_policy(step)
                )
                actions_env, policy_mask = self._apply_trust_mixture(
                    policy_actions
                )
                if self.use_trust:
                    log_probs = self._executed_action_log_probs(
                        step, actions_env
                    )
                # Correct temporal alignment: this mask belongs to the action
                # sampled at ``step``.  The stock buffer's active-mask insert
                # is next-state aligned, so assign it explicitly here.
                self.buffer.active_masks[step] = policy_mask
                obs, rewards, dones, infos = self.envs.step(actions_env)
                obs = np.asarray(obs, dtype=np.float32)
                rewards = np.asarray(rewards, dtype=np.float32)
                if rewards.ndim == 2:
                    rewards = rewards[..., None]
                dones = np.asarray(dones, dtype=bool)
                episode_reward += float(np.mean(rewards))
                discounted_returns += discount * rewards[..., 0]
                discount *= self.gamma
                self._extract_guides_and_events(infos, hit_times)

                rs[dones] = 0.0
                rsc[dones] = 0.0
                masks = np.ones(
                    (self.n_threads, self.num_agents, 1), dtype=np.float32
                )
                masks[dones] = 0.0
                self.buffer.insert(
                    self._share_obs(obs),
                    obs,
                    rs,
                    rsc,
                    actions_env,
                    log_probs,
                    values,
                    rewards,
                    masks,
                )

            self._compute_returns()
            self.trainer.prep_training()
            train_info = self.trainer.train(self.buffer)
            self.buffer.after_update()
            self.actor_scheduler.step()
            self.critic_scheduler.step()
            self._update_trust(discounted_returns)
            all_hit_rate, all_sync_rate, sync_spread = (
                self._episode_metrics(hit_times)
            )

            total_steps = (
                (update + 1) * self.episode_length * self.n_threads
            )
            guided_delta = self.guided_count - guided_before
            action_delta = self.action_count - actions_before
            row = {
                "update": update + 1,
                "environment_steps": total_steps,
                "mean_episode_return": episode_reward,
                "all_hit_rate": all_hit_rate,
                "all_sync_rate": all_sync_rate,
                "mean_sync_spread_s": sync_spread,
                "trust_mean": float(np.mean(self.trust)),
                "trust_std": float(np.std(self.trust)),
                "guided_action_fraction": (
                    float(guided_delta / action_delta)
                    if action_delta
                    else 0.0
                ),
                "policy_loss": float(train_info["policy_loss"]),
                "value_loss": float(train_info["value_loss"]),
                "entropy": float(train_info["dist_entropy"]),
                "approx_kl": float(train_info["kl_divergence"]),
                "actor_lr": float(
                    self.policy.actor_optimizer.param_groups[0]["lr"]
                ),
                "wall_time_s": float(time.time() - start_time),
            }
            self._append_metrics(row)
            for key, value in row.items():
                if key not in {"update", "environment_steps"} and np.isfinite(value):
                    self.writer.add_scalar(key, value, total_steps)

            if (
                (update + 1) % int(self.args.save_interval) == 0
                or update + 1 == self.num_updates
            ):
                self._save_checkpoint(update + 1, named=True)
            elif (update + 1) % int(self.args.checkpoint_interval) == 0:
                self._save_checkpoint(update + 1, named=False)

            if (
                (update + 1) % int(self.args.log_interval) == 0
                or update == self.start_update
            ):
                fps = int(total_steps / max(time.time() - start_time, 1e-6))
                print(
                    f"[{self.variant}] update={update + 1}/{self.num_updates} "
                    f"steps={total_steps} return={episode_reward:.3f} "
                    f"all_hit={all_hit_rate:.3f} all_sync={all_sync_rate:.3f} "
                    f"trust={np.mean(self.trust):.3f} "
                    f"guided={row['guided_action_fraction']:.3f} fps={fps}",
                    flush=True,
                )

        self.writer.export_scalars_to_json(
            str(self.run_dir / "logs" / "summary.json")
        )
        self.writer.close()
        return self.metrics_path
