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
import hashlib
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
        self.actor_init_metadata = None
        actor_init_path = getattr(self.args, "actor_init_path", None)
        if actor_init_path:
            actor_init_path = Path(actor_init_path).resolve()
            payload = torch.load(str(actor_init_path), map_location=self.device)
            state_dict = (
                payload["actor"] if isinstance(payload, dict)
                and "actor" in payload else payload
            )
            self.policy.actor.load_state_dict(state_dict, strict=True)
            digest = hashlib.sha256(actor_init_path.read_bytes()).hexdigest()
            self.actor_init_metadata = {
                "path": str(actor_init_path),
                "sha256": digest,
                "critic_initialized_from_actor_file": False,
                "optimizer_state_loaded": False,
                "scheduler_state_loaded": False,
            }
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
        self.actor_eligible_count = 0
        self.next_pn = np.zeros(
            (self.n_threads, self.num_agents, 3), dtype=np.float32
        )
        self.next_probe = np.zeros_like(self.next_pn)
        self.source_hold_remaining = np.zeros(
            (self.n_threads, self.num_agents), dtype=np.int32
        )
        self.source_is_guided = np.zeros(
            (self.n_threads, self.num_agents), dtype=bool
        )
        self.source_guide_choice = np.zeros(
            (self.n_threads, self.num_agents), dtype=np.int8
        )
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
                "pn_navigation_constant": float(
                    getattr(
                        self.args,
                        "trust_pn_navigation_constant",
                        3.0,
                    )
                ),
                "action_source_hold_steps": int(
                    getattr(self.args, "trust_action_hold_steps", 1)
                ),
                "guide_command_refreshed_each_step": True,
                "guided_transitions_masked_from_gaussian_actor_loss": bool(
                    getattr(
                        self.args,
                        "art_mask_guided_actor_samples",
                        True,
                    )
                ),
                "guided_transitions_retained_for_critic_learning": True,
            },
            "optimization": {
                "num_env_steps": self.total_steps,
                "episode_length": self.episode_length,
                "rollout_threads": self.n_threads,
                "rollout_buffer_transitions": int(
                    self.episode_length * self.n_threads
                ),
                "rollout_buffer_is_not_episode_horizon": True,
                "engagements_continue_across_rollout_updates": True,
                "environment_resets_only_after_all_defenders_terminal": True,
                "trust_updates_use_completed_episode_returns_only": True,
                "physical_episode_horizon_steps": int(
                    self.args.physical_episode_horizon_steps
                ),
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
                "recurrent_sequence_minibatches": bool(
                    self.args.use_recurrent_policy
                ),
                "naive_recurrent_minibatches": bool(
                    self.args.use_naive_recurrent_policy
                ),
                "bptt_chunk_length": int(self.args.data_chunk_length),
                "stored_action_is_executed_clipped_action": True,
            },
            "action_envelope": {
                "low": [-0.1, -1.0, -1.0],
                "high": [1.0, 1.0, 1.0],
                "clipped_before_environment_and_replay_buffer": True,
            },
            "training_diagnostic_definition": {
                "all_sync_tolerance_s": float(
                    self.args.art_metric_sync_tol
                ),
                "affects_reward_or_optimization": False,
            },
            "actor_initialization": self.actor_init_metadata,
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

    def _apply_trust_mixture(self, policy_actions, alive_masks):
        alive = np.asarray(alive_masks[..., 0] > 0.0, dtype=bool)
        # The manuscript samples a Gaussian command and then clips it to the
        # physical envelope before execution.  Clip here (rather than only
        # inside the dynamics) so the action stored in PPO is exactly the
        # action applied to the vehicle.
        clipped_policy_actions = np.asarray(
            policy_actions, dtype=np.float32
        ).copy()
        clipped_policy_actions[..., 0] = np.clip(
            clipped_policy_actions[..., 0], -0.1, 1.0
        )
        clipped_policy_actions[..., 1:] = np.clip(
            clipped_policy_actions[..., 1:], -1.0, 1.0
        )
        if not self.use_trust:
            policy_mask = np.asarray(alive_masks, dtype=np.float32).copy()
            alive_count = int(alive.sum())
            self.action_count += alive_count
            self.actor_eligible_count += alive_count
            return clipped_policy_actions, policy_mask

        beta = np.clip(1.0 - self.trust, 0.0, 1.0)
        hold_steps = int(
            getattr(self.args, "trust_action_hold_steps", 1)
        )
        refresh = (self.source_hold_remaining <= 0) & alive
        if np.any(refresh):
            sampled_guided = (
                self.rng.random((self.n_threads, self.num_agents))
                < beta[None, :]
            )
            sampled_choice = self.rng.choice(
                3,
                size=(self.n_threads, self.num_agents),
                p=[
                    float(self.args.trust_omega_pn),
                    float(self.args.trust_omega_probe),
                    float(self.args.trust_omega_random),
                ],
            )
            self.source_is_guided[refresh] = sampled_guided[refresh]
            self.source_guide_choice[refresh] = sampled_choice[refresh]
            self.source_hold_remaining[refresh] = hold_steps
        guided = self.source_is_guided & alive
        guide_choice = self.source_guide_choice
        actions_env = clipped_policy_actions.copy()
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
        policy_mask = np.asarray(alive_masks, dtype=np.float32).copy()
        if bool(
            getattr(
                self.args,
                "art_mask_guided_actor_samples",
                True,
            )
        ):
            # The external PN/probe/uniform components are not samples from
            # the Gaussian actor.  They remain in the joint trajectory and
            # train the centralized critic, but cannot carry a valid
            # pi_theta/pi_theta_old actor ratio.  Masking their actor loss is
            # the discrete-mixture limit of the manuscript's pi_mix:
            # Dirac guide samples have zero derivative w.r.t. theta.
            policy_mask[guided, 0] = 0.0
        self.guided_count += int(guided.sum())
        self.action_count += int(alive.sum())
        self.actor_eligible_count += int(
            np.count_nonzero(policy_mask[..., 0] > 0.0)
        )
        self.source_hold_remaining[alive] -= 1
        self.source_hold_remaining[~alive] = 0
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
        for thread in range(hit_times.shape[0]):
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
        hit_fractions = []
        target_coverage_rates = []
        complete_group_rates = []
        coordinated_group_rates = []
        for thread in range(hit_times.shape[0]):
            hits = np.isfinite(hit_times[thread])
            all_hit.append(float(np.all(hits)))
            spreads = []
            synced = True
            covered_groups = 0
            complete_groups = 0
            coordinated_groups = 0
            for target in np.unique(assignment):
                idx = np.where(assignment == target)[0]
                if np.any(hits[idx]):
                    covered_groups += 1
                if not np.all(hits[idx]):
                    synced = False
                    continue
                complete_groups += 1
                spread = float(
                    np.max(hit_times[thread, idx])
                    - np.min(hit_times[thread, idx])
                )
                spreads.append(spread)
                if spread > float(self.args.art_metric_sync_tol):
                    synced = False
                else:
                    coordinated_groups += 1
            all_sync.append(float(np.all(hits) and synced))
            hit_fractions.append(float(np.mean(hits)))
            group_count = max(len(np.unique(assignment)), 1)
            target_coverage_rates.append(covered_groups / group_count)
            complete_group_rates.append(complete_groups / group_count)
            coordinated_group_rates.append(
                coordinated_groups / group_count
            )
            mean_spreads.append(
                float(np.mean(spreads)) if spreads else np.nan
            )
        return (
            float(np.mean(all_hit)),
            float(np.mean(all_sync)),
            float(np.nanmean(mean_spreads))
            if np.isfinite(mean_spreads).any()
            else np.nan,
            float(np.mean(hit_fractions)),
            float(np.mean(target_coverage_rates)),
            float(np.mean(complete_group_rates)),
            float(np.mean(coordinated_group_rates)),
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
            "environment_steps": int(
                next_update * self.episode_length * self.n_threads
            ),
            "rollout_steps_per_update": int(
                self.episode_length * self.n_threads
            ),
            "episode_length": int(self.episode_length),
            "n_rollout_threads": int(self.n_threads),
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
            "source_hold_remaining": self.source_hold_remaining,
            "source_is_guided": self.source_is_guided,
            "source_guide_choice": self.source_guide_choice,
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
        if not bool(getattr(self.args, "resume_restart_lr", False)):
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
        self.source_hold_remaining = np.asarray(
            payload.get(
                "source_hold_remaining",
                np.zeros_like(self.source_hold_remaining),
            ),
            dtype=np.int32,
        )
        self.source_is_guided = np.asarray(
            payload.get(
                "source_is_guided",
                np.zeros_like(self.source_is_guided),
            ),
            dtype=bool,
        )
        self.source_guide_choice = np.asarray(
            payload.get(
                "source_guide_choice",
                np.zeros_like(self.source_guide_choice),
            ),
            dtype=np.int8,
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
        if bool(getattr(self.args, "resume_restart_lr", False)):
            for group in self.policy.actor_optimizer.param_groups:
                group["lr"] = float(self.args.lr)
            for group in self.policy.critic_optimizer.param_groups:
                group["lr"] = float(self.args.critic_lr)
            remaining_updates = max(
                int(self.num_updates - self.start_update), 1
            )
            self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.policy.actor_optimizer,
                T_max=remaining_updates,
                eta_min=float(self.args.art_min_lr),
            )
            self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.policy.critic_optimizer,
                T_max=remaining_updates,
                eta_min=float(self.args.art_min_lr),
            )

    def run(self):
        start_time = time.time()
        # A rollout buffer is an optimization chunk, not an engagement
        # horizon.  Reset once, then let the vector wrapper auto-reset a
        # thread only after every defender in that engagement is terminal.
        # This preserves complete physical trajectories even when the paper's
        # 4096-transition buffer is shorter than a difficult engagement.
        self._warmup()
        alive_masks = np.ones(
            (self.n_threads, self.num_agents, 1), dtype=np.float32
        )
        hit_times = np.full(
            (self.n_threads, self.num_agents), np.nan, dtype=np.float64
        )
        episode_returns = np.zeros(
            (self.n_threads, self.num_agents), dtype=np.float64
        )
        episode_discounts = np.ones(
            (self.n_threads, self.num_agents), dtype=np.float64
        )
        for update in range(self.start_update, self.num_updates):
            completed_hit_times = []
            completed_returns = []
            episode_reward = 0.0
            guided_before = self.guided_count
            actions_before = self.action_count
            actor_eligible_before = self.actor_eligible_count

            for step in range(self.episode_length):
                values, policy_actions, log_probs, rs, rsc = (
                    self._collect_policy(step)
                )
                actions_env, policy_mask = self._apply_trust_mixture(
                    policy_actions, alive_masks
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
                episode_returns += (
                    episode_discounts * rewards[..., 0]
                )
                episode_discounts *= self.gamma
                alive_masks[dones] = 0.0
                episode_reward += float(np.mean(rewards))
                self._extract_guides_and_events(infos, hit_times)
                finished_threads = np.all(dones, axis=1)
                for thread in np.flatnonzero(finished_threads):
                    completed_hit_times.append(hit_times[thread].copy())
                    completed_returns.append(
                        episode_returns[thread].copy()
                    )
                    # ``DummyVecEnv``/``SubprocVecEnv`` has already reset
                    # this thread and returned the new initial observation.
                    alive_masks[thread] = 1.0
                    hit_times[thread] = np.nan
                    episode_returns[thread] = 0.0
                    episode_discounts[thread] = 1.0
                    # Guide commands belong to the previous engagement.  A
                    # zero command for the first reset step is safer than a
                    # stale command; fresh guides arrive in the next info.
                    self.next_pn[thread] = 0.0
                    self.next_probe[thread] = 0.0
                    self.source_hold_remaining[thread] = 0
                    self.source_is_guided[thread] = False
                    self.source_guide_choice[thread] = 0

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
            # Manuscript Eq. (trust statistics) is episode based.  Update it
            # only from engagements that actually terminated in this chunk;
            # partial episodes continue into the next 4096-sample buffer.
            if completed_returns:
                self._update_trust(np.stack(completed_returns, axis=0))
            metric_hit_times = (
                np.stack(completed_hit_times, axis=0)
                if completed_hit_times
                else hit_times.copy()
            )
            (
                all_hit_rate,
                all_sync_rate,
                sync_spread,
                hit_fraction,
                target_coverage_rate,
                complete_group_rate,
                coordinated_group_rate,
            ) = (
                self._episode_metrics(metric_hit_times)
            )

            total_steps = (
                (update + 1) * self.episode_length * self.n_threads
            )
            guided_delta = self.guided_count - guided_before
            action_delta = self.action_count - actions_before
            actor_eligible_delta = (
                self.actor_eligible_count - actor_eligible_before
            )
            row = {
                "update": update + 1,
                "environment_steps": total_steps,
                "mean_episode_return": episode_reward,
                "all_hit_rate": all_hit_rate,
                "all_sync_rate": all_sync_rate,
                "hit_fraction": hit_fraction,
                "target_coverage_rate": target_coverage_rate,
                "complete_group_rate": complete_group_rate,
                "coordinated_group_rate": coordinated_group_rate,
                "mean_sync_spread_s": sync_spread,
                "trust_mean": float(np.mean(self.trust)),
                "trust_std": float(np.std(self.trust)),
                "guided_action_fraction": (
                    float(guided_delta / action_delta)
                    if action_delta
                    else 0.0
                ),
                "actor_update_fraction": (
                    float(actor_eligible_delta / action_delta)
                    if action_delta
                    else 0.0
                ),
                "completed_episodes": int(len(completed_hit_times)),
                "policy_loss": float(train_info["policy_loss"]),
                "value_loss": float(train_info["value_loss"]),
                "entropy": float(train_info["dist_entropy"]),
                "approx_kl": float(train_info["kl_divergence"]),
                "importance_ratio": float(train_info["ratio"]),
                "actor_grad_norm": float(train_info["actor_grad_norm"]),
                "critic_grad_norm": float(
                    train_info["critic_grad_norm"]
                ),
                "adaptive_kl_coef": float(train_info["kl_coef"]),
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
