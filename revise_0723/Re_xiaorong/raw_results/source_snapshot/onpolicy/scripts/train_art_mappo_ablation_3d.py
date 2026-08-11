#!/usr/bin/env python
"""Train one controlled ART-MAPPO component-ablation run.

All variants share the same three-dimensional interception environment,
reward, optimizer, rollout budget, and seed.  Only the reviewer-requested
component selected by ``--variant`` is removed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import DummyVecEnv, SubprocVecEnv
from onpolicy.envs.mpe.MPE_env import MPEEnv


VARIANTS = {
    "full": (True, True, True),
    "no_trust": (False, True, True),
    "no_gru": (True, False, True),
    "no_attention_residual": (True, True, False),
}

PAPER_GUIDANCE_PROFILES = {
    # These are the case-specific settings used to collect the paper's
    # verified fixed trajectories (see reward_sensitivity_audit.json).
    "case1": {
        "defender_guidance_base_gain": 2.0,
        "defender_guidance_tau": 0.55,
        "defender_guidance_lead": 1.45,
        "defender_command_lag_tau": 0.55,
    },
    "case2": {
        "defender_guidance_base_gain": 2.4,
        "defender_guidance_tau": 0.40,
        "defender_guidance_lead": 1.70,
        "defender_command_lag_tau": 0.40,
    },
}


def apply_paper_guidance_profile(args):
    profile = PAPER_GUIDANCE_PROFILES[str(args.case_3d)]
    for name, value in profile.items():
        setattr(args, name, value)


def make_train_env(args):
    def env_fn(rank):
        def init_env():
            env = MPEEnv(args)
            env.seed(int(args.seed) + 1000 * rank)
            return env

        return init_env

    factories = [env_fn(rank) for rank in range(args.n_rollout_threads)]
    if args.n_rollout_threads == 1:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories)


def add_ablation_arguments(parser):
    parser.add_argument(
        "--scenario_name", type=str, default="simple_world_comm_3d"
    )
    parser.add_argument("--num_landmarks", type=int, default=8)
    parser.add_argument("--num_agents", type=int, default=20)
    parser.add_argument(
        "--case_3d", choices=["case1", "case2"], default="case1"
    )
    parser.add_argument("--hit_radius_3d", type=float, default=3.0)
    parser.add_argument(
        "--variant", required=True, choices=sorted(VARIANTS)
    )
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--compare_steps", type=int, default=2_200_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint_interval", type=int, default=5)
    parser.add_argument("--art_attention_embed_dim", type=int, default=16)
    parser.add_argument("--art_attention_heads", type=int, default=4)
    parser.add_argument("--art_residual_blocks", type=int, default=2)
    parser.add_argument("--art_attention_dropout", type=float, default=0.0)
    parser.add_argument("--art_min_lr", type=float, default=1e-6)
    parser.add_argument("--trust_initial", type=float, default=0.80)
    parser.add_argument("--trust_rho", type=float, default=0.05)
    parser.add_argument("--trust_alpha", type=float, default=0.10)
    parser.add_argument("--trust_tau", type=float, default=1.0)
    parser.add_argument("--trust_epsilon", type=float, default=1e-8)
    parser.add_argument("--trust_omega_pn", type=float, default=0.60)
    parser.add_argument("--trust_omega_probe", type=float, default=0.20)
    parser.add_argument("--trust_omega_random", type=float, default=0.20)
    parser.add_argument("--kl_coef_min", type=float, default=1e-4)
    parser.add_argument("--kl_coef_max", type=float, default=1.0)


def validate_args(args):
    weights = np.array(
        [
            args.trust_omega_pn,
            args.trust_omega_probe,
            args.trust_omega_random,
        ],
        dtype=float,
    )
    if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("trust guide weights must be nonnegative and sum to one")
    if not Path(args.paper_preset_path).is_file():
        raise FileNotFoundError(args.paper_preset_path)
    if args.episode_length <= 0 or args.compare_steps < args.episode_length:
        raise ValueError("invalid rollout budget")


def main():
    parser = get_config()
    add_ablation_arguments(parser)
    # Paper Table II plus the audited nominal reward and verified
    # case-specific three-dimensional guidance profiles.
    parser.set_defaults(
        env_name="MPE",
        scenario_name="simple_world_comm_3d",
        num_agents=20,
        num_landmarks=8,
        algorithm_name="ART-MAPPO",
        experiment_name="reviewer_component_ablation",
        episode_length=1500,
        n_rollout_threads=4,
        n_training_threads=1,
        hidden_size=256,
        layer_N=1,
        recurrent_N=1,
        use_naive_recurrent_policy=False,
        use_valuenorm=True,
        use_popart=False,
        use_feature_normalization=True,
        use_orthogonal=True,
        use_huber_loss=True,
        huber_delta=15.0,
        use_clipped_value_loss=True,
        use_max_grad_norm=True,
        max_grad_norm=0.5,
        use_gae=True,
        use_proper_time_limits=False,
        use_value_active_masks=False,
        use_policy_active_masks=True,
        data_chunk_length=10,
        gamma=0.99,
        gae_lambda=0.97,
        gain=0.001,
        lr=3e-4,
        critic_lr=3e-4,
        opti_eps=1e-5,
        weight_decay=0.0,
        ppo_epoch=5,
        num_mini_batch=4,
        clip_param=0.2,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        use_linear_lr_decay=False,
        use_centralized_V=True,
        use_dual_clip=True,
        use_adaptive_kl=True,
        use_value_warmup=False,
        use_gae_norm=True,
        log_interval=1,
        save_interval=25,
        case_3d="case1",
        hit_radius_3d=3.0,
        paper_preset_path=str(
            PROJECT_ROOT
            / "paper_case_presets_original_assignment_verified.npz"
        ),
        paper_attacker_replay=0,
        paper_altitude=120.0,
        paper_altitude_step=0.0,
        paper_defender_climb_to_target=0,
        target_assignment_mode="fixed",
        defender_guidance_base_gain=1.0,
        defender_guidance_tau=0.55,
        defender_guidance_lead=1.2,
        defender_residual_scale=0.05,
        defender_load_limit=1.0,
        defender_axial_min=-0.1,
        defender_axial_max=1.0,
        defender_sync_speed_gain=0.14,
        defender_sync_tgo_ref="mean",
        defender_speed_min=12.0,
        defender_speed_max=40.0,
        defender_sensor_delay_steps=1,
        defender_sensor_delay_compensate=False,
        defender_obs_pos_noise_std=3.0,
        defender_obs_vel_noise_std=0.3,
        defender_obs_filter_alpha=1.0,
        defender_command_lag_tau=0.30,
        reward_w_smooth=0.0,
        reward_w_dist=4.0,
        reward_w_angle=0.05,
        reward_w_hit=200.0,
        reward_w_coord=18.0,
        reward_w_sync=220.0,
        reward_w_energy=0.005,
        reward_alpha_dist=2.0e-4,
        reward_alpha_angle=0.002,
        reward_alpha_coord=0.16,
        reward_alpha_sync=0.45,
        reward_alpha_energy=0.010,
        reward_hit_bonus=2200.0,
        reward_hit_shaping=3.5,
        reward_hit_band_ratio=24.0,
        reward_coord_bonus=30.0,
        reward_coord_tol=3.0,
        reward_sync_bonus=550.0,
        reward_sync_tol=3.0,
        reward_async_hit_penalty=900.0,
        reward_sync_power=1.0,
        reward_sync_hits=0,
        reward_angle_power=0.3,
        reward_coord_power=0.3,
        reward_use_progress=False,
        no_tailchase_gate=0.0,
        no_tailchase_rebound=5.0,
        no_tailchase_penalty=0.0,
        no_tailchase_terminate=False,
    )
    args = parser.parse_args()
    apply_paper_guidance_profile(args)
    trust, gru, backbone = VARIANTS[args.variant]
    args.ablation_variant = args.variant
    args.art_use_trust = trust
    args.art_use_gru = gru
    args.art_use_attention_residual = backbone
    args.art_local_obs_dim = 17
    args.use_recurrent_policy = gru
    args.num_env_steps = int(args.compare_steps)
    validate_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(args.n_training_threads)
    device = torch.device(
        "cuda:0" if args.cuda and torch.cuda.is_available() else "cpu"
    )

    run_dir = (
        Path(args.save_dir)
        / args.variant
        / args.case_3d
        / f"seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "command_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True, default=str)

    envs = make_train_env(args)
    obs_dim = int(envs.observation_space[0].shape[0])
    action_dim = int(envs.action_space[0].shape[0])
    if (obs_dim, action_dim) != (17, 3):
        envs.close()
        raise RuntimeError(
            f"paper mismatch: expected observation/action 17/3, got "
            f"{obs_dim}/{action_dim}"
        )

    from onpolicy.runner.shared.art_ablation_runner import (
        ARTMAPPOAblationRunner,
    )

    runner = ARTMAPPOAblationRunner(
        {
            "all_args": args,
            "envs": envs,
            "num_agents": args.num_agents,
            "device": device,
            "run_dir": run_dir,
        }
    )
    try:
        metrics_path = runner.run()
    finally:
        envs.close()
    print(f"[DONE] metrics={metrics_path}", flush=True)


if __name__ == "__main__":
    main()
