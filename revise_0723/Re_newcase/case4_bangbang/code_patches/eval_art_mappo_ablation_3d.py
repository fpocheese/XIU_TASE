#!/usr/bin/env python3
import argparse
import os
import sys
import csv
import json
from pathlib import Path
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path = [str(PROJECT_ROOT)] + [p for p in sys.path if str(Path(p).resolve()) != str(PROJECT_ROOT)]
stale_roots = ["/home/uav/00gao_xueshu/togsy_2025/0620septimedone/on-policy-main", "/home/uav/00gao_xueshu/togsy_2025"]
for stale in stale_roots:
    sys.path = [p for p in sys.path if not str(Path(p)).startswith(stale)]
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False
    plt = None
import torch

from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy_art import R_MAPPOPolicy_ART


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

PAPER_GUIDANCE_PROFILES = {
    "case1": {
        "defender_guidance_base_gain": 0.0,
        "defender_guidance_tau": 0.55,
        "defender_guidance_lead": 1.45,
        "defender_command_lag_tau": 0.30,
    },
    "case2": {
        "defender_guidance_base_gain": 2.6,
        "defender_guidance_tau": 0.35,
        "defender_guidance_lead": 1.70,
        "defender_residual_scale": 0.1,
        "defender_sync_speed_gain": 2.0,
        "defender_command_lag_tau": 0.40,
    },
    "case3": {
        "defender_guidance_base_gain": 2.6,
        "defender_guidance_tau": 0.35,
        "defender_guidance_lead": 1.70,
        "defender_residual_scale": 0.2,
        "defender_sync_speed_gain": 1.4,
        "defender_command_lag_tau": 0.40,
    },
    "case4": {
        "defender_guidance_base_gain": 2.6,
        "defender_guidance_tau": 0.35,
        "defender_guidance_lead": 1.70,
        "defender_residual_scale": 0.1,
        "defender_sync_speed_gain": 2.0,
        "defender_command_lag_tau": 0.40,
    },
}


def select_model_dir(model_root: Path) -> Path:
    if (model_root / "actor.pt").exists() and (model_root / "critic.pt").exists():
        return model_root
    candidates = [p for p in model_root.glob("**/actor.pt")]
    if not candidates:
        raise FileNotFoundError(f"未在 {model_root} 找到 actor.pt")
    # 优先选最近修改的子目录里的模型
    best = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return best.parent


def build_args(case_name: str, raw_args):
    parser = get_config()

    parser.add_argument("--scenario_name", type=str, default="simple_world_comm_3d")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=20)
    parser.add_argument("--algo", type=str, default="MAPPO", choices=["MAPPO", "Advanced-MAPPO", "IPPO", "IA2C", "IQL"])
    parser.add_argument("--case_3d", type=str, default="case1", choices=["case1", "case2", "case3", "case4"])
    parser.add_argument("--hit_radius_3d", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--sync_tol", type=float, default=0.5)
    parser.add_argument("--sync_min_hits", type=int, default=0, help="时间协同判定所需最少命中数；0 表示要求该目标分配到的整组拦截器全部命中")
    parser.add_argument("--outdir", type=str, default="onpolicy/scripts/results/3d_eval")
    parser.add_argument("--model_dir_case1", type=str, default=None)
    parser.add_argument("--model_dir_case2", type=str, default=None)
    parser.add_argument("--require_all_hit", action="store_true", help="要求所有目标均被拦截才算成功")
    parser.add_argument("--variant", choices=["full", "no_trust", "no_gru", "no_attention_residual"], required=True)
    parser.add_argument(
        "--attack_pattern",
        choices=[
            "nominal",
            "chirp",
            "multisine",
            "jink",
            "case3_hybrid",
            "bangbang",
        ],
        default="nominal",
    )
    parser.add_argument("--unseen_chirp_rate", type=float, default=0.004)
    parser.add_argument("--unseen_frequency_scale", type=float, default=1.35)
    parser.add_argument("--unseen_vertical_load_amp", type=float, default=0.18)
    parser.add_argument("--case3_terminal_frequency_scale", type=float, default=0.45)
    parser.add_argument("--case3_terminal_chirp_rate", type=float, default=0.0008)
    parser.add_argument("--case3_terminal_frequency_cap", type=float, default=0.16)
    parser.add_argument("--case4_bangbang_period", type=float, default=4.2)
    parser.add_argument("--case4_bangbang_min_dwell", type=float, default=1.6)
    parser.add_argument("--case4_bangbang_hysteresis", type=float, default=0.15)
    parser.add_argument("--case4_bangbang_axial_load", type=float, default=0.10)
    parser.add_argument("--case4_bangbang_lateral_scale", type=float, default=0.75)

    parser.set_defaults(
        model_dir=None,
        use_render=False,
        use_wandb=False,
        n_rollout_threads=1,
        episode_length=1500,
        hidden_size=256,
        layer_N=1,
        use_recurrent_policy=True,
        use_naive_recurrent_policy=False,
        algorithm_name="rmappo",
        use_valuenorm=True,
        use_popart=False,
        ppo_epoch=10,
        clip_param=0.1,
        lr=5e-4,
        critic_lr=5e-4,
        entropy_coef=0.01,
        max_grad_norm=0.3,
        value_loss_coef=0.3,
        huber_delta=15.0,
        gamma=0.99,
        gae_lambda=0.95,
        log_interval=5,
        save_interval=100,
        use_linear_lr_decay=True,
        use_centralized_V=True,
        eval=False,
        eval_interval=5,
        eval_episodes=5,
        save_dir=None,
        reward_definition="paper_five_component",
        reward_w_dist=4.0,
        reward_w_angle=0.05,
        reward_w_hit=200.0,
        reward_w_coord=18.0,
        reward_w_energy=0.005,
        reward_w_sync=0.0,
        reward_alpha_dist=2.0e-4,
        reward_alpha_angle=0.002,
        reward_alpha_coord=0.16,
        reward_alpha_energy=0.010,
        reward_alpha_sync=0.0,
        reward_hit_bonus=2200.0,
        reward_hit_shaping=0.0,
        reward_coord_bonus=0.0,
        reward_sync_bonus=0.0,
        reward_async_hit_penalty=0.0,
        reward_angle_power=1.0,
        reward_coord_power=1.0,
        reward_use_progress=False,
    )

    argv = [
        "--scenario_name", "simple_world_comm_3d",
        "--case_3d", case_name,
    ]
    for k, v in sorted(vars(raw_args).items()):
        if k in {"scenario_name", "case_3d", "model_dir", "model_dir_case1", "model_dir_case2", "outdir", "case1", "case2", "eval_episodes", "max_steps", "variant"}:
            continue
        if k in {"stochastic_eval", "eval_different_seed", "require_success_plot", "require_all_hit"}:
            continue
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                argv.append(f"--{k}")
            continue
        argv.extend([f"--{k}", str(v)])
    argv.extend(["--case_3d", case_name])
    argv.extend(["--variant", raw_args.variant])
    argv.extend(["--seed", str(raw_args.seed)])
    if raw_args.eval_episodes is not None:
        argv.extend(["--eval_episodes", str(raw_args.eval_episodes)])
    if raw_args.max_steps is not None:
        argv.extend(["--max_steps", str(raw_args.max_steps)])
    argv.extend(["--sync_tol", str(raw_args.sync_tol)])

    args = parser.parse_args(argv)
    args.case_3d = case_name
    for name, value in PAPER_GUIDANCE_PROFILES[case_name].items():
        setattr(args, name, value)
    args.num_env_steps = args.episode_length
    args.stochastic_eval = raw_args.stochastic_eval
    args.eval_different_seed = raw_args.eval_different_seed
    args.require_success_plot = raw_args.require_success_plot
    args.require_all_hit = raw_args.require_all_hit
    args.ablation_variant = raw_args.variant
    args.art_use_trust = raw_args.variant != "no_trust"
    args.art_use_gru = raw_args.variant != "no_gru"
    args.art_use_attention_residual = raw_args.variant != "no_attention_residual"
    args.art_local_obs_dim = 17
    args.art_attention_embed_dim = 16
    args.art_attention_heads = 4
    args.art_residual_blocks = 2
    args.art_attention_dropout = 0.0
    args.attack_pattern = raw_args.attack_pattern
    args.unseen_chirp_rate = raw_args.unseen_chirp_rate
    args.unseen_frequency_scale = raw_args.unseen_frequency_scale
    args.unseen_vertical_load_amp = raw_args.unseen_vertical_load_amp
    args.use_recurrent_policy = args.art_use_gru
    return args


def collect_model(args, model_root: Path):
    model_dir = select_model_dir(model_root)
    actor_path = model_dir / "actor.pt"
    critic_path = model_dir / "critic.pt"
    if not actor_path.exists() or not critic_path.exists():
        raise FileNotFoundError(f"模型路径不完整: {model_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = MPEEnv(args)
    env.seed(args.seed)
    env.world.seed = args.seed if hasattr(env.world, "seed") else None

    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.manual_seed_all(args.seed)
    else:
        device = torch.device("cpu")

    policy = R_MAPPOPolicy_ART(
        args,
        env.observation_space[0],
        env.share_observation_space[0],
        env.action_space[0],
        device=device,
    )
    policy.actor.load_state_dict(torch.load(str(actor_path), map_location=device))
    policy.critic.load_state_dict(torch.load(str(critic_path), map_location=device))
    policy.actor.eval()
    policy.critic.eval()

    return env, policy, device, model_dir


def evaluate_case(args, env, policy, device):
    n_agents = len(env.world.policy_agents)
    target_num = None

    all_case_results = []
    all_success = []
    best_hit_count = -1
    best_min_dist = float("inf")
    rep_att = None
    rep_def = None
    rep_ctrl = None
    rep_tgo = None
    rep_hit_count = 0
    rep_min_dist = float("inf")
    hit_events = []
    episode_summary_events = []
    success_episode_rows = []
    

    for ep in range(args.eval_episodes):
        if args.eval_different_seed:
            # 每个回合使用不同随机种子，扩展初始态势采样覆盖范围
            eval_seed = args.seed + ep
            np.random.seed(eval_seed)
            torch.manual_seed(eval_seed)

        obs = env.reset()
        obs = np.asarray(obs, dtype=np.float32)

        rnn_states = np.zeros((n_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
        masks = np.ones((n_agents, 1), dtype=np.float32)

        traj_att = []
        traj_def = []
        ctrl_hist = []
        tgo_hist = []
        hit_time = np.full(n_agents, np.nan, dtype=np.float32)
        dist_hist = []
        ep_hit_events = []
        ep_agent_return = np.zeros(n_agents, dtype=np.float64)

        adv_agents = [ag for ag in env.world.agents if ag.adversary]
        tgt_ids = sorted({int(ag.target) for ag in adv_agents})
        tgt_size = len(tgt_ids)
        target_num = tgt_size
        tgt_index = {tid: k for k, tid in enumerate(tgt_ids)}

        def snapshot_state():
            def_pos = []
            att_pos = []
            tgo_this = []
            ctrl_this = []
            for ag in env.world.agents:
                if ag.adversary:
                    def_pos.append(ag.state.p_pos.copy())
                    ctrl_this.append(ag.state.load.copy())
                    tgo_this.append(ag.state.time_tgo[0])
                else:
                    att_pos.append(ag.state.p_pos.copy())
            traj_def.append(np.array(def_pos))
            traj_att.append(np.array(att_pos))
            ctrl_hist.append(np.array(ctrl_this))
            tgo_hist.append(np.array(tgo_this))

        snapshot_state()

        for t in range(args.max_steps):
            with torch.no_grad():
                actions, rnn_states = policy.act(
                    obs,
                    rnn_states,
                    masks,
                    deterministic=not args.stochastic_eval,
                )
            actions_np = actions.detach().cpu().numpy()
            actions_np[..., 0] = np.clip(actions_np[..., 0], -0.1, 1.0)
            actions_np[..., 1:] = np.clip(actions_np[..., 1:], -1.0, 1.0)
            obs_next, rewards, dones, infos = env.step(actions_np, t)
            reward_array = np.asarray(rewards, dtype=np.float64).reshape(
                n_agents, -1
            )
            ep_agent_return += reward_array.sum(axis=1)
            obs = np.asarray(obs_next, dtype=np.float32)

            def_pos = []
            att_pos = []
            tgo_this = []
            ctrl_this = []
            for ag in env.world.agents:
                if ag.adversary:
                    def_pos.append(ag.state.p_pos.copy())
                    ctrl_this.append(ag.state.load.copy())
                    tgo_this.append(ag.state.time_tgo[0])
                    target = env.world.agents[ag.target]
                    dist_hist.append(np.linalg.norm(target.state.p_pos - ag.state.p_pos))
                else:
                    att_pos.append(ag.state.p_pos.copy())

            traj_def.append(np.array(def_pos))
            traj_att.append(np.array(att_pos))
            ctrl_hist.append(np.array(ctrl_this))
            tgo_hist.append(np.array(tgo_this))

            dones_arr = np.asarray(dones, dtype=bool)
            new_hit_events = []
            for i, ag in enumerate(adv_agents):
                if getattr(ag.state, "actual_hit", False) and np.isnan(hit_time[i]):
                    hit_time[i] = (t + 1) * env.dt
                    target_id = int(ag.target)
                    target = env.world.agents[target_id]
                    speed, pitch, yaw = ag.state.v_vel.tolist() if hasattr(ag.state, "v_vel") else (float("nan"), float("nan"), float("nan"))
                    p_pos = ag.state.p_pos
                    p_vel = ag.state.p_vel if hasattr(ag.state, "p_vel") else np.zeros(3)
                    load = ag.state.load if hasattr(ag.state, "load") else np.zeros(2)
                    new_hit_events.append({
                        "episode": ep + 1,
                        "step": t + 1,
                        "time": hit_time[i],
                        "case": args.case_3d,
                        "defender_id": int(ag.namenumber),
                        "target_id": target_id,
                        "defender_pos_x": float(p_pos[0]),
                        "defender_pos_y": float(p_pos[1]),
                        "defender_pos_z": float(p_pos[2]),
                        "defender_vx": float(p_vel[0]),
                        "defender_vy": float(p_vel[1]),
                        "defender_vz": float(p_vel[2]),
                        "defender_speed": float(speed),
                        "defender_pitch": float(pitch),
                        "defender_yaw": float(yaw),
                        "load_y": float(load[0]),
                        "load_z": float(load[1]),
                        "load_norm": float(np.linalg.norm(load)),
                        "dist_to_target": float(np.linalg.norm(target.state.p_pos - p_pos)),
                        "tgo": float(ag.state.time_tgo[0]) if hasattr(ag.state, "time_tgo") and len(ag.state.time_tgo) > 0 else float("nan"),
                    })
            if new_hit_events:
                hit_events.extend(new_hit_events)
                ep_hit_events.extend(new_hit_events)
            if dones_arr.all():
                break

            masks = np.ones_like(masks, dtype=np.float32)
            masks[dones_arr, 0] = 0.0

        # 按照目标统计是否成功
        hit_targets = np.zeros(tgt_size, dtype=bool)
        sync_targets = np.zeros(tgt_size, dtype=bool)
        per_target_time = np.full(tgt_size, np.nan, dtype=np.float32)
        per_target_spread = np.full(tgt_size, np.nan, dtype=np.float32)
        for tid in tgt_ids:
            idx = tgt_index[int(tid)]
            group_times = [hit_time[i] for i, ag in enumerate(adv_agents) if int(ag.target) == int(tid) and not np.isnan(hit_time[i])]
            group_size = sum(1 for ag in adv_agents if int(ag.target) == int(tid))
            if group_times:
                hit_targets[idx] = True
                per_target_time[idx] = float(np.min(group_times))
                if len(group_times) >= 2:
                    per_target_spread[idx] = float(max(group_times) - min(group_times))
                required_hits = group_size if args.sync_min_hits <= 0 else max(1, min(group_size, args.sync_min_hits))
                if len(group_times) >= required_hits and (max(group_times) - min(group_times)) <= args.sync_tol:
                    sync_targets[idx] = True

        hit_count = int(np.nansum(~np.isnan(hit_time)))
        hit_target_count = int(np.count_nonzero(hit_targets))
        sync_target_count = int(np.count_nonzero(sync_targets))
        all_hit = hit_target_count >= tgt_size
        all_sync = sync_target_count >= tgt_size
        success_rate = hit_count / max(1, len(adv_agents))
        all_hit_rate = 1.0 if all_hit else 0.0
        all_sync_rate = 1.0 if all_sync else 0.0
        if np.isfinite(per_target_time).any():
            mean_target_time = float(np.nanmean(per_target_time))
        else:
            mean_target_time = float("nan")
        if np.isfinite(per_target_spread).any():
            max_sync_spread = float(np.nanmax(per_target_spread))
            mean_sync_spread = float(np.nanmean(per_target_spread))
        else:
            max_sync_spread = float("nan")
            mean_sync_spread = float("nan")

        ep_traj_def = np.array(traj_def)
        ep_traj_att = np.array(traj_att)
        ep_ctrl = np.array(ctrl_hist)
        ep_tgo = np.array(tgo_hist)
        ep_min_dist = float(np.nanmin(np.array(dist_hist))) if len(dist_hist) > 0 else float("inf")

        if (not args.require_all_hit and (hit_count > best_hit_count or (hit_count == best_hit_count and ep_min_dist < best_min_dist))) or (
            args.require_all_hit and all_hit and ep_min_dist < best_min_dist
        ):
            best_hit_count = hit_count
            best_min_dist = ep_min_dist
            rep_att = ep_traj_att
            rep_def = ep_traj_def
            rep_ctrl = ep_ctrl
            rep_tgo = ep_tgo
            rep_hit_count = hit_count
            rep_min_dist = ep_min_dist

        all_case_results.append({
            "success_rate": success_rate,
            "hit_targets": hit_target_count,
            "sync_targets": sync_target_count,
            "all_hit": bool(all_hit),
            "all_sync": bool(all_sync),
            "mean_target_time": mean_target_time,
            "target_mask": hit_targets,
            "sync_mask": sync_targets,
            "hit_time": hit_time,
            "target_spread": per_target_spread,
            "max_sync_spread": max_sync_spread,
            "mean_sync_spread": mean_sync_spread,
            "mean_agent_return": float(np.mean(ep_agent_return)),
            "team_return": float(np.sum(ep_agent_return)),
        })
        episode_summary_events.append({
            "episode": ep + 1,
            "case": args.case_3d,
            "target_num": int(target_num if target_num is not None else 0),
            "hit_count": int(np.nansum(~np.isnan(hit_time))),
            "target_hit_count": int(np.count_nonzero(hit_targets)),
            "target_sync_count": int(np.count_nonzero(sync_targets)),
            "all_hit": bool(all_hit),
            "all_sync": bool(all_sync),
            "max_sync_spread": max_sync_spread,
            "mean_sync_spread": mean_sync_spread,
            "mean_agent_return": float(np.mean(ep_agent_return)),
            "team_return": float(np.sum(ep_agent_return)),
        })
        if all_hit and all_sync:
            success_episode_rows.append({
                "episode": ep + 1,
                "case": args.case_3d,
                "target_num": int(target_num if target_num is not None else 0),
                "hit_count": int(np.nansum(~np.isnan(hit_time))),
                "target_hit_count": int(np.count_nonzero(hit_targets)),
                "target_sync_count": int(np.count_nonzero(sync_targets)),
                "all_hit": bool(all_hit),
                "all_sync": bool(all_sync),
                "mean_target_time": mean_target_time,
                "max_sync_spread": max_sync_spread,
                "mean_sync_spread": mean_sync_spread,
                "mean_agent_return": float(np.mean(ep_agent_return)),
                "team_return": float(np.sum(ep_agent_return)),
            })
            success_root = Path(args.outdir) / args.case_3d / "success_episodes"
            success_dir = success_root / f"ep_{ep + 1:03d}"
            success_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                success_dir / "episode.npz",
                rep_att=ep_traj_att,
                rep_def=ep_traj_def,
                rep_ctrl=ep_ctrl,
                rep_tgo=ep_tgo,
                hit_time=hit_time,
                selected_hit_count=hit_count,
                selected_min_dist=ep_min_dist,
                episode=ep + 1,
                case=args.case_3d,
            )
            with open(success_dir / "episode_summary.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "episode",
                        "case",
                        "target_num",
                        "hit_count",
                        "target_hit_count",
                        "target_sync_count",
                        "all_hit",
                        "all_sync",
                        "mean_target_time",
                        "max_sync_spread",
                        "mean_sync_spread",
                        "mean_agent_return",
                        "team_return",
                    ],
                )
                writer.writeheader()
                writer.writerow(success_episode_rows[-1])
            hit_fieldnames = [
                "episode",
                "step",
                "time",
                "case",
                "defender_id",
                "target_id",
                "defender_pos_x",
                "defender_pos_y",
                "defender_pos_z",
                "defender_vx",
                "defender_vy",
                "defender_vz",
                "defender_speed",
                "defender_pitch",
                "defender_yaw",
                "load_y",
                "load_z",
                "load_norm",
                "dist_to_target",
                "tgo",
            ]
            with open(success_dir / "hit_events.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=hit_fieldnames)
                writer.writeheader()
                for evt in ep_hit_events:
                    writer.writerow(evt)
        all_success.append(all_hit_rate if args.require_all_hit else success_rate)
        if (ep + 1) % 50 == 0:
            print(
                f"[INFO] case={args.case_3d} ep={ep + 1}/{args.eval_episodes} "
                f"best_hit={best_hit_count} best_min_dist={best_min_dist:.2f} "
                f"all_hit_mode={args.require_all_hit}"
            )

    return {
        "case_summary": {
            "episodes": args.eval_episodes,
        "success_rate": float(np.mean(all_success)),
            "all_hit_rate": float(np.mean([1.0 if r["all_hit"] else 0.0 for r in all_case_results])) if all_case_results else 0.0,
            "target_success_rate": float(np.mean([r["hit_targets"] for r in all_case_results]) / max(1, target_num if target_num is not None else 1)),
            "sync_success_rate": float(np.mean([r["sync_targets"] for r in all_case_results]) / max(1, target_num if target_num is not None else 1)),
            "all_sync_rate": float(np.mean([1.0 if r["all_sync"] else 0.0 for r in all_case_results])) if all_case_results else 0.0,
        "mean_hit_time": float(np.nanmean([r["mean_target_time"] for r in all_case_results])) if all_case_results else float("nan"),
            "max_sync_spread": float(np.nanmax([r["max_sync_spread"] for r in all_case_results])) if all_case_results and np.isfinite([r["max_sync_spread"] for r in all_case_results]).any() else float("nan"),
            "mean_sync_spread": float(np.nanmean([r["mean_sync_spread"] for r in all_case_results])) if all_case_results and np.isfinite([r["mean_sync_spread"] for r in all_case_results]).any() else float("nan"),
            "mean_agent_return": float(np.mean([r["mean_agent_return"] for r in all_case_results])) if all_case_results else float("nan"),
            "team_return": float(np.mean([r["team_return"] for r in all_case_results])) if all_case_results else float("nan"),
        },
        "rep_att": np.array(rep_att) if rep_att is not None else np.empty((0,)),
        "rep_def": np.array(rep_def) if rep_def is not None else np.empty((0,)),
        "rep_ctrl": np.array(rep_ctrl) if rep_ctrl is not None else np.empty((0,)),
        "rep_tgo": np.array(rep_tgo) if rep_tgo is not None else np.empty((0,)),
        "selected_hit_count": int(rep_hit_count),
        "selected_min_dist": float(rep_min_dist),
        "all_results": all_case_results,
        "success_events": hit_events,
        "episode_summary_events": episode_summary_events,
        "success_episode_rows": success_episode_rows,
    }


def plot_case(case_name: str, data: dict, outdir: Path):
    if not _HAS_MATPLOTLIB:
        print("[WARN] 未检测到 matplotlib，已跳过作图，仅输出评估指标")
        return
    outdir.mkdir(parents=True, exist_ok=True)

    att = data["rep_att"]
    deff = data["rep_def"]
    ctrl = data["rep_ctrl"]
    tgo = data["rep_tgo"]

    if data["rep_att"].size == 0 or data["rep_def"].size == 0:
        print(f"[WARN] case={case_name} 未找到符合条件的拦截样本，跳过作图")
        return

    t_steps = np.arange(att.shape[0]) * 0.05

    def save_figure(fig, *stems):
        for stem in stems:
            for ext in ["png", "pdf"]:
                fig.savefig(outdir / f"{stem}.{ext}", dpi=200)

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    for j in range(att.shape[1]):
        ax.plot(
            att[:, j, 0],
            att[:, j, 1],
            att[:, j, 2],
            linewidth=1.0,
            color="crimson",
            alpha=0.55,
            linestyle="--",
            label="attacker" if j == 0 else None,
        )
    for i in range(deff.shape[1]):
        ax.plot(
            deff[:, i, 0],
            deff[:, i, 1],
            deff[:, i, 2],
            linewidth=0.8,
            alpha=0.75,
            label="defender" if i == 0 else None,
        )
    ax.scatter([0.0], [0.0], [0.0], marker="*", s=80, color="black", label="target")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    status_text = "success" if data.get("selected_hit_count", 0) > 0 else "no success"
    ax.set_title(f"Case {case_name} 3D Trajectory ({status_text})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save_figure(fig, f"{case_name}_trajectory", f"{case_name}_trajectory_3d")
    plt.close(fig)

    fig_top, ax_top = plt.subplots(figsize=(8, 6))
    for j in range(att.shape[1]):
        ax_top.plot(
            att[:, j, 0],
            att[:, j, 1],
            linestyle="--",
            color="crimson",
            alpha=0.55,
            label="attacker" if j == 0 else None,
        )
    for i in range(deff.shape[1]):
        ax_top.plot(
            deff[:, i, 0],
            deff[:, i, 1],
            linewidth=0.8,
            alpha=0.75,
            label="defender" if i == 0 else None,
        )
    ax_top.scatter([0.0], [0.0], marker="*", s=80, color="black", label="target")
    ax_top.set_xlabel("x")
    ax_top.set_ylabel("y")
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.set_title(f"Case {case_name} Top-down Trajectory")
    ax_top.legend(loc="upper right", fontsize=8)
    fig_top.tight_layout()
    save_figure(fig_top, f"{case_name}_trajectory_xy")
    plt.close(fig_top)

    fig_side, ax_side = plt.subplots(figsize=(8, 5))
    for j in range(att.shape[1]):
        ax_side.plot(
            att[:, j, 0],
            att[:, j, 2],
            linestyle="--",
            color="crimson",
            alpha=0.55,
            label="attacker" if j == 0 else None,
        )
    for i in range(deff.shape[1]):
        ax_side.plot(
            deff[:, i, 0],
            deff[:, i, 2],
            linewidth=0.8,
            alpha=0.75,
            label="defender" if i == 0 else None,
        )
    ax_side.scatter([0.0], [0.0], marker="*", s=80, color="black", label="target")
    ax_side.set_xlabel("x")
    ax_side.set_ylabel("z")
    ax_side.set_title(f"Case {case_name} Elevation Trajectory")
    ax_side.legend(loc="upper right", fontsize=8)
    fig_side.tight_layout()
    save_figure(fig_side, f"{case_name}_trajectory_xz")
    plt.close(fig_side)

    fig2, ax2 = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax2[0].plot(t_steps[:ctrl.shape[0]], ctrl[:, :, 0], alpha=0.2, color="tab:blue")
    if ctrl.ndim == 3 and ctrl.shape[2] > 0:
        ax2[0].plot(t_steps[:ctrl.shape[0]], np.nanmean(ctrl[:, :, 0], axis=1), color="tab:blue", linewidth=1.6)
    ax2[0].set_ylabel("Yaw overload (g)")
    ax2[0].set_title(f"Case {case_name} Defender Commands")

    ax2[1].plot(t_steps[:ctrl.shape[0]], ctrl[:, :, 1], alpha=0.2, color="tab:orange")
    if ctrl.ndim == 3 and ctrl.shape[2] > 1:
        ax2[1].plot(t_steps[:ctrl.shape[0]], np.nanmean(ctrl[:, :, 1], axis=1), color="tab:orange", linewidth=1.6)
    ax2[1].set_xlabel("Time (s)")
    ax2[1].set_ylabel("Pitch overload (g)")

    fig2.tight_layout()
    save_figure(fig2, f"{case_name}_control")
    plt.close(fig2)

    if ctrl.ndim == 3 and ctrl.shape[2] > 0:
        fig_yaw, ax_yaw = plt.subplots(figsize=(8, 4))
        ax_yaw.plot(t_steps[:ctrl.shape[0]], ctrl[:, :, 0], alpha=0.18, color="tab:blue")
        ax_yaw.plot(t_steps[:ctrl.shape[0]], np.nanmean(ctrl[:, :, 0], axis=1), color="tab:blue", linewidth=1.8)
        ax_yaw.set_title(f"Case {case_name} Yaw Overload Command")
        ax_yaw.set_xlabel("Time (s)")
        ax_yaw.set_ylabel("Yaw overload (g)")
        fig_yaw.tight_layout()
        save_figure(fig_yaw, f"{case_name}_yaw_overload")
        plt.close(fig_yaw)

    if ctrl.ndim == 3 and ctrl.shape[2] > 1:
        fig_pitch, ax_pitch = plt.subplots(figsize=(8, 4))
        ax_pitch.plot(t_steps[:ctrl.shape[0]], ctrl[:, :, 1], alpha=0.18, color="tab:orange")
        ax_pitch.plot(t_steps[:ctrl.shape[0]], np.nanmean(ctrl[:, :, 1], axis=1), color="tab:orange", linewidth=1.8)
        ax_pitch.set_title(f"Case {case_name} Pitch Overload Command")
        ax_pitch.set_xlabel("Time (s)")
        ax_pitch.set_ylabel("Pitch overload (g)")
        fig_pitch.tight_layout()
        save_figure(fig_pitch, f"{case_name}_pitch_overload")
        plt.close(fig_pitch)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    min_tgo = np.nanmin(tgo, axis=1) if tgo.size else np.array([])
    if min_tgo.ndim == 0:
        min_tgo = np.array([float(min_tgo)])
    ax3.plot(t_steps[:len(min_tgo)], min_tgo, color="tab:green")
    ax3.set_title(f"Case {case_name} Min TGO")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("time-to-go (s)")
    fig3.tight_layout()
    save_figure(fig3, f"{case_name}_tgo")
    plt.close(fig3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["full", "no_trust", "no_gru", "no_attention_residual"])
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--hit_radius_3d", type=float, default=3.0)
    parser.add_argument("--sync_tol", type=float, default=0.5, help="同目标组最大到达时间差阈值，单位 s")
    parser.add_argument("--sync_min_hits", type=int, default=0, help="时间协同判定所需最少命中数；0 表示要求该目标分配到的整组拦截器全部命中")
    parser.add_argument("--defender_guidance_base_gain", type=float, default=1.0)
    parser.add_argument("--defender_guidance_tau", type=float, default=0.55)
    parser.add_argument("--defender_guidance_lead", type=float, default=1.2)
    parser.add_argument("--defender_residual_scale", type=float, default=1.0)
    parser.add_argument("--defender_load_limit", type=float, default=1.0)
    parser.add_argument("--defender_axial_min", type=float, default=-0.1)
    parser.add_argument("--defender_axial_max", type=float, default=1.0)
    parser.add_argument("--defender_sync_speed_gain", type=float, default=0.14)
    parser.add_argument("--defender_sync_tgo_ref", type=str, default="mean", choices=["mean", "max", "min"])
    parser.add_argument("--defender_speed_target", type=float, default=0.0)
    parser.add_argument("--defender_speed_gain", type=float, default=0.0)
    parser.add_argument("--defender_min_accel_load", type=float, default=0.0)
    parser.add_argument("--defender_speed_min", type=float, default=12.0)
    parser.add_argument("--defender_speed_max", type=float, default=40.0)
    parser.add_argument("--defender_sensor_delay_steps", type=int, default=1)
    parser.add_argument("--defender_sensor_delay_compensate", action="store_true")
    parser.add_argument("--defender_obs_pos_noise_std", type=float, default=3.0)
    parser.add_argument("--defender_obs_vel_noise_std", type=float, default=0.3)
    parser.add_argument("--defender_obs_filter_alpha", type=float, default=1.0)
    parser.add_argument("--defender_command_lag_tau", type=float, default=0.30)
    parser.add_argument("--reward_w_smooth", type=float, default=0.0)
    parser.add_argument("--reference_control_root", type=str, default="")
    parser.add_argument("--reward_w_ref_control", type=float, default=0.0)
    parser.add_argument("--reward_w_ref_rate", type=float, default=0.0)
    parser.add_argument("--defender_reference_blend", type=float, default=0.0)
    parser.add_argument("--target_assignment_mode", type=str, default="fixed", choices=["fixed", "dynamic"])
    parser.add_argument("--target_assignment_spread_weight", type=float, default=6.0)
    parser.add_argument("--attack_maneuver_gain", type=float, default=1.20)
    parser.add_argument("--attack_maneuver_offset_gain", type=float, default=1.25)
    parser.add_argument("--attack_maneuver_freq", type=float, default=0.17)
    parser.add_argument("--attack_maneuver_fade_range", type=float, default=450.0)
    parser.add_argument(
        "--attack_pattern",
        choices=["nominal", "chirp", "multisine", "jink", "case3_hybrid", "bangbang"],
        default="nominal",
    )
    parser.add_argument("--unseen_chirp_rate", type=float, default=0.004)
    parser.add_argument("--unseen_frequency_scale", type=float, default=1.35)
    parser.add_argument("--unseen_vertical_load_amp", type=float, default=0.18)
    parser.add_argument("--case3_terminal_frequency_scale", type=float, default=0.45)
    parser.add_argument("--case3_terminal_chirp_rate", type=float, default=0.0008)
    parser.add_argument("--case3_terminal_frequency_cap", type=float, default=0.16)
    parser.add_argument("--case4_bangbang_period", type=float, default=4.2)
    parser.add_argument("--case4_bangbang_min_dwell", type=float, default=1.6)
    parser.add_argument("--case4_bangbang_hysteresis", type=float, default=0.15)
    parser.add_argument("--case4_bangbang_axial_load", type=float, default=0.10)
    parser.add_argument("--case4_bangbang_lateral_scale", type=float, default=0.75)
    parser.add_argument("--case1_lateral_base", type=float, default=0.95)
    parser.add_argument("--case1_lateral_tail", type=float, default=0.40)
    parser.add_argument("--case1_vertical_amp", type=float, default=0.35)
    parser.add_argument("--case2_lateral_amp", type=float, default=1.00)
    parser.add_argument("--case2_maneuver_freq", type=float, default=2.0 * np.pi / 50.0)
    parser.add_argument("--case2_vertical_amp", type=float, default=0.25)
    parser.add_argument("--case2_vertical_freq_scale", type=float, default=0.50)
    parser.add_argument("--stochastic_eval", action="store_true", help="使用随机采样动作进行评估")
    parser.add_argument("--eval_different_seed", action="store_true", help="每个回合用不同随机种子重置初始状态")
    parser.add_argument("--require_success_plot", action="store_true", help="优先选取有拦截成功样本进行作图")
    parser.add_argument("--require_all_hit", action="store_true", help="要求所有目标均被拦截才能视为成功")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_dir_case1", type=str, default=None)
    parser.add_argument("--model_dir_case2", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="onpolicy/scripts/results/3d_eval")
    parser.add_argument("--case1", action="store_true")
    parser.add_argument("--case2", action="store_true")
    parser.add_argument("--paper_preset_path", type=str, default=str(PROJECT_ROOT / "paper_case_presets_original_assignment_verified.npz"))
    parser.add_argument("--paper_attacker_replay", type=int, default=1)
    parser.add_argument("--paper_altitude", type=float, default=120.0)
    parser.add_argument("--paper_altitude_step", type=float, default=0.0)
    parser.add_argument("--paper_defender_climb_to_target", type=int, default=0)
    parser.add_argument("--no_tailchase_gate", type=float, default=0.0)
    parser.add_argument("--no_tailchase_rebound", type=float, default=5.0)
    parser.add_argument("--no_tailchase_penalty", type=float, default=0.0)
    parser.add_argument("--no_tailchase_terminate", action="store_true")
    parser.add_argument("--attacker_speed_min", type=float, default=12.0)
    parser.add_argument("--attacker_speed_max", type=float, default=65.0)
    parser.add_argument("--attacker_axial_min", type=float, default=-4.0)
    parser.add_argument("--attacker_axial_max", type=float, default=4.0)
    parser.add_argument("--attacker_load_limit", type=float, default=200.0)
    parser.add_argument("--attacker_yaw_scale", type=float, default=1.0)
    parser.add_argument("--attacker_pitch_scale", type=float, default=1.0)
    args = parser.parse_args()

    if not args.case1 and not args.case2:
        cases = ["case1", "case2"]
    else:
        cases = []
        if args.case1:
            cases.append("case1")
        if args.case2:
            cases.append("case2")

    case_summary_rows = []
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for case_name in cases:
        cfg = build_args(case_name, args)
        case_model_root = {
            "case1": args.model_dir_case1,
            "case2": args.model_dir_case2,
        }.get(case_name)
        model_root = Path(case_model_root) if case_model_root else Path(args.model_dir)
        env, policy, device, model_dir = collect_model(cfg, model_root)
        result = evaluate_case(cfg, env, policy, device)

        case_outdir = outdir / case_name
        case_outdir.mkdir(parents=True, exist_ok=True)
        plot_case(case_name, result, case_outdir)
        np.savez(
            case_outdir / f"{case_name}_selected_episode.npz",
            rep_att=result["rep_att"],
            rep_def=result["rep_def"],
            rep_ctrl=result["rep_ctrl"],
            rep_tgo=result["rep_tgo"],
            selected_hit_count=result["selected_hit_count"],
            selected_min_dist=result["selected_min_dist"],
            case=case_name,
        )

        row = {
            "case": case_name,
            "model": str(model_dir),
            "episodes": result["case_summary"]["episodes"],
            "attack_success_rate": result["case_summary"]["success_rate"],
            "target_success_rate": result["case_summary"]["target_success_rate"],
            "all_hit_rate": result["case_summary"]["all_hit_rate"],
            "sync_success_rate": result["case_summary"]["sync_success_rate"],
            "all_sync_rate": result["case_summary"]["all_sync_rate"],
            "mean_target_time": result["case_summary"]["mean_hit_time"],
            "max_sync_spread": result["case_summary"]["max_sync_spread"],
            "mean_sync_spread": result["case_summary"]["mean_sync_spread"],
            "mean_agent_return": result["case_summary"]["mean_agent_return"],
            "team_return": result["case_summary"]["team_return"],
        }

        hit_events = result.get("success_events", [])
        hit_fieldnames = [
            "episode",
            "step",
            "time",
            "case",
            "defender_id",
            "target_id",
            "defender_pos_x",
            "defender_pos_y",
            "defender_pos_z",
            "defender_vx",
            "defender_vy",
            "defender_vz",
            "defender_speed",
            "defender_pitch",
            "defender_yaw",
            "load_y",
            "load_z",
            "load_norm",
            "dist_to_target",
            "tgo",
        ]
        with open(case_outdir / f"{case_name}_hit_events.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=hit_fieldnames)
            writer.writeheader()
            for evt in hit_events:
                writer.writerow(evt)

        with open(case_outdir / f"{case_name}_episode_summary.csv", "w", newline="", encoding="utf-8") as f:
            summary_rows = result.get("episode_summary_events", [])
            summary_fieldnames = [
                "episode",
                "case",
                "target_num",
                "hit_count",
                "target_hit_count",
                "target_sync_count",
                "all_hit",
                "all_sync",
                "max_sync_spread",
                "mean_sync_spread",
                "mean_agent_return",
                "team_return",
            ]
            w = csv.DictWriter(f, fieldnames=summary_fieldnames)
            w.writeheader()
            for evt in summary_rows:
                w.writerow(evt)

        success_rows = result.get("success_episode_rows", [])
        with open(case_outdir / f"{case_name}_success_episodes.csv", "w", newline="", encoding="utf-8") as f:
            summary_fieldnames = [
                "episode",
                "case",
                "target_num",
                "hit_count",
                "target_hit_count",
                "target_sync_count",
                "all_hit",
                "all_sync",
                "mean_target_time",
                "max_sync_spread",
                "mean_sync_spread",
                "mean_agent_return",
                "team_return",
            ]
            w = csv.DictWriter(f, fieldnames=summary_fieldnames)
            w.writeheader()
            for evt in success_rows:
                w.writerow(evt)

        case_summary_rows.append(row)

    with open(outdir / "eval_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "case",
            "model",
            "episodes",
            "attack_success_rate",
            "target_success_rate",
            "all_hit_rate",
            "sync_success_rate",
            "all_sync_rate",
            "mean_target_time",
            "max_sync_spread",
            "mean_sync_spread",
            "mean_agent_return",
            "team_return",
        ])
        writer.writeheader()
        for row in case_summary_rows:
            writer.writerow(row)

    print("[INFO] 评测完成")
    for r in case_summary_rows:
        print(f"{r['case']}: hit_rate={r['attack_success_rate']:.3f}, target_success={r['target_success_rate']:.3f}, all_hit={r['all_hit_rate']:.3f}, sync_success={r['sync_success_rate']:.3f}, all_sync={r['all_sync_rate']:.3f}, mean_time={r['mean_target_time']}, max_spread={r['max_sync_spread']}, mean_spread={r['mean_sync_spread']}")


if __name__ == "__main__":
    main()
