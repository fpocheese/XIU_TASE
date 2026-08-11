#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from eval_3d_guidance import build_args, collect_model


CASE_PREFIX = {"case1": "nopn", "case2": "sin"}
CASE_FOLDER = {"case1": "mappo_success_nopn", "case2": "mappo_success_sin"}


class DelayLine:
    def __init__(self, delay_steps, initial_value):
        self.delay_steps = max(0, int(delay_steps))
        self.queue = [np.array(initial_value, dtype=np.float32).copy() for _ in range(self.delay_steps)]

    def push(self, value):
        value = np.array(value, dtype=np.float32).copy()
        if self.delay_steps <= 0:
            return value
        self.queue.append(value)
        return self.queue.pop(0)


class InterceptorPolicyNode:
    """One interceptor-side policy endpoint separated from the simulator."""

    def __init__(self, idx, policy, args, obs0, action_dim, rng, hil=False, hil_cfg=None):
        self.idx = idx
        self.policy = policy
        self.args = args
        self.rng = rng
        self.hil = hil
        self.hil_cfg = hil_cfg or {}
        self.rnn_state = np.zeros((1, args.recurrent_N, args.hidden_size), dtype=np.float32)
        self.mask = np.ones((1, 1), dtype=np.float32)
        self.last_action = np.zeros(action_dim, dtype=np.float32)
        self.obs_delay = DelayLine(self.hil_cfg.get("obs_delay_steps", 0) if hil else 0, obs0)
        self.action_delay = DelayLine(
            self.hil_cfg.get("action_delay_steps", 0) if hil else 0,
            np.zeros(action_dim, dtype=np.float32),
        )

    def reset(self, obs0):
        self.rnn_state.fill(0.0)
        self.mask.fill(1.0)
        self.last_action.fill(0.0)
        self.obs_delay = DelayLine(self.hil_cfg.get("obs_delay_steps", 0) if self.hil else 0, obs0)
        self.action_delay = DelayLine(
            self.hil_cfg.get("action_delay_steps", 0) if self.hil else 0,
            np.zeros_like(self.last_action),
        )

    def set_done(self, done):
        self.mask[0, 0] = 0.0 if done else 1.0

    def _sense(self, obs):
        sensed = self.obs_delay.push(obs)
        if self.hil:
            std = float(self.hil_cfg.get("obs_noise_std", 0.0))
            if std > 0.0:
                sensed = sensed + self.rng.normal(0.0, std, size=sensed.shape).astype(np.float32)
        return sensed.astype(np.float32)

    def _actuate(self, action):
        applied = self.action_delay.push(action)
        if self.hil:
            drop_prob = float(self.hil_cfg.get("drop_prob", 0.0))
            if drop_prob > 0.0 and self.rng.random() < drop_prob:
                applied = self.last_action.copy()
            std = float(self.hil_cfg.get("action_noise_std", 0.0))
            if std > 0.0:
                applied = applied + self.rng.normal(0.0, std, size=applied.shape).astype(np.float32)
        applied = np.clip(applied, -1.0, 1.0).astype(np.float32)
        self.last_action = applied.copy()
        return applied

    def step(self, obs, deterministic=True):
        sensed = self._sense(obs)
        with torch.no_grad():
            action, next_rnn = self.policy.act(
                sensed.reshape(1, -1),
                self.rnn_state,
                self.mask,
                deterministic=deterministic,
            )
        if hasattr(next_rnn, "detach"):
            self.rnn_state = next_rnn.detach().cpu().numpy()
        else:
            self.rnn_state = np.asarray(next_rnn, dtype=np.float32)
        action_np = action.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return self._actuate(action_np)


def filtered_eval_args(args):
    names = [
        "seed", "eval_episodes", "max_steps", "hit_radius_3d", "sync_tol", "sync_min_hits",
        "defender_guidance_base_gain", "defender_guidance_tau", "defender_guidance_lead",
        "defender_residual_scale", "defender_load_limit", "defender_axial_min",
        "defender_axial_max", "defender_sync_speed_gain", "defender_sync_tgo_ref",
        "target_assignment_mode", "target_assignment_spread_weight", "stochastic_eval",
        "eval_different_seed", "require_success_plot", "require_all_hit", "model_dir",
        "model_dir_case1", "model_dir_case2", "outdir", "case1", "case2",
    ]
    return argparse.Namespace(**{name: getattr(args, name) for name in names})


def collect_histories(env):
    defenders = [ag for ag in env.world.agents if ag.adversary]
    attackers = [ag for ag in env.world.agents if not ag.adversary]

    def_pos, att_pos, ctrl, tgo, vel, dist = [], [], [], [], [], []
    for ag in defenders:
        target = env.world.agents[ag.target]
        def_pos.append(ag.state.p_pos.copy())
        load = ag.state.load if ag.state.load is not None else np.zeros(3)
        ctrl.append(np.array(load, dtype=np.float32).copy())
        d = float(np.linalg.norm(target.state.p_pos - ag.state.p_pos))
        speed = float(ag.state.v_vel[0])
        tgo.append(d / max(speed, 1.0))
        dist.append(d)
        vel.append(np.array([speed, float(ag.state.v_vel[2])], dtype=np.float32))
    for ag in attackers:
        att_pos.append(ag.state.p_pos.copy())
    return (
        np.array(def_pos, dtype=np.float32),
        np.array(att_pos, dtype=np.float32),
        np.array(ctrl, dtype=np.float32),
        np.array(tgo, dtype=np.float32),
        np.array(vel, dtype=np.float32),
        np.array(dist, dtype=np.float32),
    )


def summarize_episode(args, adv_agents, hit_time):
    tgt_ids = sorted({int(ag.target) for ag in adv_agents})
    hit_targets = np.zeros(len(tgt_ids), dtype=bool)
    sync_targets = np.zeros(len(tgt_ids), dtype=bool)
    per_target_time = np.full(len(tgt_ids), np.nan, dtype=np.float32)
    per_target_spread = np.full(len(tgt_ids), np.nan, dtype=np.float32)

    for j, tid in enumerate(tgt_ids):
        group = [i for i, ag in enumerate(adv_agents) if int(ag.target) == tid]
        times = [hit_time[i] for i in group if np.isfinite(hit_time[i])]
        if times:
            hit_targets[j] = True
            per_target_time[j] = float(np.min(times))
            if len(times) >= 2:
                per_target_spread[j] = float(np.max(times) - np.min(times))
            required = len(group) if args.sync_min_hits <= 0 else max(1, min(len(group), args.sync_min_hits))
            if len(times) >= required and (np.max(times) - np.min(times)) <= args.sync_tol:
                sync_targets[j] = True

    return {
        "target_num": len(tgt_ids),
        "hit_count": int(np.count_nonzero(np.isfinite(hit_time))),
        "target_hit_count": int(np.count_nonzero(hit_targets)),
        "target_sync_count": int(np.count_nonzero(sync_targets)),
        "all_hit": bool(np.count_nonzero(hit_targets) >= len(tgt_ids)),
        "all_sync": bool(np.count_nonzero(sync_targets) >= len(tgt_ids)),
        "mean_target_time": float(np.nanmean(per_target_time)) if np.isfinite(per_target_time).any() else float("nan"),
        "max_sync_spread": float(np.nanmax(per_target_spread)) if np.isfinite(per_target_spread).any() else float("nan"),
        "mean_sync_spread": float(np.nanmean(per_target_spread)) if np.isfinite(per_target_spread).any() else float("nan"),
    }


def evaluate_case_separated(args, env, policy, case_name, mode):
    n_agents = len(env.world.policy_agents)
    action_dim = env.action_space[0].shape[0]
    rng = np.random.default_rng(args.seed + (1000 if mode == "hil" else 0))
    hil_ids = {int(x) for x in args.hil_interceptors.split(",") if x.strip() != ""}
    hil_cfg = {
        "obs_delay_steps": args.hil_obs_delay_steps,
        "action_delay_steps": args.hil_action_delay_steps,
        "obs_noise_std": args.hil_obs_noise_std,
        "action_noise_std": args.hil_action_noise_std,
        "drop_prob": args.hil_drop_prob,
    }

    all_rows, success_rows, hit_events = [], [], []
    best = None

    for ep in range(args.eval_episodes):
        if args.eval_different_seed:
            eval_seed = args.seed + ep
            np.random.seed(eval_seed)
            torch.manual_seed(eval_seed)
        obs = np.asarray(env.reset(), dtype=np.float32)
        nodes = [
            InterceptorPolicyNode(i, policy, args, obs[i], action_dim, rng, hil=(mode == "hil" and i in hil_ids), hil_cfg=hil_cfg)
            for i in range(n_agents)
        ]

        adv_agents = [ag for ag in env.world.agents if ag.adversary]
        hit_time = np.full(n_agents, np.nan, dtype=np.float32)
        ep_hits = []
        hist_def, hist_att, hist_ctrl, hist_tgo, hist_vel, hist_dist = [], [], [], [], [], []

        for step in range(args.max_steps):
            sensed_obs = np.vstack([node._sense(obs[i]) for i, node in enumerate(nodes)]).astype(np.float32)
            rnn_states = np.vstack([node.rnn_state for node in nodes]).astype(np.float32)
            if args.reset_rnn_each_step:
                rnn_states.fill(0.0)
            masks = np.vstack([node.mask for node in nodes]).astype(np.float32)
            with torch.no_grad():
                raw_actions, next_rnn = policy.act(
                    sensed_obs,
                    rnn_states,
                    masks,
                    deterministic=not args.stochastic_eval,
                )
            raw_actions = raw_actions.detach().cpu().numpy().astype(np.float32)
            if hasattr(next_rnn, "detach"):
                next_rnn = next_rnn.detach().cpu().numpy().astype(np.float32)
            else:
                next_rnn = np.asarray(next_rnn, dtype=np.float32)
            actions = []
            for i, node in enumerate(nodes):
                node.rnn_state = (
                    np.zeros_like(next_rnn[i:i + 1])
                    if args.reset_rnn_each_step
                    else next_rnn[i:i + 1]
                )
                actions.append(node._actuate(raw_actions[i]))
            actions = np.vstack(actions).astype(np.float32)
            obs_next, _, dones, _ = env.step(actions, step)
            obs = np.asarray(obs_next, dtype=np.float32)

            def_pos, att_pos, ctrl, tgo, vel, dist = collect_histories(env)
            hist_def.append(def_pos)
            hist_att.append(att_pos)
            hist_ctrl.append(ctrl)
            hist_tgo.append(tgo)
            hist_vel.append(vel)
            hist_dist.append(dist)

            for i, ag in enumerate(adv_agents):
                if getattr(ag.state, "actual_hit", False) and not np.isfinite(hit_time[i]):
                    hit_time[i] = (step + 1) * env.dt
                    target = env.world.agents[ag.target]
                    event = {
                        "episode": ep + 1,
                        "step": step + 1,
                        "time": float(hit_time[i]),
                        "case": case_name,
                        "mode": mode,
                        "defender_id": int(ag.namenumber),
                        "target_id": int(ag.target),
                        "dist_to_target": float(np.linalg.norm(target.state.p_pos - ag.state.p_pos)),
                    }
                    ep_hits.append(event)
                    hit_events.append(event)

            dones_arr = np.asarray(dones, dtype=bool)
            for i, node in enumerate(nodes):
                node.set_done(bool(dones_arr[i]))
            if dones_arr.all():
                break

        summary = summarize_episode(args, adv_agents, hit_time)
        summary.update({"episode": ep + 1, "case": case_name, "mode": mode})
        all_rows.append(summary)
        if summary["all_hit"] and summary["all_sync"]:
            success_rows.append(summary)

        episode_data = {
            "rep_def": np.asarray(hist_def, dtype=np.float32),
            "rep_att": np.asarray(hist_att, dtype=np.float32),
            "rep_ctrl": np.asarray(hist_ctrl, dtype=np.float32),
            "rep_tgo": np.asarray(hist_tgo, dtype=np.float32),
            "rep_vel": np.asarray(hist_vel, dtype=np.float32),
            "rep_dist": np.asarray(hist_dist, dtype=np.float32),
            "hit_time": hit_time,
            "summary": summary,
            "hit_events": ep_hits,
        }
        score = (1000 if summary["all_hit"] and summary["all_sync"] else 0) + summary["hit_count"]
        if best is None or score > best[0]:
            best = (score, episode_data)

    def mean_bool(key):
        return float(np.mean([1.0 if row[key] else 0.0 for row in all_rows])) if all_rows else 0.0

    case_summary = {
        "case": case_name,
        "mode": mode,
        "episodes": args.eval_episodes,
        "attack_success_rate": float(np.mean([row["hit_count"] / max(1, n_agents) for row in all_rows])) if all_rows else 0.0,
        "target_success_rate": float(np.mean([row["target_hit_count"] / max(1, row["target_num"]) for row in all_rows])) if all_rows else 0.0,
        "all_hit_rate": mean_bool("all_hit"),
        "sync_success_rate": float(np.mean([row["target_sync_count"] / max(1, row["target_num"]) for row in all_rows])) if all_rows else 0.0,
        "all_sync_rate": mean_bool("all_sync"),
        "mean_target_time": float(np.nanmean([row["mean_target_time"] for row in all_rows])) if all_rows else float("nan"),
        "max_sync_spread": float(np.nanmax([row["max_sync_spread"] for row in all_rows])) if all_rows else float("nan"),
        "mean_sync_spread": float(np.nanmean([row["mean_sync_spread"] for row in all_rows])) if all_rows else float("nan"),
    }
    return case_summary, all_rows, success_rows, hit_events, best[1]


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_v9_folder(v9_base, case_name, best_episode, episode_rows):
    folder = v9_base / CASE_FOLDER[case_name]
    folder.mkdir(parents=True, exist_ok=True)
    deff = best_episode["rep_def"]
    att = best_episode["rep_att"]
    ctrl = best_episode["rep_ctrl"]
    tgo = best_episode["rep_tgo"]
    vel = best_episode["rep_vel"]
    dist = best_episode["rep_dist"]

    steps = deff.shape[0]
    agentspos = np.zeros((steps, 56), dtype=np.float32)
    agentsall = np.zeros((steps, 40), dtype=np.float32)
    agentsvel = np.zeros((steps, 40), dtype=np.float32)
    agentstimetgo = np.zeros((steps, 40), dtype=np.float32)

    for i in range(min(20, deff.shape[1])):
        agentspos[:, 2 * i] = deff[:, i, 0]
        agentspos[:, 2 * i + 1] = deff[:, i, 1]
        agentsall[:, 2 * i] = ctrl[:, i, 0]
        agentsall[:, 2 * i + 1] = ctrl[:, i, 1]
        agentsvel[:, 2 * i] = vel[:, i, 0]
        agentsvel[:, 2 * i + 1] = vel[:, i, 1]
        agentstimetgo[:, 2 * i] = tgo[:, i]
        agentstimetgo[:, 2 * i + 1] = dist[:, i]
    for j in range(min(8, att.shape[1])):
        agentspos[:, 40 + 2 * j] = att[:, j, 0]
        agentspos[:, 40 + 2 * j + 1] = att[:, j, 1]

    np.savetxt(folder / "agentspos.txt", agentspos, fmt="%.9f")
    np.savetxt(folder / "agentsall.txt", agentsall, fmt="%.9f")
    np.savetxt(folder / "agentsvel.txt", agentsvel, fmt="%.9f")
    np.savetxt(folder / "agentstimetgo.txt", agentstimetgo, fmt="%.9f")

    eval_rows = []
    for row in episode_rows:
        miss_rate = 1.0 - row["target_hit_count"] / max(1, row["target_num"])
        eval_rows.append([1.0 if row["all_sync"] else 0.0, row["mean_sync_spread"], miss_rate])
    eval_arr = np.asarray(eval_rows if eval_rows else [[0.0, np.nan, 1.0]], dtype=np.float32)
    if case_name == "case2":
        eval_dir = folder / "sinmappo_eval"
        eval_dir.mkdir(exist_ok=True)
        np.savetxt(eval_dir / "agentseval.txt", eval_arr, fmt="%.9f")
    np.savetxt(folder / f"{CASE_FOLDER[case_name]}_eval.txt", eval_arr, fmt="%.9f")


def run_case(args, case_name, mode):
    cfg = build_args(case_name, filtered_eval_args(args))
    cfg.reset_rnn_each_step = args.reset_rnn_each_step
    for name in [
        "hil_interceptors",
        "hil_obs_delay_steps",
        "hil_action_delay_steps",
        "hil_obs_noise_std",
        "hil_action_noise_std",
        "hil_drop_prob",
    ]:
        setattr(cfg, name, getattr(args, name))
    model_root = Path(args.model_dir_case1 if case_name == "case1" else args.model_dir_case2)
    env, policy, _, model_dir = collect_model(cfg, model_root)
    summary, episode_rows, success_rows, hit_events, best = evaluate_case_separated(cfg, env, policy, case_name, mode)
    summary["model"] = str(model_dir)

    out = Path(args.outdir) / mode / case_name
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / f"{case_name}_{mode}_selected_episode.npz",
        rep_def=best["rep_def"],
        rep_att=best["rep_att"],
        rep_ctrl=best["rep_ctrl"],
        rep_tgo=best["rep_tgo"],
        rep_vel=best["rep_vel"],
        rep_dist=best["rep_dist"],
        hit_time=best["hit_time"],
    )
    with open(out / "hil_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "mode": mode,
            "hil_interceptors": args.hil_interceptors,
            "obs_delay_steps": args.hil_obs_delay_steps,
            "action_delay_steps": args.hil_action_delay_steps,
            "obs_noise_std": args.hil_obs_noise_std,
            "action_noise_std": args.hil_action_noise_std,
            "drop_prob": args.hil_drop_prob,
            "reset_rnn_each_step": args.reset_rnn_each_step,
        }, f, indent=2)

    summary_fields = [
        "episode", "case", "mode", "target_num", "hit_count", "target_hit_count",
        "target_sync_count", "all_hit", "all_sync", "mean_target_time",
        "max_sync_spread", "mean_sync_spread",
    ]
    write_csv(out / f"{case_name}_episode_summary.csv", episode_rows, summary_fields)
    write_csv(out / f"{case_name}_success_episodes.csv", success_rows, summary_fields)
    write_csv(out / f"{case_name}_hit_events.csv", hit_events, [
        "episode", "step", "time", "case", "mode", "defender_id", "target_id", "dist_to_target",
    ])
    export_v9_folder(Path(args.outdir) / f"v9_{mode}", case_name, best, episode_rows)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=2400)
    parser.add_argument("--sync_tol", type=float, default=3.0)
    parser.add_argument("--sync_min_hits", type=int, default=0)
    parser.add_argument("--hit_radius_3d", type=float, default=12.0)
    parser.add_argument("--defender_guidance_base_gain", type=float, default=1.0)
    parser.add_argument("--defender_guidance_tau", type=float, default=0.55)
    parser.add_argument("--defender_guidance_lead", type=float, default=1.2)
    parser.add_argument("--defender_residual_scale", type=float, default=0.05)
    parser.add_argument("--defender_load_limit", type=float, default=1.6)
    parser.add_argument("--defender_axial_min", type=float, default=-0.8)
    parser.add_argument("--defender_axial_max", type=float, default=1.0)
    parser.add_argument("--defender_sync_speed_gain", type=float, default=0.34)
    parser.add_argument("--defender_sync_tgo_ref", type=str, default="mean")
    parser.add_argument("--target_assignment_mode", type=str, default="dynamic")
    parser.add_argument("--target_assignment_spread_weight", type=float, default=0.0)
    parser.add_argument("--stochastic_eval", action="store_true")
    parser.add_argument("--eval_different_seed", action="store_true", default=True)
    parser.add_argument("--require_success_plot", action="store_true")
    parser.add_argument("--require_all_hit", action="store_true", default=True)
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--model_dir_case1", type=str, required=True)
    parser.add_argument("--model_dir_case2", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="../../hil_v9_results")
    parser.add_argument("--case1", action="store_true")
    parser.add_argument("--case2", action="store_true")
    parser.add_argument("--modes", type=str, default="baseline,hil")
    parser.add_argument("--hil_interceptors", type=str, default="0,1,2,3,4")
    parser.add_argument("--hil_obs_delay_steps", type=int, default=2)
    parser.add_argument("--hil_action_delay_steps", type=int, default=1)
    parser.add_argument("--hil_obs_noise_std", type=float, default=0.003)
    parser.add_argument("--hil_action_noise_std", type=float, default=0.02)
    parser.add_argument("--hil_drop_prob", type=float, default=0.01)
    parser.add_argument(
        "--reset_rnn_each_step",
        action="store_true",
        help="Inference-only GRU ablation: zero the recurrent state before every action.",
    )
    args = parser.parse_args()

    cases = []
    if args.case1:
        cases.append("case1")
    if args.case2:
        cases.append("case2")
    if not cases:
        cases = ["case1", "case2"]

    summaries = []
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        for case_name in cases:
            print(f"[RUN] mode={mode} case={case_name}")
            summary = run_case(args, case_name, mode)
            summaries.append(summary)
            print(
                f"[DONE] {mode}/{case_name}: hit={summary['attack_success_rate']:.3f}, "
                f"target={summary['target_success_rate']:.3f}, all_hit={summary['all_hit_rate']:.3f}, "
                f"sync={summary['sync_success_rate']:.3f}, all_sync={summary['all_sync_rate']:.3f}, "
                f"max_spread={summary['max_sync_spread']}"
            )

    fieldnames = [
        "case", "mode", "model", "episodes", "attack_success_rate", "target_success_rate",
        "all_hit_rate", "sync_success_rate", "all_sync_rate", "mean_target_time",
        "max_sync_spread", "mean_sync_spread",
    ]
    write_csv(Path(args.outdir) / "eval_summary.csv", summaries, fieldnames)


if __name__ == "__main__":
    main()
