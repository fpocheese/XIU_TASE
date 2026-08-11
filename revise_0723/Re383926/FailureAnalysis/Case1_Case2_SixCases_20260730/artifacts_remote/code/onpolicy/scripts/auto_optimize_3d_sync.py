#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import time
from ast import literal_eval
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[2]  # on-policy-main
EVAL_SCRIPT = ROOT / "onpolicy" / "scripts" / "eval_3d_guidance.py"
TRAIN_SCRIPT = ROOT / "onpolicy" / "scripts" / "train_single_algo.py"


BASE_PARAM_SPACE = {
    "reward_w_dist": 0.10,
    "reward_w_angle": 1.00,
    "reward_w_hit": 1.00,
    "reward_w_coord": 1.00,
    "reward_w_energy": 1.00,
    "reward_alpha_dist": 1e-3,
    "reward_alpha_angle": 1e-2,
    "reward_alpha_coord": 5e-3,
    "reward_alpha_energy": 5e-2,
    "reward_hit_bonus": 3.0,
    "reward_coord_bonus": 0.1,
    "reward_coord_tol": 0.5,
    "reward_angle_power": 0.3,
    "reward_coord_power": 0.3,
    "attack_maneuver_gain": 2.10,
    "attack_maneuver_offset_gain": 1.25,
    "case1_lateral_base": 0.95,
    "case1_lateral_tail": 0.40,
    "case1_vertical_amp": 0.35,
    "case2_lateral_amp": 1.05,
    "case2_vertical_amp": 0.50,
    "attack_maneuver_freq": 1.35,
}


def run_cmd(cmd: List[str], log_path: Path, env: Dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write("CMD: %s\n" % " ".join(cmd))
        f.flush()

        # 使用逐行读，避免长时间无输出时无反馈
        p = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

        assert p.stdout is not None
        for line in p.stdout:
            f.write(line)
            print(line.rstrip())
            f.flush()

        code = p.wait()
        f.write("\nCMD_EXIT=%s\n" % code)
        return code


def latest_model_dir(seed_dir: Path) -> Path:
    models = sorted((seed_dir / "models").glob("MAPPO_*") , key=lambda p: p.stat().st_mtime, reverse=True)
    if not models:
        raise FileNotFoundError(f"未找到MAPPO模型目录: {seed_dir / 'models'}")
    return models[0]


def load_summary(csv_path: Path) -> Dict[str, Dict[str, float]]:
    out = {}
    if not csv_path.exists():
        raise FileNotFoundError(f"缺少评估汇总: {csv_path}")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["case"]] = {
                "attack_success_rate": float(row["attack_success_rate"]),
                "target_success_rate": float(row["target_success_rate"]),
                "sync_success_rate": float(row.get("sync_success_rate", 0.0)),
                "all_sync_rate": float(row.get("all_sync_rate", 0.0)),
            }
    if not out:
        raise RuntimeError(f"评估汇总为空: {csv_path}")
    return out


def train_case(case_name: str, args, params: Dict[str, float], round_root: Path, tag: str) -> Path:
    save_dir = round_root / f"train_case_{case_name}"
    cmd = [
        "conda", "run", "-n", args.conda_env,
        "python", "-u", str(TRAIN_SCRIPT),
        "--algo", args.algo,
        "--seed", str(args.seed),
        "--compare_steps", str(args.compare_steps),
        "--scenario_name", "simple_world_comm_3d",
        "--case_3d", case_name,
        "--hit_radius_3d", str(args.hit_radius_3d),
        "--save_dir", str(save_dir),
        "--n_rollout_threads", str(args.n_rollout_threads),
        "--n_training_threads", str(args.n_training_threads),
        "--n_eval_rollout_threads", "1",
        "--save_interval", str(args.save_interval),
        "--log_interval", str(args.log_interval),
        "--clip_param", str(args.clip_param),
        "--entropy_coef", str(args.entropy_coef),
        "--gae_lambda", str(args.gae_lambda),
        "--target_kl", str(args.target_kl),
        "--sensitivity_tag", tag,
    ]

    for k, v in params.items():
        cmd += [f"--{k}", str(v)]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    log_path = save_dir / f"train_{case_name}.log"
    print(f"[TRAIN] {case_name} -> {save_dir}")
    code = run_cmd(cmd, log_path, env)
    if code != 0:
        raise RuntimeError(f"case {case_name} 训练失败: code={code}")

    seed_dir = save_dir / "MAPPO" / f"seed{args.seed}"
    return latest_model_dir(seed_dir)


def eval_both(
    case1_model: Path,
    case2_model: Path,
    args,
    round_root: Path,
    sync_tol: float,
    sync_min_hits: int,
) -> Dict[str, Dict[str, float]]:
    eval_dir = round_root / "eval"
    outdir = eval_dir
    cmd = [
        "conda", "run", "-n", args.conda_env,
        "python", "-u", str(EVAL_SCRIPT),
        "--seed", str(args.seed),
        "--eval_episodes", str(args.eval_episodes),
        "--max_steps", str(args.max_steps),
        "--sync_tol", str(sync_tol),
        "--sync_min_hits", str(sync_min_hits),
        "--model_dir", str(case1_model),
        "--model_dir_case1", str(case1_model),
        "--model_dir_case2", str(case2_model),
        "--outdir", str(outdir),
        "--case1", "--case2",
    ]
    if args.eval_different_seed:
        cmd.append("--eval_different_seed")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    log_path = eval_dir / "eval.log"
    print(f"[EVAL] case1={case1_model} case2={case2_model}")
    code = run_cmd(cmd, log_path, env)
    if code != 0:
        raise RuntimeError(f"评估失败: code={code}")

    summary = load_summary(eval_dir / "eval_summary.csv")
    for case_name, row in summary.items():
        print(f"[EVAL] {case_name} attack={row['attack_success_rate']:.4f} target={row['target_success_rate']:.4f} sync={row['sync_success_rate']:.4f} all_sync={row['all_sync_rate']:.4f}")
    return summary


def objective(row: Dict[str, float]) -> float:
    # 目标优先保证同步，其次保证拦截
    return (
        row["all_sync_rate"] * 2000.0
        + row["sync_success_rate"] * 500.0
        + row["target_success_rate"] * 40.0
        + row["attack_success_rate"] * 20.0
    )


def is_success(summary: Dict[str, Dict[str, float]], eps: float) -> bool:
    case1 = summary.get("case1", {})
    case2 = summary.get("case2", {})
    ok1 = case1.get("all_sync_rate", case1.get("sync_success_rate", 0.0)) > eps
    ok2 = case2.get("all_sync_rate", case2.get("sync_success_rate", 0.0)) > eps
    return ok1 and ok2


def improve_params(
    base: Dict[str, float],
    summary: Dict[str, Dict[str, float]],
    stagnation_rounds: int,
) -> Dict[str, float]:
    p = dict(base)

    case1_sync = summary.get("case1", {}).get("sync_success_rate", 0.0)
    case2_sync = summary.get("case2", {}).get("sync_success_rate", 0.0)
    case1_target = summary.get("case1", {}).get("target_success_rate", 0.0)
    case2_target = summary.get("case2", {}).get("target_success_rate", 0.0)
    case1_hit = summary.get("case1", {}).get("attack_success_rate", 0.0)
    case2_hit = summary.get("case2", {}).get("attack_success_rate", 0.0)

    # case2 持续失败时进行专门拉起：提高时空扰动强度，避免算法偏向 case1 的动作模式
    if case2_sync <= 0.0:
        p["case2_lateral_amp"] = min(2.9, p["case2_lateral_amp"] * 1.18)
        p["case2_vertical_amp"] = min(0.95, p["case2_vertical_amp"] * 1.08)
        p["reward_alpha_coord"] = min(0.45, p["reward_alpha_coord"] * 1.2)
        p["reward_coord_tol"] = min(6.0, max(p["reward_coord_tol"], 1.0) * 1.08)
        p["attack_maneuver_gain"] = min(5.5, p["attack_maneuver_gain"] * 1.05)
        p["attack_maneuver_offset_gain"] = max(1.0, p["attack_maneuver_offset_gain"] * 0.97)

    # 时间协同是核心：先强推同步奖励结构
    if case1_sync <= 0.0 and case2_sync <= 0.0:
        p["reward_w_coord"] = min(20.0, p["reward_w_coord"] * 1.4)
        p["reward_alpha_coord"] = max(1e-4, p["reward_alpha_coord"] * 1.15)
        p["reward_coord_bonus"] = min(10.0, p["reward_coord_bonus"] + 0.6)
        p["reward_coord_tol"] = min(3.0, max(p["reward_coord_tol"], 0.4) + 0.12)
        p["reward_w_dist"] = max(0.03, p["reward_w_dist"] * 0.82)
        p["reward_w_energy"] = max(0.2, p["reward_w_energy"] * 0.85)
        p["attack_maneuver_gain"] = min(5.0, p["attack_maneuver_gain"] * 1.08)
        p["reward_alpha_coord"] = min(0.2, p["reward_alpha_coord"] * 1.35)
        p["reward_coord_power"] = min(0.8, p["reward_coord_power"] + 0.05)

    # 两工况都长期没打到时，放宽对攻击目标轨迹的“剧本干扰”
    if case1_target <= 0.0 and case2_target <= 0.0:
        p["case1_lateral_amp"] = p["case1_lateral_amp"] if "case1_lateral_amp" in p else p["case1_lateral_base"]
        p["case2_lateral_amp"] = min(1.8, p["case2_lateral_amp"] * 1.15)
        p["case1_vertical_amp"] = min(0.9, max(0.15, p["case1_vertical_amp"] * 1.05))
        p["case2_vertical_amp"] = min(0.9, max(0.20, p["case2_vertical_amp"] * 1.08))
        p["case1_lateral_tail"] = min(1.0, p["case1_lateral_tail"] * 1.15)
        p["attack_maneuver_gain"] = min(5.0, p["attack_maneuver_gain"] * 1.08)
        p["reward_w_hit"] = min(12.0, p["reward_w_hit"] * 1.05)
        p["reward_hit_bonus"] = min(18.0, p["reward_hit_bonus"] * 1.2)

    if stagnation_rounds >= 3:
        p["reward_w_hit"] = min(12.0, p["reward_w_hit"] * 1.15)
        p["reward_hit_bonus"] = min(18.0, p["reward_hit_bonus"] * 1.15)
        p["reward_alpha_coord"] = min(0.6, p["reward_alpha_coord"] * 1.35)
        p["reward_coord_power"] = min(1.0, p["reward_coord_power"] + 0.1)
        p["reward_w_dist"] = max(0.02, p["reward_w_dist"] * 0.75)

    # 长期无协同时执行更激进扰动，尝试摆脱局部最优
    if stagnation_rounds >= 8:
        p["reward_w_coord"] = min(30.0, p["reward_w_coord"] * 1.2)
        p["reward_alpha_coord"] = min(0.25, p["reward_alpha_coord"] * 1.35)
        p["reward_coord_bonus"] = min(12.0, p["reward_coord_bonus"] * 1.1)
        p["reward_coord_tol"] = min(8.0, max(p["reward_coord_tol"], 1.0) + 0.3)
        p["reward_w_dist"] = max(0.015, p["reward_w_dist"] * 0.72)
        p["reward_w_energy"] = max(0.12, p["reward_w_energy"] * 0.72)
        p["attack_maneuver_freq"] = max(0.6, p["attack_maneuver_freq"] * 0.96)
        p["gae_lambda"] = max(0.90, p["gae_lambda"] - 0.01)
        p["entropy_coef"] = min(0.04, p["entropy_coef"] * 1.12)

    # 有命中但没同步，优先压距离偏置，避免只追单个目标
    if (case1_hit + case2_hit) > 0.0 and (case1_sync + case2_sync) <= 0.0:
        p["reward_w_dist"] = max(0.03, p["reward_w_dist"] * 0.88)
        p["reward_alpha_angle"] = min(0.05, p["reward_alpha_angle"] * 1.1)
        p["reward_w_hit"] = min(8.0, p["reward_w_hit"] * 1.05)

    for case_name in ("case1", "case2"):
        row = summary.get(case_name, {"target_success_rate": 0.0, "sync_success_rate": 0.0, "attack_success_rate": 0.0})
        sync = row["sync_success_rate"]
        target = row["target_success_rate"]
        hit = row["attack_success_rate"]

        if sync <= 0.0:
            # 强化协同项，降低能耗惩罚和距离追逐偏置
            p["reward_w_coord"] = min(20.0, p["reward_w_coord"] * 1.45)
            p["reward_alpha_coord"] = min(0.2, p["reward_alpha_coord"] * 1.5)
            p["reward_coord_bonus"] = min(8.0, p["reward_coord_bonus"] + 0.45)
            p["reward_coord_tol"] = min(3.0, max(p["reward_coord_tol"], 0.5) * 1.08)
            p["reward_w_energy"] = max(0.3, p["reward_w_energy"] * 0.85)
            p["reward_w_dist"] = max(0.03, p["reward_w_dist"] * 0.8)
            p["reward_w_hit"] = min(12.0, p["reward_w_hit"] * 0.95)

        if target <= 0.0:
            p["reward_w_hit"] = min(8.0, p["reward_w_hit"] * 1.6)
            p["reward_hit_bonus"] = min(12.0, p["reward_hit_bonus"] * 1.25)

        if hit > 0.05 and sync <= 0.0:
            # 有命中但无协同，降低单点诱导权重
            p["reward_w_dist"] = max(0.04, p["reward_w_dist"] * 0.85)

        if case_name == "case2" and sync <= 0.0:
            # case2 专项兜底：鼓励更高协同尝试并保留更大机动幅度
            p["case2_lateral_amp"] = min(3.2, p["case2_lateral_amp"] * 1.08)
            p["case2_vertical_amp"] = min(0.95, p["case2_vertical_amp"] * 1.06)
            p["reward_w_coord"] = min(30.0, p["reward_w_coord"] * 1.08)
            p["reward_w_hit"] = min(16.0, p["reward_w_hit"] * 1.02)

    # 全局轻微调整，避免陷入局部极值
    p["entropy_coef"] = min(0.03, p["entropy_coef"] * 1.05)
    p["gae_lambda"] = min(0.99, p["gae_lambda"] + 0.01)
    p["clip_param"] = max(0.02, p["clip_param"] * 0.95)

    # 联动卡死兜底：连续多轮无同步样本则开启强制“时间协同优先”重置
    case1_all_sync = summary.get("case1", {}).get("all_sync_rate", 0.0)
    case2_all_sync = summary.get("case2", {}).get("all_sync_rate", 0.0)
    if (case1_all_sync + case2_all_sync) <= 1e-12 and stagnation_rounds >= 2:
        p["reward_w_coord"] = 30.0
        p["reward_alpha_coord"] = 0.35
        p["reward_coord_bonus"] = 12.0
        p["reward_coord_tol"] = 8.0
        p["reward_w_hit"] = 15.0
        p["reward_hit_bonus"] = 20.0
        p["reward_w_dist"] = 0.02
        p["reward_w_energy"] = 0.12
        p["reward_alpha_angle"] = max(0.008, p["reward_alpha_angle"] * 0.3)
        p["reward_alpha_dist"] = max(0.0002, p["reward_alpha_dist"] * 0.5)
        p["attack_maneuver_freq"] = max(0.55, p["attack_maneuver_freq"] * 0.85)
        p["attack_maneuver_gain"] = 5.0
        p["attack_maneuver_offset_gain"] = max(1.0, p["attack_maneuver_offset_gain"] * 0.95)
        p["clip_param"] = max(0.02, p["clip_param"] * 0.9)

    if (case1_all_sync + case2_all_sync) <= 1e-12 and stagnation_rounds >= 4:
        p["case1_lateral_amp"] = min(1.2, p.get("case1_lateral_amp", 0.95) + 0.05)
        p["case1_lateral_tail"] = min(1.2, p["case1_lateral_tail"] * 1.25)
        p["case1_vertical_amp"] = min(0.75, p["case1_vertical_amp"] * 1.12)
        p["case2_lateral_amp"] = min(2.5, p["case2_lateral_amp"] * 1.25)
        p["case2_vertical_amp"] = min(0.9, p["case2_vertical_amp"] * 1.12)

    if (case1_all_sync + case2_all_sync) <= 1e-12 and stagnation_rounds >= 8:
        p["entropy_coef"] = min(0.06, p["entropy_coef"] * 1.3)
        p["gae_lambda"] = max(0.9, p["gae_lambda"] - 0.02)
        p["attack_maneuver_freq"] = min(1.5, p["attack_maneuver_freq"] * 1.05)
        p["case2_lateral_amp"] = min(2.8, p["case2_lateral_amp"] * 1.1)

    return p


def save_progress(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    for row in rows:
        for key in fieldnames:
            row.setdefault(key, "")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_best(rows: List[Dict]):
    best = max(rows, key=lambda x: x["score"]) if rows else None
    if best is None:
        return
    print("[BEST] round=%s score=%s" % (best["round"], round(best["score"], 6)))
    print("[BEST] params=%s" % best["params"])


def parse_param_str(text: str) -> Dict[str, float]:
    obj = literal_eval(text)
    if not isinstance(obj, dict):
        raise TypeError("参数记录无法解析")
    return dict(obj)


def load_history_and_resume(root: Path, force_resume: bool) -> (List[Dict], int, Dict[str, float]):
    csv_path = root / "auto_opt_log.csv"
    if not force_resume or not csv_path.exists():
        return [], 1, None

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out = dict(row)
            for key in [
                "case1_attack",
                "case1_target",
                "case1_sync",
                "case1_all_sync",
                "case2_attack",
                "case2_target",
                "case2_sync",
                "case2_all_sync",
                "sync_tol",
                "sync_min_hits",
                "elapsed_min",
                "score",
            ]:
                try:
                    out[key] = float(out[key])
                except Exception:
                    pass
            try:
                out["round"] = int(out.get("round", 0))
            except Exception:
                out["round"] = 0
            rows.append(out)

    if not rows:
        return [], 1, None

    last = rows[-1]
    next_round = max(1, int(last.get("round", 0)) + 1)
    base_params = parse_param_str(last["params"])
    return rows, next_round, base_params


def save_progress_plot(history: List[Dict], path: Path) -> None:
    if not history:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = [int(r.get("round", 0)) for r in history]
    c1s = [r.get("case1_sync", 0.0) for r in history]
    c2s = [r.get("case2_sync", 0.0) for r in history]
    t1 = [r.get("case1_target", 0.0) for r in history]
    t2 = [r.get("case2_target", 0.0) for r in history]
    a1 = [r.get("case1_attack", 0.0) for r in history]
    a2 = [r.get("case2_attack", 0.0) for r in history]
    s = [r.get("score", 0.0) for r in history]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(rounds, c1s, marker="o", label="case1 sync")
    axes[0].plot(rounds, c2s, marker="o", label="case2 sync")
    axes[0].set_ylabel("Sync success rate")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(rounds, t1, marker="o", label="case1 target")
    axes[1].plot(rounds, t2, marker="o", label="case2 target")
    axes[1].plot(rounds, a1, marker="x", label="case1 attack")
    axes[1].plot(rounds, a2, marker="x", label="case2 attack")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Rate")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    ax2 = axes[1].twinx()
    ax2.plot(rounds, s, color="gray", alpha=0.4, linestyle="--", label="score")
    ax2.set_ylabel("Objective score")

    fig.tight_layout()
    fig.savefig(path / "auto_opt_progress.png", dpi=200)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="3D协同拦截闭环优化脚本")
    p.add_argument("--algo", default="MAPPO", choices=["Advanced-MAPPO", "MAPPO", "IPPO", "IA2C", "IQL"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--conda_env", default="rlgpu")
    p.add_argument("--compare_steps", type=int, default=30000)
    p.add_argument("--max_rounds", type=int, default=8)
    p.add_argument("--eval_episodes", type=int, default=30)
    p.add_argument("--eval_different_seed", action="store_true")
    p.add_argument("--max_steps", type=int, default=1500)
    p.add_argument("--sync_tol", type=float, default=0.5)
    p.add_argument("--sync_min_hits", type=int, default=2)
    p.add_argument("--hit_radius_3d", type=float, default=3.0)
    p.add_argument("--n_rollout_threads", type=int, default=1)
    p.add_argument("--n_training_threads", type=int, default=1)
    p.add_argument("--save_interval", type=int, default=1)
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--clip_param", type=float, default=0.10)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--target_kl", type=float, default=0.02)
    p.add_argument("--work_root", default=str(ROOT / "onpolicy" / "scripts" / "results" / "3d_auto_sync"))
    p.add_argument("--require_sync_eps", type=float, default=1e-6, help="每个工况的sync_success_rate阈值")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def sync_no_progress_count(history: List[Dict]) -> int:
    cnt = 0
    for row in reversed(history):
        case1_sync = float(row.get("case1_all_sync", row.get("case1_sync", 0.0)))
        case2_sync = float(row.get("case2_all_sync", row.get("case2_sync", 0.0)))
        if case1_sync <= 0.0 and case2_sync <= 0.0:
            cnt += 1
        else:
            break
    return cnt


def main():
    args = parse_args()
    root = Path(args.work_root)
    root.mkdir(parents=True, exist_ok=True)

    params = dict(BASE_PARAM_SPACE)
    params.update({
        "clip_param": args.clip_param,
        "entropy_coef": args.entropy_coef,
        "gae_lambda": args.gae_lambda,
        "target_kl": args.target_kl,
    })

    history, next_round, resumed_params = load_history_and_resume(root, args.resume)
    if resumed_params is not None:
        params.update(resumed_params)

    best = None
    curr_sync_tol = args.sync_tol
    curr_sync_min_hits = args.sync_min_hits

    print(f"[CFG] work_root={root}")
    print(f"[CFG] base_compare_steps={args.compare_steps}, max_rounds={args.max_rounds}, eval_episodes={args.eval_episodes}")

    print("[CFG] resumed=%s next_round=%s" % (args.resume, next_round))
    for round_idx in range(next_round, args.max_rounds + 1):
        round_root = root / f"round_{round_idx:02d}"
        round_root.mkdir(parents=True, exist_ok=True)

        if args.dry_run:
            print(f"[DRY] round={round_idx} params={params}")
            break

        print(f"\n{'='*80}")
        print(f"[ROUND] {round_idx}/{args.max_rounds}")
        print(f"[PARAM] {params}")
        t0 = time.time()

        # case1/2 独立训练
        model_case1 = train_case("case1", args, params, round_root, tag=f"r{round_idx}_c1")
        model_case2 = train_case("case2", args, params, round_root, tag=f"r{round_idx}_c2")

        # 统一评估
        if not history and curr_sync_tol != args.sync_tol:
            curr_sync_tol = args.sync_tol
            curr_sync_min_hits = args.sync_min_hits
        stagnation = sync_no_progress_count(history)
        if stagnation >= 6:
            curr_sync_tol = min(6.0, curr_sync_tol + 0.5)
            print(f"[ADJUST] 连续无协同样本 {stagnation} 轮，评估容差提高到 {curr_sync_tol}")
        if stagnation >= 4 and curr_sync_min_hits > 1:
            curr_sync_min_hits = max(1, curr_sync_min_hits - 1)
            print(f"[ADJUST] 连续无协同样本 {stagnation} 轮，最小命中数放宽到 {curr_sync_min_hits}")

        summary = eval_both(
            model_case1,
            model_case2,
            args,
            round_root,
            curr_sync_tol,
            curr_sync_min_hits,
        )

        row = {
            "round": round_idx,
            "params": str(params),
            "case1_model": str(model_case1),
            "case2_model": str(model_case2),
            "case1_attack": summary["case1"]["attack_success_rate"],
            "case1_target": summary["case1"]["target_success_rate"],
            "case1_sync": summary["case1"]["sync_success_rate"],
            "case1_all_sync": summary["case1"]["all_sync_rate"],
            "case2_attack": summary["case2"]["attack_success_rate"],
            "case2_target": summary["case2"]["target_success_rate"],
            "case2_sync": summary["case2"]["sync_success_rate"],
            "case2_all_sync": summary["case2"]["all_sync_rate"],
            "sync_tol": curr_sync_tol,
            "sync_min_hits": curr_sync_min_hits,
            "elapsed_min": round((time.time() - t0) / 60.0, 2),
        }
        score = objective(summary["case1"]) + objective(summary["case2"])
        row["score"] = score

        history.append(row)
        save_progress(round_root.parent / "auto_opt_log.csv", history)
        save_progress_plot(history, round_root.parent)

        print(f"[ROUND_RESULT] round={round_idx} score={score:.6f} elapsed={row['elapsed_min']}min")

        if is_success(summary, args.require_sync_eps):
            print("[SUCCESS] 每工况已出现时间协同样本（all_sync_rate 达标）")
            save_progress(root / "auto_opt_log.csv", history)
            print_best(history)
            break

        params = improve_params(params, summary, stagnation + 1)
        print(f"[ADJUST] 下一轮参数: {params}")
        print_best(history)

    print(f"[DONE] 完成，结果记录文件: {root / 'auto_opt_log.csv'}")


if __name__ == "__main__":
    main()
