#!/usr/bin/env python3
"""Plot five replayed successful partial-failure trajectories per case.

Each selected episode is rendered as one three-dimensional trajectory and one
horizontal-plane trajectory.  The figures use only recorded frozen-policy
replay states; no trajectory coordinates are synthesized or modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


DT = 0.05
COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#8C564B",
    "#6A3D9A",
]
METRICS = ("E_co_time_s", "E_n_g", "E_miss_m", "E_t_s")


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.8,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


def read_rows_by_seed(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["seed"]): row for row in csv.DictReader(handle)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_episode(case: str, seed: int, replay_root: Path) -> dict:
    source_dir = replay_root / f"seed_{seed}"
    npz_path = source_dir / f"{case}_representative_success.npz"
    row = read_single_row(source_dir / "episodes.csv")
    with np.load(npz_path) as data:
        result = {key: np.asarray(data[key]) for key in data.files}
    result.update(
        {
            "case": case,
            "seed_value": seed,
            "source_dir": source_dir,
            "npz_path": npz_path,
            "row": row,
        }
    )
    if int(row["interception_success"]) != 1 or int(row["targets_covered"]) != 8:
        raise RuntimeError(f"{case} seed {seed} is not an eight-target interception success")
    if int(result["seed"][0]) != seed:
        raise RuntimeError(f"seed mismatch in {npz_path}")
    expected_failed = np.asarray(
        [int(item) for item in row["failed_defender_ids"].split(";")], dtype=int
    )
    if not np.array_equal(result["failed_ids"], expected_failed):
        raise RuntimeError(f"failed-defender mismatch for {case} seed {seed}")
    completion_steps = min(
        result["defender_positions"].shape[0],
        max(2, int(math.ceil(float(row["E_t_s"]) / DT)) + 1),
    )
    result["completion_steps"] = completion_steps
    return result


def common_limits(episodes: list[dict]) -> tuple[tuple[float, float], ...]:
    arrays = []
    for episode in episodes:
        end = episode["completion_steps"]
        arrays.extend(
            [
                episode["defender_positions"][:end].reshape(-1, 3),
                episode["attacker_positions"][:end].reshape(-1, 3),
            ]
        )
    points = np.concatenate(arrays, axis=0)
    points = np.concatenate([points, np.zeros((1, 3))], axis=0)
    limits = []
    for axis in range(3):
        low = float(np.nanmin(points[:, axis]))
        high = float(np.nanmax(points[:, axis]))
        span = max(high - low, 1.0)
        margin = 0.055 * span
        limits.append((low - margin, high + margin))
    return tuple(limits)


def trajectory_segments(episode: dict):
    defenders = episode["defender_positions"]
    attackers = episode["attacker_positions"]
    assignment = episode["assignment"].astype(int)
    failed = set(int(value) for value in episode["failed_ids"])
    hit_time = episode["hit_time"].astype(float)
    completion = int(episode["completion_steps"])
    defender_segments = []
    for defender_id in range(defenders.shape[1]):
        if defender_id in failed:
            continue
        if np.isfinite(hit_time[defender_id]):
            stop = min(completion, max(2, int(round(hit_time[defender_id] / DT))))
        else:
            stop = completion
        defender_segments.append(
            (defender_id, int(assignment[defender_id]), defenders[:stop, defender_id])
        )

    attacker_segments = []
    intercept_points = []
    for target_id in range(attackers.shape[1]):
        group = np.where(assignment == target_id)[0]
        times = hit_time[group]
        times = times[np.isfinite(times)]
        first_hit = float(np.min(times)) if times.size else (completion - 1) * DT
        stop = min(completion, max(2, int(round(first_hit / DT))))
        segment = attackers[:stop, target_id]
        attacker_segments.append((target_id, segment))
        intercept_points.append((target_id, segment[-1]))
    return defender_segments, attacker_segments, intercept_points


def legend_handles(failed_ids):
    failed_label = "Failed D-UAVs (" + ", ".join(f"D{int(i)}" for i in failed_ids) + ")"
    return [
        Line2D([0], [0], color="#333333", lw=1.0, marker="o", mfc="white", ms=4, label="Operational D-UAV"),
        Line2D([0], [0], color="#333333", lw=1.4, ls="--", marker="s", ms=4, label="A-UAV"),
        Line2D([0], [0], color="#555555", lw=0, marker="X", ms=6, label=failed_label),
        Line2D([0], [0], color="#333333", lw=0, marker="D", mfc="white", ms=4, label="Interception point"),
        Line2D([0], [0], color="black", lw=0, marker="*", ms=8, label="Protected asset"),
    ]


def plot_2d(episode: dict, limits, output: Path) -> None:
    defenders, attackers, points = trajectory_segments(episode)
    fig, ax = plt.subplots(figsize=(6.55, 5.15))
    for defender_id, target_id, segment in defenders:
        color = COLORS[target_id % len(COLORS)]
        ax.plot(segment[:, 0], segment[:, 1], color=color, lw=0.9, alpha=0.82)
        ax.plot(segment[0, 0], segment[0, 1], marker="o", ms=3.1, mfc="white", mec=color, mew=0.7)
    for target_id, segment in attackers:
        color = COLORS[target_id % len(COLORS)]
        ax.plot(segment[:, 0], segment[:, 1], color=color, lw=1.45, ls="--", alpha=0.95)
        ax.plot(segment[0, 0], segment[0, 1], marker="s", ms=3.7, color=color)
    for target_id, point in points:
        ax.plot(point[0], point[1], marker="D", ms=3.8, mfc="white", mec=COLORS[target_id], mew=0.9)
    failed = episode["failed_ids"].astype(int)
    initial = episode["defender_positions"][0]
    for defender_id in failed:
        ax.scatter(initial[defender_id, 0], initial[defender_id, 1], marker="X", s=52, c="#555555", edgecolors="white", linewidths=0.5, zorder=8)
    ax.scatter(0.0, 0.0, marker="*", s=78, color="black", zorder=9)
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls="--", lw=0.45, color="#C8C8C8", alpha=0.62)
    ax.legend(handles=legend_handles(failed), loc="best", frameon=True, framealpha=0.92, handlelength=2.2)
    fig.tight_layout(pad=0.55)
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def plot_3d(episode: dict, limits, output: Path) -> None:
    defenders, attackers, points = trajectory_segments(episode)
    fig = plt.figure(figsize=(6.65, 5.60))
    ax = fig.add_subplot(111, projection="3d")
    for defender_id, target_id, segment in defenders:
        color = COLORS[target_id % len(COLORS)]
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, lw=0.9, alpha=0.82)
        ax.plot([segment[0, 0]], [segment[0, 1]], [segment[0, 2]], marker="o", ms=3.0, mfc="white", mec=color, mew=0.7)
    for target_id, segment in attackers:
        color = COLORS[target_id % len(COLORS)]
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, lw=1.45, ls="--", alpha=0.95)
        ax.plot([segment[0, 0]], [segment[0, 1]], [segment[0, 2]], marker="s", ms=3.6, color=color)
    for target_id, point in points:
        ax.plot([point[0]], [point[1]], [point[2]], marker="D", ms=3.7, mfc="white", mec=COLORS[target_id], mew=0.9)
    failed = episode["failed_ids"].astype(int)
    initial = episode["defender_positions"][0]
    for defender_id in failed:
        ax.scatter(initial[defender_id, 0], initial[defender_id, 1], initial[defender_id, 2], marker="X", s=48, c="#555555", edgecolors="white", linewidths=0.5, depthshade=False)
    ax.scatter([0.0], [0.0], [0.0], marker="*", s=75, color="black", depthshade=False)
    ax.set_xlabel(r"$x$ (m)", labelpad=3)
    ax.set_ylabel(r"$y$ (m)", labelpad=3)
    ax.set_zlabel(r"$z$ (m)", labelpad=3)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.20, 1.00, 0.55))
    ax.grid(True, ls="--", lw=0.35, color="#C8C8C8", alpha=0.55)
    ax.legend(handles=legend_handles(failed), loc="upper left", frameon=True, framealpha=0.90, handlelength=2.1)
    fig.subplots_adjust(left=0.01, right=0.98, bottom=0.02, top=0.99)
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def validate(case: str, episodes: list[dict], original_csv: Path) -> list[dict]:
    originals = read_rows_by_seed(original_csv)
    rows = []
    for run_index, episode in enumerate(episodes, start=1):
        seed = episode["seed_value"]
        original = originals[seed]
        replay = episode["row"]
        if original["failed_defender_ids"] != replay["failed_defender_ids"]:
            raise RuntimeError(f"failed IDs changed for {case} seed {seed}")
        row = {
            "case": case,
            "run": run_index,
            "seed": seed,
            "failed_defender_ids": replay["failed_defender_ids"],
            "failed_target_indices": replay["failed_target_indices"],
            "interception_success": int(replay["interception_success"]),
            "all_active_defenders_hit": int(replay["all_active_defenders_hit"]),
            "cooperative_success": int(replay["cooperative_success"]),
            "targets_covered": int(replay["targets_covered"]),
        }
        for metric in METRICS:
            reference = float(original[metric])
            reproduced = float(replay[metric])
            delta = reproduced - reference
            row[f"original_{metric}"] = reference
            row[f"replay_{metric}"] = reproduced
            row[f"delta_{metric}"] = delta
            # The original trials ran on a remote GPU, whereas the retained
            # trajectories are replayed on CPU.  Permit at most 1 cm in the
            # distance aggregate and 1e-4 in the remaining scalar metrics.
            tolerance = 1.0e-2 if metric == "E_miss_m" else 1.0e-4
            if abs(delta) > tolerance:
                raise RuntimeError(
                    f"{case} seed {seed} {metric} replay delta {delta} exceeds {tolerance}"
                )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case1-root", type=Path, required=True)
    parser.add_argument("--case2-root", type=Path, required=True)
    parser.add_argument("--case1-original", type=Path, required=True)
    parser.add_argument("--case2-original", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--case1-seeds", type=int, nargs=5, required=True)
    parser.add_argument("--case2-seeds", type=int, nargs=5, required=True)
    parser.add_argument("--case1-model", type=Path, required=True)
    parser.add_argument("--case2-model", type=Path, required=True)
    parser.add_argument("--preset", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    png_dir = args.outdir / "png"
    pdf_dir = args.outdir / "pdf"
    data_dir = args.outdir / "data"
    for path in (png_dir, pdf_dir, data_dir):
        path.mkdir(parents=True, exist_ok=True)

    case_data = {
        "case1": [load_episode("case1", seed, args.case1_root) for seed in args.case1_seeds],
        "case2": [load_episode("case2", seed, args.case2_root) for seed in args.case2_seeds],
    }
    validation = validate("case1", case_data["case1"], args.case1_original)
    validation += validate("case2", case_data["case2"], args.case2_original)
    write_csv(args.outdir / "trajectory_selection_and_validation.csv", validation)

    for case, episodes in case_data.items():
        limits = common_limits(episodes)
        for run_index, episode in enumerate(episodes, start=1):
            seed = episode["seed_value"]
            stem = f"{case}_success_{run_index:02d}_seed{seed}"
            plot_2d(episode, limits, png_dir / f"{stem}_2d")
            plot_3d(episode, limits, png_dir / f"{stem}_3d")
            shutil.move(str((png_dir / f"{stem}_2d.pdf")), pdf_dir / f"{stem}_2d.pdf")
            shutil.move(str((png_dir / f"{stem}_3d.pdf")), pdf_dir / f"{stem}_3d.pdf")
            destination = data_dir / case / f"seed_{seed}"
            shutil.copytree(episode["source_dir"], destination, dirs_exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    shutil.copy2(Path(__file__), args.outdir / Path(__file__).name)
    replay_script = script_dir / "replay_selected_partial_failure_trajectories.py"
    shutil.copy2(replay_script, args.outdir / replay_script.name)
    manifest = {
        "data_status": "deterministic frozen-policy inference replays; not synthetic trajectories",
        "training_performed": False,
        "optimizer_steps": 0,
        "backpropagation_performed": False,
        "case1_seeds": args.case1_seeds,
        "case2_seeds": args.case2_seeds,
        "success_definition": "all eight attacking aircraft intercepted (targets_covered = 8)",
        "case1_selection": "successful asset-safe episodes; selected runs are also strict cooperative successes",
        "case2_selection": "successful asset-safe episodes with E_t <= 45 s; strict cooperative flag retained in validation CSV",
        "trajectory_crop": "plotted through the recorded episode-level E_t",
        "figure_count": {"unique_figures": 20, "png": 20, "pdf": 20},
        "model_sha256": {
            "case1_actor": sha256(args.case1_model / "actor.pt"),
            "case1_critic": sha256(args.case1_model / "critic.pt"),
            "case2_actor": sha256(args.case2_model / "actor.pt"),
            "case2_critic": sha256(args.case2_model / "critic.pt"),
        },
        "preset_sha256": sha256(args.preset),
    }
    (args.outdir / "trajectory_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.outdir / "README.md").write_text(
        "# Partial-interceptor-failure trajectory candidates\n\n"
        "本目录包含两个工况各5个真实固定策略推理回放。每个回放提供三维轨迹和水平面二维轨迹，"
        "共20个不同图件，并同时提供PNG（600 dpi）与PDF。所有回合均满足8个进攻飞行器被拦截。"
        "Case 1所选回合同时满足严格协同成功；Case 2来自资产安全且 $E_t\\le45$ s 的成功子集，"
        "其中严格协同标志见 `trajectory_selection_and_validation.csv`，不得将普通拦截成功误写为"
        "严格协同成功。灰色叉号标记两架失效拦截器，颜色表示其分配目标组。\n\n"
        "轨迹由同哈希策略权重、同哈希场景预设及原蒙特卡洛种子确定性重放得到；未训练模型，"
        "未修改轨迹坐标。`data/`保存逐回合NPZ和指标文件。\n",
        encoding="utf-8",
    )
    products = sorted(
        path for path in args.outdir.rglob("*") if path.is_file() and path.name != "checksums.sha256"
    )
    (args.outdir / "checksums.sha256").write_text(
        "\n".join(f"{sha256(path)}  {path.relative_to(args.outdir)}" for path in products) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
