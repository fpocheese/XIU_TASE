#!/usr/bin/env python3
import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np


CASE_FOLDER = {"case1": "mappo_success_nopn", "case2": "mappo_success_sin"}
CASE_PREFIX = {"case1": "nopn", "case2": "sin"}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("ieee_plot_v9_tase", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load v9 plotting script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_eval_summary(eval_root: Path):
    path = eval_root / "eval_summary.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["case"]: row for row in csv.DictReader(f)}


def flight_xy(pos_xy: np.ndarray, dt: float):
    if pos_xy.shape[0] < 2:
        return np.zeros((pos_xy.shape[0], 2), dtype=np.float32)
    vel = np.gradient(pos_xy, dt, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    heading = np.arctan2(vel[:, 1], vel[:, 0])
    return np.stack([speed, heading], axis=1).astype(np.float32)


def final_target_mapping(deff: np.ndarray, att: np.ndarray):
    step = min(deff.shape[0], att.shape[0]) - 1
    mapping = []
    for i in range(deff.shape[1]):
        dpos = deff[step, i]
        dist = np.linalg.norm(att[step] - dpos[None, :], axis=1)
        mapping.append(int(np.argmin(dist)))
    return mapping


def export_case_v9(eval_root: Path, v9_base: Path, case_name: str, summary_rows: dict, dt: float):
    npz_path = eval_root / case_name / f"{case_name}_selected_episode.npz"
    if not npz_path.exists():
        print(f"[SKIP] missing {npz_path}")
        return None

    data = np.load(npz_path)
    deff = np.asarray(data["rep_def"], dtype=np.float32)
    att = np.asarray(data["rep_att"], dtype=np.float32)
    ctrl = np.asarray(data["rep_ctrl"], dtype=np.float32)
    tgo = np.asarray(data["rep_tgo"], dtype=np.float32)
    steps = min(deff.shape[0], att.shape[0], ctrl.shape[0], tgo.shape[0])
    deff, att, ctrl, tgo = deff[:steps], att[:steps], ctrl[:steps], tgo[:steps]

    mapping = final_target_mapping(deff, att)
    folder = v9_base / CASE_FOLDER[case_name]
    folder.mkdir(parents=True, exist_ok=True)

    agentspos = np.zeros((steps, 56), dtype=np.float32)
    agentsall = np.zeros((steps, 40), dtype=np.float32)
    agentsvel = np.zeros((steps, 40), dtype=np.float32)
    agentstimetgo = np.zeros((steps, 40), dtype=np.float32)

    for i in range(min(20, deff.shape[1])):
        agentspos[:, 2 * i] = deff[:, i, 0]
        agentspos[:, 2 * i + 1] = deff[:, i, 1]
        agentsall[:, 2 * i] = ctrl[:, i, 0]
        agentsall[:, 2 * i + 1] = ctrl[:, i, 1]
        fv = flight_xy(deff[:, i, :2], dt)
        agentsvel[:, 2 * i] = fv[:, 0]
        agentsvel[:, 2 * i + 1] = fv[:, 1]
        target_idx = mapping[i]
        dist = np.linalg.norm(att[:, target_idx, :] - deff[:, i, :], axis=1)
        agentstimetgo[:, 2 * i] = tgo[:, i] if i < tgo.shape[1] else dist / np.maximum(fv[:, 0], 1.0)
        agentstimetgo[:, 2 * i + 1] = dist

    for j in range(min(8, att.shape[1])):
        agentspos[:, 40 + 2 * j] = att[:, j, 0]
        agentspos[:, 40 + 2 * j + 1] = att[:, j, 1]

    np.savetxt(folder / "agentspos.txt", agentspos, fmt="%.9f")
    np.savetxt(folder / "agentsall.txt", agentsall, fmt="%.9f")
    np.savetxt(folder / "agentsvel.txt", agentsvel, fmt="%.9f")
    np.savetxt(folder / "agentstimetgo.txt", agentstimetgo, fmt="%.9f")

    row = summary_rows.get(case_name, {})
    all_sync = float(row.get("all_sync_rate", 0.0) or 0.0)
    spread = float(row.get("mean_sync_spread", 0.0) or 0.0)
    miss_rate = 1.0 - float(row.get("target_success_rate", 0.0) or 0.0)
    eval_arr = np.asarray([[all_sync, spread, miss_rate]], dtype=np.float32)
    np.savetxt(folder / f"{CASE_FOLDER[case_name]}_eval.txt", eval_arr, fmt="%.9f")
    if case_name == "case2":
        eval_dir = folder / "sinmappo_eval"
        eval_dir.mkdir(exist_ok=True)
        np.savetxt(eval_dir / "agentseval.txt", eval_arr, fmt="%.9f")
    return {"rep_def": deff, "rep_att": att, "mapping": mapping}


def plot_v9_3d(v9, episode, prefix: str, out: Path):
    plt = v9.plt
    colors = v9.ACADEMIC_COLORS_8
    width = v9.SINGLE_COL_WIDTH
    deff = episode["rep_def"]
    att = episode["rep_att"]
    mapping = episode["mapping"]

    fig = plt.figure(figsize=(width, width * 0.95))
    ax = fig.add_subplot(111, projection="3d")
    for i in range(min(20, deff.shape[1])):
        color = colors[mapping[i] % len(colors)]
        ax.plot(deff[:, i, 0], deff[:, i, 1], deff[:, i, 2], color=color, lw=0.9, alpha=0.85)
        ax.plot([deff[0, i, 0]], [deff[0, i, 1]], [deff[0, i, 2]], "o", color=color, ms=2.6, mfc="white", mew=0.5)
    for j in range(min(8, att.shape[1])):
        color = colors[j % len(colors)]
        ax.plot(att[:, j, 0], att[:, j, 1], att[:, j, 2], "--", color=color, lw=1.0, alpha=0.9)
        ax.plot([att[0, j, 0]], [att[0, j, 1]], [att[0, j, 2]], "s", color=color, ms=3.0)
    ax.set_xlabel("$x$ (m)", labelpad=2)
    ax.set_ylabel("$y$ (m)", labelpad=2)
    ax.set_zlabel("$z$ (m)", labelpad=2)
    ax.tick_params(labelsize=7, pad=0)
    ax.grid(True, ls="--", lw=0.3, color="#CCCCCC", alpha=0.5)
    ax.view_init(elev=23, azim=-58)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    fig.savefig(out / f"mappo_{prefix}_trajectory_3d.png")
    plt.close(fig)


def plot_v9_outputs(eval_root: Path, v9_base: Path, out: Path, v9_script: Path, dt: float):
    v9 = load_module(v9_script)
    out.mkdir(parents=True, exist_ok=True)
    summary_rows = load_eval_summary(eval_root)
    episodes = {}
    for case_name in ("case1", "case2"):
        ep = export_case_v9(eval_root, v9_base, case_name, summary_rows, dt)
        if ep is not None:
            episodes[case_name] = ep

    loader = v9.DataLoader(str(v9_base))
    for case_name, folder in CASE_FOLDER.items():
        if not (v9_base / folder / "agentspos.txt").exists():
            continue
        prefix = CASE_PREFIX[case_name]
        res = v9.process_single_dataset(loader, folder)
        if res is None:
            continue
        v9.plot_trajectory_single(loader, res, prefix, "MAPPO", out)
        v9.plot_ny_single(loader, res, prefix, "MAPPO", out)
        v9.plot_nx_single(loader, res, prefix, "MAPPO", out)
        v9.plot_velocity_single(loader, res, prefix, "MAPPO", out)
        v9.plot_heading_single(loader, res, prefix, "MAPPO", out)
        v9.plot_tgo_single(loader, res, prefix, "MAPPO", out)
        v9.plot_tgo_error_single(loader, res, prefix, "MAPPO", out)
        v9.plot_distance_single(loader, res, prefix, "MAPPO", out)
        v9.plot_time_sync_single(loader, res, prefix, "MAPPO", out)
        plot_v9_3d(v9, episodes[case_name], prefix, out)
    v9.plot_standalone_duav_legend(out)
    print(f"[DONE] v9 figures={out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--v9_base", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--v9_script", default=str(Path(__file__).with_name("ieee_plot_v9_tase.py")))
    parser.add_argument("--dt", type=float, default=0.05)
    args = parser.parse_args()

    eval_root = Path(args.eval_root).resolve()
    v9_base = Path(args.v9_base).resolve() if args.v9_base else eval_root / "v9_export"
    out = Path(args.out).resolve() if args.out else eval_root / "figures_v9"
    plot_v9_outputs(eval_root, v9_base, out, Path(args.v9_script).resolve(), args.dt)


if __name__ == "__main__":
    main()
