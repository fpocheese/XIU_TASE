#!/usr/bin/env python3
import argparse
import csv
import importlib.util
from collections import Counter, defaultdict
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


def load_csv_rows(path: Path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def flight_xy(pos_xy: np.ndarray, dt: float):
    if pos_xy.shape[0] < 2:
        return np.zeros((pos_xy.shape[0], 2), dtype=np.float32)
    vel = np.gradient(pos_xy, dt, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    heading = np.arctan2(vel[:, 1], vel[:, 0])
    return np.stack([speed, heading], axis=1).astype(np.float32)


def flight_3d(pos_xyz: np.ndarray, dt: float):
    if pos_xyz.shape[0] < 2:
        return np.zeros((pos_xyz.shape[0], 3), dtype=np.float32)
    vel = np.gradient(pos_xyz, dt, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    horiz = np.linalg.norm(vel[:, :2], axis=1)
    pitch = np.arctan2(vel[:, 2], np.maximum(horiz, 1e-6))
    yaw = np.arctan2(vel[:, 1], vel[:, 0])
    return np.stack([speed, pitch, yaw], axis=1).astype(np.float32)


def target_index_from_id(target_id):
    tid = int(float(target_id))
    return tid - 20 if tid >= 20 else tid


def target_id_from_index(target_index):
    return int(target_index) + 20


def hit_event_mapping(hit_rows):
    votes = defaultdict(Counter)
    for row in hit_rows:
        try:
            d = int(float(row["defender_id"]))
            a = target_index_from_id(row["target_id"])
        except Exception:
            continue
        votes[d][a] += 1
    return {d: counts.most_common(1)[0][0] for d, counts in votes.items() if counts}


def final_target_mapping(deff: np.ndarray, att: np.ndarray, hit_rows=None):
    event_map = hit_event_mapping(hit_rows or [])
    step = min(deff.shape[0], att.shape[0]) - 1
    mapping = []
    for i in range(deff.shape[1]):
        if i in event_map:
            mapping.append(int(event_map[i]))
            continue
        dpos = deff[step, i]
        dist = np.linalg.norm(att[step] - dpos[None, :], axis=1)
        mapping.append(int(np.argmin(dist)))
    return mapping


def select_stats_episode(success_rows, hit_rows):
    if not success_rows:
        return None
    try:
        best = min(
            success_rows,
            key=lambda r: (
                float(r.get("max_sync_spread", "inf") or "inf"),
                float(r.get("mean_sync_spread", "inf") or "inf"),
            ),
        )
        return int(float(best["episode"]))
    except Exception:
        pass
    episodes = []
    for row in hit_rows:
        try:
            episodes.append(int(float(row["episode"])))
        except Exception:
            continue
    return min(episodes) if episodes else None


def build_target_stats(case_name: str, mapping, hit_rows, success_rows):
    selected_episode = select_stats_episode(success_rows, hit_rows)
    rows = []
    grouped = defaultdict(list)
    for row in hit_rows:
        try:
            if selected_episode is not None and int(float(row["episode"])) != selected_episode:
                continue
            target_idx = target_index_from_id(row["target_id"])
            item = {
                "case": case_name,
                "episode": int(float(row["episode"])),
                "defender_id": int(float(row["defender_id"])),
                "target_id": target_id_from_index(target_idx),
                "target_index": target_idx,
                "hit_time": float(row["time"]),
                "hit_distance": float(row.get("dist_to_target", "nan")),
                "hit_x": float(row.get("defender_pos_x", "nan")),
                "hit_y": float(row.get("defender_pos_y", "nan")),
                "hit_z": float(row.get("defender_pos_z", "nan")),
            }
        except Exception:
            continue
        grouped[target_idx].append(item)

    assigned = defaultdict(list)
    for defender_id, target_idx in enumerate(mapping):
        assigned[int(target_idx)].append(defender_id)

    for target_idx in sorted(assigned):
        hits = grouped.get(target_idx, [])
        times = [h["hit_time"] for h in hits]
        dists = [h["hit_distance"] for h in hits]
        rows.append({
            "case": case_name,
            "episode": selected_episode if selected_episode is not None else "",
            "target_id": target_id_from_index(target_idx),
            "assigned_defenders": " ".join(str(d) for d in assigned[target_idx]),
            "hit_defenders": " ".join(str(h["defender_id"]) for h in sorted(hits, key=lambda x: x["defender_id"])),
            "assigned_count": len(assigned[target_idx]),
            "hit_count": len(hits),
            "all_assigned_hit": len(hits) >= len(assigned[target_idx]),
            "min_hit_time": min(times) if times else np.nan,
            "max_hit_time": max(times) if times else np.nan,
            "hit_time_spread": (max(times) - min(times)) if len(times) >= 2 else 0.0 if times else np.nan,
            "mean_hit_distance": float(np.nanmean(dists)) if dists else np.nan,
            "max_hit_distance": float(np.nanmax(dists)) if dists else np.nan,
        })
    return rows, [item for vals in grouped.values() for item in vals]


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

    hit_rows = load_csv_rows(eval_root / case_name / f"{case_name}_hit_events.csv")
    success_rows = load_csv_rows(eval_root / case_name / f"{case_name}_success_episodes.csv")
    tail_rows = load_csv_rows(eval_root / case_name / f"{case_name}_tailchase_stats.csv")
    mapping = final_target_mapping(deff, att, hit_rows)
    target_stats, selected_hits = build_target_stats(case_name, mapping, hit_rows, success_rows)
    folder = v9_base / CASE_FOLDER[case_name]
    folder.mkdir(parents=True, exist_ok=True)

    agentspos = np.zeros((steps, 56), dtype=np.float32)
    agentspos_3d = np.zeros((steps, 84), dtype=np.float32)
    agentsall = np.zeros((steps, 40), dtype=np.float32)
    agentsvel = np.zeros((steps, 40), dtype=np.float32)
    agentsvel_3d = np.zeros((steps, 60), dtype=np.float32)
    agentstimetgo = np.zeros((steps, 40), dtype=np.float32)
    rel_dist_3d = np.zeros((steps, min(20, deff.shape[1])), dtype=np.float32)

    for i in range(min(20, deff.shape[1])):
        agentspos[:, 2 * i] = deff[:, i, 0]
        agentspos[:, 2 * i + 1] = deff[:, i, 1]
        agentspos_3d[:, 3 * i:3 * i + 3] = deff[:, i, :3]
        agentsall[:, 2 * i] = ctrl[:, i, 0]
        agentsall[:, 2 * i + 1] = ctrl[:, i, 1]
        fv = flight_xy(deff[:, i, :2], dt)
        fv3 = flight_3d(deff[:, i, :3], dt)
        agentsvel[:, 2 * i] = fv[:, 0]
        agentsvel[:, 2 * i + 1] = fv[:, 1]
        agentsvel_3d[:, 3 * i:3 * i + 3] = fv3
        target_idx = mapping[i]
        dist = np.linalg.norm(att[:, target_idx, :] - deff[:, i, :], axis=1)
        rel_dist_3d[:, i] = dist
        agentstimetgo[:, 2 * i] = tgo[:, i] if i < tgo.shape[1] else dist / np.maximum(fv[:, 0], 1.0)
        agentstimetgo[:, 2 * i + 1] = dist

    for j in range(min(8, att.shape[1])):
        agentspos[:, 40 + 2 * j] = att[:, j, 0]
        agentspos[:, 40 + 2 * j + 1] = att[:, j, 1]
        agentspos_3d[:, 60 + 3 * j:60 + 3 * j + 3] = att[:, j, :3]

    np.savetxt(folder / "agentspos.txt", agentspos, fmt="%.9f")
    np.savetxt(folder / "agentspos_3d.txt", agentspos_3d, fmt="%.9f")
    np.savetxt(folder / "agentsall.txt", agentsall, fmt="%.9f")
    np.savetxt(folder / "agentsvel.txt", agentsvel, fmt="%.9f")
    np.savetxt(folder / "agentsvel_3d.txt", agentsvel_3d, fmt="%.9f")
    np.savetxt(folder / "agentstimetgo.txt", agentstimetgo, fmt="%.9f")
    np.savetxt(folder / "relative_distance_3d.txt", rel_dist_3d, fmt="%.9f")
    np.savez(
        folder / "trajectory_3d.npz",
        rep_def=deff,
        rep_att=att,
        rep_ctrl=ctrl,
        rep_tgo=tgo,
        mapping=np.asarray(mapping, dtype=np.int32),
        relative_distance_3d=rel_dist_3d,
    )

    if tail_rows:
        with open(folder / "defender_tailchase_stats_3d.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(tail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(tail_rows)

    if target_stats:
        with open(folder / "target_sync_stats_3d.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(target_stats[0].keys()))
            writer.writeheader()
            writer.writerows(target_stats)

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
    return {
        "rep_def": deff,
        "rep_att": att,
        "rep_ctrl": ctrl,
        "rep_tgo": tgo,
        "mapping": mapping,
        "relative_distance_3d": rel_dist_3d,
        "target_stats": target_stats,
        "tailchase_stats": tail_rows,
        "selected_hits": selected_hits,
    }


def _save(fig, path):
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    fig.canvas.draw()
    v9plt = fig.axes[0].figure
    v9plt.clf()


def plot_trajectory_3d(v9, episode, prefix: str, out: Path):
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
        ax.plot([deff[-1, i, 0]], [deff[-1, i, 1]], [deff[-1, i, 2]], "*", color="red", ms=4.0, mec="darkred", mew=0.3)
    for j in range(min(8, att.shape[1])):
        color = colors[j % len(colors)]
        ax.plot(att[:, j, 0], att[:, j, 1], att[:, j, 2], "--", color=color, lw=1.0, alpha=0.9)
        ax.plot([att[0, j, 0]], [att[0, j, 1]], [att[0, j, 2]], "s", color=color, ms=3.0)
    for hit in episode.get("selected_hits", []):
        ax.scatter([hit["hit_x"]], [hit["hit_y"]], [hit["hit_z"]], marker="*", s=24, color="red", edgecolor="darkred", linewidth=0.25)
    ax.set_xlabel("$x$ (m)", labelpad=2)
    ax.set_ylabel("$y$ (m)", labelpad=2)
    ax.set_zlabel("$z$ (m)", labelpad=2)
    ax.tick_params(labelsize=7, pad=0)
    ax.grid(True, ls="--", lw=0.3, color="#CCCCCC", alpha=0.5)
    ax.view_init(elev=23, azim=-58)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    fig.savefig(out / f"mappo_{prefix}_trajectory_3d.png")
    fig.savefig(out / f"mappo_{prefix}_trajectory_3d.pdf")
    plt.close(fig)


def plot_altitude_time(v9, episode, prefix: str, out: Path, dt: float):
    plt = v9.plt
    colors = v9.ACADEMIC_COLORS_8
    deff = episode["rep_def"]
    att = episode["rep_att"]
    mapping = episode["mapping"]
    t = np.arange(deff.shape[0]) * dt
    fig, ax = plt.subplots(figsize=(v9.SINGLE_COL_WIDTH, v9.SINGLE_COL_WIDTH * 0.62))
    for i in range(min(20, deff.shape[1])):
        ax.plot(t, deff[:, i, 2], color=colors[mapping[i] % len(colors)], lw=0.65, alpha=0.65)
    for j in range(min(8, att.shape[1])):
        ax.plot(t, att[:len(t), j, 2], "--", color=colors[j % len(colors)], lw=0.9, alpha=0.85)
    ax.set_xlabel("$t$ (s)")
    ax.set_ylabel("$z$ (m)")
    ax.grid(True, ls="--", lw=0.3, color="#CCCCCC", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out / f"mappo_{prefix}_altitude_z.png")
    fig.savefig(out / f"mappo_{prefix}_altitude_z.pdf")
    plt.close(fig)


def plot_relative_distance_3d(v9, episode, prefix: str, out: Path, dt: float):
    plt = v9.plt
    colors = v9.COLORS_20
    dist = episode["relative_distance_3d"]
    t = np.arange(dist.shape[0]) * dt
    fig, ax = plt.subplots(figsize=(v9.SINGLE_COL_WIDTH, v9.SINGLE_COL_WIDTH * 0.62))
    for i in range(dist.shape[1]):
        ax.plot(t, dist[:, i], color=colors[i % len(colors)], lw=0.65, alpha=0.72)
    for row in episode.get("tailchase_stats", []):
        try:
            d = int(float(row["defender_id"]))
            tm = float(row["first_min_time"])
            rm = float(row["first_min_distance"])
            failed = str(row.get("tailchase_failure", "False")).lower() == "true"
        except Exception:
            continue
        if 0 <= d < dist.shape[1]:
            ax.plot(tm, rm, "x" if failed else "o", color="red" if failed else "black", ms=3.2, mew=0.8)
    ax.set_xlabel("$t$ (s)")
    ax.set_ylabel("3D range (m)")
    ax.grid(True, ls="--", lw=0.3, color="#CCCCCC", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out / f"mappo_{prefix}_distance_3d.png")
    fig.savefig(out / f"mappo_{prefix}_distance_3d.pdf")
    plt.close(fig)


def plot_closing_velocity_3d(v9, episode, prefix: str, out: Path, dt: float):
    plt = v9.plt
    colors = v9.COLORS_20
    dist = episode["relative_distance_3d"]
    if dist.size == 0:
        return
    closing = -np.gradient(dist, dt, axis=0)
    t = np.arange(dist.shape[0]) * dt
    fig, ax = plt.subplots(figsize=(v9.SINGLE_COL_WIDTH, v9.SINGLE_COL_WIDTH * 0.62))
    for i in range(closing.shape[1]):
        ax.plot(t, closing[:, i], color=colors[i % len(colors)], lw=0.65, alpha=0.72)
    ax.axhline(0.0, color="black", lw=0.6, ls="--", alpha=0.6)
    for row in episode.get("tailchase_stats", []):
        try:
            d = int(float(row["defender_id"]))
            tm = float(row["first_min_time"])
            failed = str(row.get("tailchase_failure", "False")).lower() == "true"
            idx = max(0, min(closing.shape[0] - 1, int(round(tm / dt))))
        except Exception:
            continue
        if 0 <= d < closing.shape[1]:
            ax.plot(tm, closing[idx, d], "x" if failed else "o", color="red" if failed else "black", ms=3.2, mew=0.8)
    ax.set_xlabel("$t$ (s)")
    ax.set_ylabel("closing speed (m/s)")
    ax.grid(True, ls="--", lw=0.3, color="#CCCCCC", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out / f"mappo_{prefix}_closing_velocity_3d.png")
    fig.savefig(out / f"mappo_{prefix}_closing_velocity_3d.pdf")
    plt.close(fig)


def plot_target_pair_3d(v9, episode, prefix: str, out: Path):
    plt = v9.plt
    colors = v9.ACADEMIC_COLORS_8
    deff = episode["rep_def"]
    att = episode["rep_att"]
    mapping = episode["mapping"]
    fig = plt.figure(figsize=(v9.DOUBLE_COL_WIDTH, v9.DOUBLE_COL_WIDTH * 0.52))
    axes = [fig.add_subplot(2, 4, idx + 1, projection="3d") for idx in range(min(8, att.shape[1]))]
    for target_idx, ax in enumerate(axes):
        color = colors[target_idx % len(colors)]
        ax.plot(att[:, target_idx, 0], att[:, target_idx, 1], att[:, target_idx, 2], "--", color=color, lw=0.9)
        for d, mapped in enumerate(mapping):
            if int(mapped) != target_idx:
                continue
            ax.plot(deff[:, d, 0], deff[:, d, 1], deff[:, d, 2], color=color, lw=0.7, alpha=0.78)
            ax.scatter([deff[-1, d, 0]], [deff[-1, d, 1]], [deff[-1, d, 2]], marker="*", s=12, color="red")
        ax.set_xlabel("$x$", labelpad=-5)
        ax.set_ylabel("$y$", labelpad=-5)
        ax.set_zlabel("$z$", labelpad=-5)
        ax.tick_params(labelsize=5, pad=-2)
        ax.view_init(elev=22, azim=-58)
    fig.tight_layout(pad=0.4)
    fig.savefig(out / f"mappo_{prefix}_target_defender_3d.png")
    fig.savefig(out / f"mappo_{prefix}_target_defender_3d.pdf")
    plt.close(fig)


def plot_sync_stats(v9, episode, prefix: str, out: Path):
    stats = episode.get("target_stats") or []
    if not stats:
        return
    plt = v9.plt
    labels = [f"A{target_index_from_id(row['target_id']) + 1}" for row in stats]
    spread = np.asarray([float(row["hit_time_spread"]) for row in stats], dtype=float)
    max_dist = np.asarray([float(row["max_hit_distance"]) for row in stats], dtype=float)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(v9.SINGLE_COL_WIDTH, v9.SINGLE_COL_WIDTH * 0.62))
    ax.bar(x, spread, color="#0072B2", alpha=0.82, label="$\\Delta t$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Target")
    ax.set_ylabel("Hit-time spread (s)")
    ax.grid(True, axis="y", ls="--", lw=0.3, color="#CCCCCC", alpha=0.5)
    ax2 = ax.twinx()
    ax2.plot(x, max_dist, "o-", color="#D55E00", lw=1.0, ms=3.0, label="max range")
    ax2.set_ylabel("Max hit range (m)")
    fig.tight_layout()
    fig.savefig(out / f"mappo_{prefix}_sync_error_3d.png")
    fig.savefig(out / f"mappo_{prefix}_sync_error_3d.pdf")
    plt.close(fig)


def plot_v9_3d(v9, episode, prefix: str, out: Path, dt: float):
    plot_trajectory_3d(v9, episode, prefix, out)
    plot_altitude_time(v9, episode, prefix, out, dt)
    plot_relative_distance_3d(v9, episode, prefix, out, dt)
    plot_closing_velocity_3d(v9, episode, prefix, out, dt)
    plot_target_pair_3d(v9, episode, prefix, out)
    plot_sync_stats(v9, episode, prefix, out)


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
        plot_v9_3d(v9, episodes[case_name], prefix, out, dt)
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
