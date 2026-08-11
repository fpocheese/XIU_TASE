#!/usr/bin/env python3
"""Render the five replayed Case-3 trials with the original IEEE V10 style."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


V10_SOURCE = Path(
    "/home/uav/00gao_xueshu/DT_PAPER/XIU_code/xiu_onpolicy_3d_fix/"
    "on-policy-main/onpolicy/scripts/ieee_plot_v10_tase.py"
)
sys.path.insert(0, str(V10_SOURCE.parent))

import ieee_plot_v10_tase as v10  # noqa: E402


v9 = v10.v9
eval_v9 = v10.eval_v9
SEEDS = (74001, 74002, 74003, 74005, 74009)


def _require_shape(name: str, array: np.ndarray, expected: tuple[int | None, ...]) -> None:
    if array.ndim != len(expected):
        raise ValueError(f"{name}: expected {len(expected)} dimensions, got {array.shape}")
    for actual, wanted in zip(array.shape, expected):
        if wanted is not None and actual != wanted:
            raise ValueError(f"{name}: expected shape {expected}, got {array.shape}")


def build_v10_dataset(npz_path: Path):
    with np.load(npz_path) as raw:
        def_pos = np.asarray(raw["def_pos"], dtype=np.float32)
        atk_pos = np.asarray(raw["atk_pos"], dtype=np.float32)
        def_vel = np.asarray(raw["def_vel"], dtype=np.float32)
        loads = np.asarray(raw["loads"], dtype=np.float32)
        distances = np.asarray(raw["distances"], dtype=np.float32)
        assignment = np.asarray(raw["assignment"], dtype=np.int64)
        hit_times = np.asarray(raw["hit_times"], dtype=float)
        dt = float(raw["dt"])
        seed = int(raw["seed"])

    _require_shape("def_pos", def_pos, (None, 20, 3))
    _require_shape("atk_pos", atk_pos, (def_pos.shape[0], 8, 3))
    _require_shape("def_vel", def_vel, (def_pos.shape[0], 20, 3))
    _require_shape("loads", loads, (def_pos.shape[0], 20, 3))
    _require_shape("distances", distances, (def_pos.shape[0], 20))
    _require_shape("assignment", assignment, (20,))
    _require_shape("hit_times", hit_times, (20,))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"invalid dt={dt} in {npz_path}")
    if not np.isfinite(hit_times).all():
        raise ValueError(f"non-finite hit times in {npz_path}")
    if np.any((assignment < 0) | (assignment >= 8)):
        raise ValueError(f"invalid target assignment in {npz_path}")

    steps = def_pos.shape[0]
    agentspos = np.zeros((steps, 56), dtype=np.float32)
    agentsall = np.zeros((steps, 40), dtype=np.float32)
    agentsnz = loads[:, :, 2].copy()
    agentsvel = np.zeros((steps, 40), dtype=np.float32)
    agentstimetgo = np.zeros((steps, 40), dtype=np.float32)

    # Match the original V10 export conventions exactly: horizontal speed/yaw
    # are differentiated from x-y positions, while t_go is 3-D range divided
    # by total interceptor speed (the same definition used by the environment).
    total_speed = np.linalg.norm(def_vel, axis=2)
    tgo = distances / np.maximum(total_speed, 1.0)
    for defender in range(20):
        agentspos[:, 2 * defender] = def_pos[:, defender, 0]
        agentspos[:, 2 * defender + 1] = def_pos[:, defender, 1]
        agentsall[:, 2 * defender] = loads[:, defender, 0]
        agentsall[:, 2 * defender + 1] = loads[:, defender, 1]
        horizontal_flight = eval_v9.flight_xy(def_pos[:, defender, :2], dt)
        agentsvel[:, 2 * defender] = horizontal_flight[:, 0]
        agentsvel[:, 2 * defender + 1] = horizontal_flight[:, 1]
        agentstimetgo[:, 2 * defender] = tgo[:, defender]
        agentstimetgo[:, 2 * defender + 1] = distances[:, defender]
    for target in range(8):
        agentspos[:, 40 + 2 * target] = atk_pos[:, target, 0]
        agentspos[:, 40 + 2 * target + 1] = atk_pos[:, target, 1]

    # V9/V10 stop each curve just before the first repeated post-hit state.
    # The replay archives preserve the exact hit time, so this is more robust
    # than rediscovering the terminal row from floating-point position equality.
    hit_indices = np.rint(hit_times / dt).astype(int)
    repeat_start_rows = np.clip(hit_indices + 1, 2, steps)
    plot_end_rows = np.maximum(repeat_start_rows - 2, 1)
    mapping = {i: int(assignment[i]) for i in range(20)}

    res = {
        "data": {
            "agentspos": agentspos,
            "agentsall": agentsall,
            "agentsnz": agentsnz,
            "agentsvel": agentsvel,
            "agentstimetgo": agentstimetgo,
        },
        "repeat_start_rows": repeat_start_rows,
        "plot_end_rows": plot_end_rows,
        "episode_end": steps - 1,
        "mapping": mapping,
    }
    episode = {
        "rep_def": def_pos,
        "rep_att": atk_pos,
        "mapping": assignment.tolist(),
    }

    assigned_distance = np.stack(
        [
            np.linalg.norm(atk_pos[:, assignment[i], :] - def_pos[:, i, :], axis=1)
            for i in range(20)
        ],
        axis=1,
    )
    diagnostics = {
        "seed": seed,
        "samples": steps,
        "dt_s": dt,
        "duration_s": (steps - 1) * dt,
        "max_distance_consistency_error_m": float(
            np.max(np.abs(assigned_distance - distances))
        ),
        "tgo_definition": "distance_to_assigned_target / max(total_speed, 1 m/s)",
        "tgo_min_s": float(np.min(tgo)),
        "tgo_max_s": float(np.max(tgo)),
        "all_values_finite": bool(
            all(
                np.isfinite(array).all()
                for array in (def_pos, atk_pos, def_vel, loads, distances, tgo)
            )
        ),
    }
    return res, episode, diagnostics, dt


def plot_trial(npz_path: Path, outdir: Path) -> dict:
    res, episode, diagnostics, dt = build_v10_dataset(npz_path)
    loader = v9.DataLoader(str(npz_path.parent))
    loader.dt = dt
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = "case3"
    method = "MAPPO"

    # This is the original V10 panel sequence and its original smoothing policy.
    v9.plot_trajectory_single(loader, res, prefix, method, outdir)
    v10._plot_overload(loader, res, prefix, method, outdir, "ny")
    v10._plot_overload(loader, res, prefix, method, outdir, "nz")
    v9.plot_nx_single(loader, res, prefix, method, outdir)
    v9.plot_velocity_single(loader, res, prefix, method, outdir)
    v9.plot_heading_single(loader, res, prefix, method, outdir)
    v10._plot_attitude_angles(loader, res, episode, prefix, method, outdir)
    v9.plot_tgo_single(loader, res, prefix, method, outdir)
    v9.plot_tgo_error_single(loader, res, prefix, method, outdir)
    v9.plot_distance_single(loader, res, prefix, method, outdir)
    v9.plot_time_sync_single(loader, res, prefix, method, outdir)
    eval_v9.plot_v9_3d(v9, episode, prefix, method, outdir)
    v9.plot_standalone_duav_legend(outdir)

    expected = {
        "mappo_case3_trajectory.png",
        "mappo_case3_trajectory_3d.png",
        "mappo_case3_distance.png",
        "mappo_case3_nx.png",
        "mappo_case3_ny.png",
        "mappo_case3_nz.png",
        "mappo_case3_velocity.png",
        "mappo_case3_heading.png",
        "mappo_case3_yaw.png",
        "mappo_case3_pitch.png",
        "mappo_case3_tgo.png",
        "mappo_case3_tgo_error.png",
        "mappo_case3_time_sync.png",
        "standalone_duav_legend.png",
    }
    actual = {path.name for path in outdir.glob("*.png")}
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError(f"missing V10 figures for seed {diagnostics['seed']}: {missing}")
    diagnostics["figure_count"] = len(expected)
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    if not V10_SOURCE.exists():
        raise FileNotFoundError(V10_SOURCE)

    diagnostics = []
    for seed in SEEDS:
        npz_path = (
            args.input_root
            / f"seed_{seed}"
            / f"seed_{seed}_trajectory_raw.npz"
        )
        diagnostics.append(
            plot_trial(npz_path, args.out_root / f"seed_{seed}")
        )

    manifest = {
        "v10_source": str(V10_SOURCE),
        "v10_source_unchanged": True,
        "method": "MAPPO",
        "seeds": list(SEEDS),
        "figures_per_seed": 14,
        "data_support": {
            "direct": [
                "horizontal/3-D trajectory",
                "n_x/n_y/n_z overload",
                "range-to-go",
                "target assignment",
                "terminal arrival spread",
            ],
            "deterministic_from_recorded_state": [
                "speed",
                "yaw/heading",
                "pitch",
                "time-to-go",
                "within-group time-to-go error",
            ],
        },
        "trials": diagnostics,
    }
    (args.out_root / "case3_v10_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[DONE] five Case-3 V10 trials -> {args.out_root}")


if __name__ == "__main__":
    main()
