#!/usr/bin/env python3
"""Frozen-policy counterfactuals for attributing the two boundary failures."""

from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path("/home/a2rl/reviewer_failure_cases_20260730")
PYTHON = Path("/home/a2rl/miniconda3/envs/rlgpu/bin/python")
EVALUATOR = ROOT / "code/onpolicy/scripts/eval_3d_guidance.py"
PRESET = ROOT / "presets/paper_case_presets_original_assignment_verified.npz"
OUT = ROOT / "results/failure_cause_counterfactuals_v2"
SELECTED = {"case1": 76048, "case2": 77008}
ORIGINAL_MISSED = {"case1": 5, "case2": 1}
ASSIGNMENT = np.array(
    [20, 21, 22, 23, 24, 25, 26, 27, 20, 21,
     22, 23, 24, 25, 26, 27, 20, 21, 22, 23],
    dtype=int,
)

CASE_CONFIG = {
    "case1": {
        "model": ROOT / "models/case1",
        "gain": 2.0,
        "tau": 0.25,
        "lead": 1.60,
        "sync_gain": 0.14,
        "speed_gain": 0.016,
        "lag": 0.25,
    },
    "case2": {
        "model": ROOT / "models/case2",
        "gain": 2.6,
        "tau": 0.35,
        "lead": 1.70,
        "sync_gain": 1.40,
        "speed_gain": 0.008,
        "lag": 0.40,
    },
}

# One-factor removals plus a combined ideal-sensing/actuation condition.
CONDITIONS = {
    "observed_boundary": {"delay": 1, "pos_noise": 3.0, "vel_noise": 0.3,
                          "lag": "nominal"},
    "no_observation_delay": {"delay": 0, "pos_noise": 3.0, "vel_noise": 0.3,
                             "lag": "nominal"},
    "no_measurement_noise": {"delay": 1, "pos_noise": 0.0, "vel_noise": 0.0,
                             "lag": "nominal"},
    "ideal_observation": {"delay": 0, "pos_noise": 0.0, "vel_noise": 0.0,
                          "lag": "nominal"},
    "no_command_lag": {"delay": 1, "pos_noise": 3.0, "vel_noise": 0.3,
                       "lag": 0.0},
    "ideal_observation_no_lag": {
        "delay": 0, "pos_noise": 0.0, "vel_noise": 0.0, "lag": 0.0
    },
}


def build_command(case: str, seed: int, condition: str) -> list[str]:
    cfg = CASE_CONFIG[case]
    cond = CONDITIONS[condition]
    lag = cfg["lag"] if cond["lag"] == "nominal" else cond["lag"]
    run_dir = OUT / case / condition
    return [
        str(PYTHON), str(EVALUATOR), f"--{case}",
        "--seed", str(seed), "--eval_episodes", "1",
        "--eval_different_seed", "--max_steps", "1500",
        "--hit_radius_3d", "3.0", "--sync_tol", "0.5",
        "--model_dir", str(cfg["model"]), "--outdir", str(run_dir),
        "--paper_preset_path", str(PRESET), "--paper_attacker_replay", "1",
        "--paper_altitude", "120",
        "--defender_guidance_base_gain", str(cfg["gain"]),
        "--defender_guidance_tau", str(cfg["tau"]),
        "--defender_guidance_lead", str(cfg["lead"]),
        "--defender_residual_scale", "0.20",
        "--defender_load_limit", "1.0",
        "--defender_axial_min", "-0.1", "--defender_axial_max", "1.0",
        "--defender_sync_speed_gain", str(cfg["sync_gain"]),
        "--defender_sync_tgo_ref", "min",
        "--defender_speed_target", "40",
        "--defender_speed_gain", str(cfg["speed_gain"]),
        "--defender_speed_min", "12", "--defender_speed_max", "40",
        "--defender_sensor_delay_steps", str(cond["delay"]),
        "--defender_obs_pos_noise_std", str(cond["pos_noise"]),
        "--defender_obs_vel_noise_std", str(cond["vel_noise"]),
        "--defender_obs_filter_alpha", "1.0",
        "--defender_command_lag_tau", str(lag),
        "--attacker_load_limit", "1.75",
        "--attacker_yaw_scale", "1.55",
        "--attacker_pitch_scale", "1.55",
    ]


def run_one(case: str, seed: int, condition: str) -> dict:
    run_dir = OUT / case / condition
    run_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(case, seed, condition)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    log = run_dir / "run.log"
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command, cwd=ROOT / "code", env=env,
            stdout=handle, stderr=subprocess.STDOUT, check=False,
        )
    return {
        "case": case, "seed": seed, "condition": condition,
        "returncode": result.returncode, "command": command,
    }


def read_one(case: str, seed: int, condition: str) -> dict:
    folder = OUT / case / condition / case
    with (folder / f"{case}_episode_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        summary = list(csv.DictReader(handle))[0]
    with (folder / f"{case}_hit_events.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        events = list(csv.DictReader(handle))
    event_by_def = {int(row["defender_id"]): row for row in events}
    npz_path = folder / f"{case}_selected_episode.npz"
    z = np.load(npz_path)
    deff = np.asarray(z["rep_def"], dtype=float)
    att = np.asarray(z["rep_att"], dtype=float)
    original_id = ORIGINAL_MISSED[case]
    target_index = int(ASSIGNMENT[original_id] - 20)
    distance = np.linalg.norm(
        deff[:, original_id, :] - att[:, target_index, :], axis=1
    )
    step = int(np.argmin(distance))
    missed_ids = sorted(set(range(20)) - set(event_by_def))
    cond = CONDITIONS[condition]
    lag = CASE_CONFIG[case]["lag"] if cond["lag"] == "nominal" else cond["lag"]
    return {
        "case": case,
        "seed": seed,
        "condition": condition,
        "delay_steps": cond["delay"],
        "position_noise_std_m": cond["pos_noise"],
        "velocity_noise_std_mps": cond["vel_noise"],
        "command_lag_tau_s": lag,
        "defender_hit_count": int(summary["hit_count"]),
        "target_coverage_count": int(summary["target_hit_count"]),
        "complete_coordinated_group_count": int(summary["target_sync_count"]),
        "max_observed_hit_spread_s": float(summary["max_sync_spread"]),
        "originally_missed_defender_id": original_id,
        "originally_missed_defender_hit": original_id in event_by_def,
        "originally_missed_defender_hit_time_s": (
            float(event_by_def[original_id]["time"])
            if original_id in event_by_def else ""
        ),
        "originally_missed_defender_min_distance_m": float(distance[step]),
        "originally_missed_defender_closest_time_s": step * 0.05,
        "all_missed_defender_ids": ";".join(map(str, missed_ids)),
        "native_npz": str(npz_path),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [
        (case, seed, condition)
        for case, seed in SELECTED.items()
        for condition in CONDITIONS
    ]
    manifest = {
        "purpose": (
            "frozen-policy counterfactual attribution of observation delay, "
            "measurement noise, and command lag"
        ),
        "training_performed": False,
        "optimizer_steps": 0,
        "selected": SELECTED,
        "conditions": CONDITIONS,
        "commands": [build_command(*job) for job in jobs],
    }
    (OUT / "counterfactual_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(run_one, *job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(json.dumps(row), flush=True)
    failed = [row for row in results if row["returncode"]]
    if failed:
        raise RuntimeError(f"counterfactual execution failures: {failed}")
    rows = [read_one(*job) for job in jobs]
    with (OUT / "counterfactual_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
