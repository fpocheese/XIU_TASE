#!/usr/bin/env python3
"""Parallel screen for genuine partial-defender failure episodes.

The trained policies, target assignment, lethal radius, and attacker replay
remain fixed.  An episode qualifies only when all eight targets are covered
but 1--3 of the 20 assigned defenders do not reach the lethal radius.
"""

from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess


ROOT = Path("/home/a2rl/reviewer_failure_cases_20260730")
PYTHON = Path("/home/a2rl/miniconda3/envs/rlgpu/bin/python")
EVALUATOR = ROOT / "code/onpolicy/scripts/eval_3d_guidance.py"
PRESET = ROOT / "presets/paper_case_presets_original_assignment_verified.npz"
SCREEN = ROOT / "results/partial_failure_screen_nominal_n100"

CASE_CONFIG = {
    "case1": {
        "model": ROOT / "models/case1",
        "seed": 76001,
        "gain": 2.0,
        "tau": 0.25,
        "lead": 1.60,
        "sync_gain": 0.14,
        "speed_gain": 0.016,
        "lag": 0.25,
    },
    "case2": {
        "model": ROOT / "models/case2",
        "seed": 77001,
        "gain": 2.6,
        "tau": 0.35,
        "lead": 1.70,
        "sync_gain": 1.40,
        "speed_gain": 0.008,
        "lag": 0.40,
    },
}


def command_for(case: str, chunk: int, episodes: int = 20) -> list[str]:
    cfg = CASE_CONFIG[case]
    seed = cfg["seed"] + chunk * episodes
    outdir = SCREEN / case / f"chunk_{chunk:02d}"
    return [
        str(PYTHON),
        str(EVALUATOR),
        f"--{case}",
        "--seed",
        str(seed),
        "--eval_episodes",
        str(episodes),
        "--eval_different_seed",
        "--max_steps",
        "1500",
        "--hit_radius_3d",
        "3.0",
        "--sync_tol",
        "0.5",
        "--model_dir",
        str(cfg["model"]),
        "--outdir",
        str(outdir),
        "--paper_preset_path",
        str(PRESET),
        "--paper_attacker_replay",
        "1",
        "--paper_altitude",
        "120",
        "--defender_guidance_base_gain",
        str(cfg["gain"]),
        "--defender_guidance_tau",
        str(cfg["tau"]),
        "--defender_guidance_lead",
        str(cfg["lead"]),
        "--defender_residual_scale",
        "0.20",
        "--defender_load_limit",
        "1.0",
        "--defender_axial_min",
        "-0.1",
        "--defender_axial_max",
        "1.0",
        "--defender_sync_speed_gain",
        str(cfg["sync_gain"]),
        "--defender_sync_tgo_ref",
        "min",
        "--defender_speed_target",
        "40",
        "--defender_speed_gain",
        str(cfg["speed_gain"]),
        "--defender_speed_min",
        "12",
        "--defender_speed_max",
        "40",
        "--defender_sensor_delay_steps",
        "1",
        "--defender_obs_pos_noise_std",
        "3.0",
        "--defender_obs_vel_noise_std",
        "0.3",
        "--defender_obs_filter_alpha",
        "1.0",
        "--defender_command_lag_tau",
        str(cfg["lag"]),
        "--attacker_load_limit",
        "1.75",
        "--attacker_yaw_scale",
        "1.55",
        "--attacker_pitch_scale",
        "1.55",
    ]


def run_job(case: str, chunk: int) -> dict:
    command = command_for(case, chunk)
    outdir = SCREEN / case / f"chunk_{chunk:02d}"
    outdir.mkdir(parents=True, exist_ok=True)
    log = outdir / "run.log"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT / "code",
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    return {
        "case": case,
        "chunk": chunk,
        "returncode": completed.returncode,
        "command": command,
        "log": str(log),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    SCREEN.mkdir(parents=True, exist_ok=True)
    jobs = [(case, chunk) for case in CASE_CONFIG for chunk in range(5)]
    manifest = {
        "purpose": "screen genuine boundary failures without retraining",
        "episodes_per_case": 100,
        "qualification": {
            "target_hit_count": 8,
            "defender_hit_count_min": 17,
            "defender_hit_count_max": 19,
        },
        "training_performed": False,
        "optimizer_steps": 0,
        "commands": [command_for(*job) for job in jobs],
    }
    (SCREEN / "screen_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    results = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(run_job, *job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    failures = [result for result in results if result["returncode"]]
    if failures:
        raise RuntimeError(f"screen worker failures: {failures}")

    rows = []
    for case, chunk in jobs:
        local_seed = CASE_CONFIG[case]["seed"] + chunk * 20
        path = SCREEN / case / f"chunk_{chunk:02d}" / case
        path = path / f"{case}_episode_summary.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row["eval_seed"] = local_seed + int(row["episode"]) - 1
                row["chunk"] = chunk
                row["source_summary"] = str(path)
                rows.append(row)
    rows.sort(key=lambda row: (row["case"], int(row["eval_seed"])))
    candidates = [
        row
        for row in rows
        if row["all_hit"].lower() == "true"
        and int(row["target_hit_count"]) == 8
        and 17 <= int(row["hit_count"]) <= 19
    ]
    write_csv(SCREEN / "screen_all_episodes.csv", rows)
    write_csv(SCREEN / "partial_failure_candidates.csv", candidates)
    counts = {
        case: sum(row["case"] == case for row in candidates) for case in CASE_CONFIG
    }
    (SCREEN / "candidate_counts.json").write_text(
        json.dumps(counts, indent=2), encoding="utf-8"
    )
    print(json.dumps({"candidate_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
