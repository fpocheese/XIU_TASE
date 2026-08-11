#!/usr/bin/env python3
"""Reproduce the six preselected partial-defender failure episodes."""

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
OUT = ROOT / "results/selected_six_failure_cases"

# Fixed rule applied to the nominal 100-episode screen:
# earliest three seeds with 8/8 target coverage, exactly 19/20 defender hits,
# seven complete/coordinated groups, and observed hit-group spread <= 0.5 s.
SELECTED = {
    "case1": [76014, 76048, 76052],
    "case2": [77008, 77020, 77023],
}

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


def build_command(case: str, seed: int) -> list[str]:
    cfg = CASE_CONFIG[case]
    run_dir = OUT / case / f"seed_{seed}"
    return [
        str(PYTHON),
        str(EVALUATOR),
        f"--{case}",
        "--seed",
        str(seed),
        "--eval_episodes",
        "1",
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
        str(run_dir),
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


def run_one(case: str, seed: int) -> dict:
    run_dir = OUT / case / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(case, seed)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    log_path = run_dir / "replay.log"
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=ROOT / "code",
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    return {
        "case": case,
        "seed": seed,
        "returncode": result.returncode,
        "command": command,
        "log": str(log_path),
    }


def read_single_summary(case: str, seed: int) -> dict:
    path = OUT / case / f"seed_{seed}" / case / f"{case}_episode_summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected one summary row in {path}")
    row = rows[0]
    if not (
        row["all_hit"].lower() == "true"
        and int(row["target_hit_count"]) == 8
        and int(row["hit_count"]) == 19
        and int(row["target_sync_count"]) == 7
    ):
        raise RuntimeError(f"replay no longer satisfies selection: {row}")
    return {
        "case": case,
        "seed": seed,
        "defender_hit_count": int(row["hit_count"]),
        "target_coverage_count": int(row["target_hit_count"]),
        "complete_coordinated_groups": int(row["target_sync_count"]),
        "all_targets_intercepted": True,
        "strict_group_complete": False,
        "max_observed_hit_group_spread_s": float(row["max_sync_spread"]),
        "mean_observed_hit_group_spread_s": float(row["mean_sync_spread"]),
        "source_summary": str(path),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(case, seed) for case, seeds in SELECTED.items() for seed in seeds]
    manifest = {
        "selection_rule": (
            "earliest three nominal-screen seeds per case with 8/8 target "
            "coverage, exactly 19/20 defender hits, seven complete/coordinated "
            "groups, and observed hit-group spread <= 0.5 s"
        ),
        "selected": SELECTED,
        "training_performed": False,
        "optimizer_steps": 0,
        "commands": [build_command(*job) for job in jobs],
    }
    (OUT / "replay_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(run_one, *job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    failures = [row for row in results if row["returncode"]]
    if failures:
        raise RuntimeError(f"replay failures: {failures}")
    summaries = [read_single_summary(*job) for job in jobs]
    with (OUT / "selected_case_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (OUT / "replay_validation.json").write_text(
        json.dumps(
            {
                "selected_episode_count": len(summaries),
                "case1_count": sum(row["case"] == "case1" for row in summaries),
                "case2_count": sum(row["case"] == "case2" for row in summaries),
                "all_replays_match_selection": True,
                "training_performed": False,
                "optimizer_steps": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
