#!/usr/bin/env python3
"""Run reviewer Sections 3.8, 3.9, and 1.6 with frozen policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case1_model", type=Path, required=True)
    parser.add_argument("--case2_model", type=Path, required=True)
    parser.add_argument("--outroot", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    outroot = args.outroot.resolve()
    outroot.mkdir(parents=True, exist_ok=True)
    evaluator = Path(__file__).with_name("run_reviewer_eval_parallel.py")

    jobs = []

    def add(name, case, model, seed, extra):
        outdir = outroot / name
        command = [
            sys.executable,
            str(evaluator),
            "--episodes",
            str(args.episodes),
            "--workers",
            str(args.workers),
            "--seed",
            str(seed),
            "--outdir",
            str(outdir),
            "--condition",
            name,
            "--case",
            case,
            "--variant",
            "full",
            "--model_dir",
            str(model.resolve()),
            "--max_steps",
            "1500",
            "--cpu_eval",
            *extra,
        ]
        jobs.append(
            {
                "name": name,
                "case": case,
                "model": str(model.resolve()),
                "seed_first": seed,
                "seed_last": seed + args.episodes - 1,
                "command": command,
            }
        )

    # Section 3.8: the nominal row shares all non-stressor settings.
    add(
        "failure_nominal",
        "case2",
        args.case2_model,
        93001,
        ["--assignment_mode", "fixed"],
    )
    add(
        "failure_sensing_stress",
        "case2",
        args.case2_model,
        93001,
        [
            "--assignment_mode",
            "fixed",
            "--sensor_delay_steps",
            "3",
            "--position_noise_std",
            "6.0",
            "--velocity_noise_std",
            "0.6",
        ],
    )
    add(
        "failure_command_lag",
        "case2",
        args.case2_model,
        93001,
        [
            "--assignment_mode",
            "fixed",
            "--command_lag_tau",
            "0.60",
        ],
    )
    add(
        "failure_compound",
        "case2",
        args.case2_model,
        93001,
        [
            "--assignment_mode",
            "fixed",
            "--sensor_delay_steps",
            "3",
            "--position_noise_std",
            "6.0",
            "--velocity_noise_std",
            "0.6",
            "--command_lag_tau",
            "0.60",
            "--attack_pattern",
            "multisine",
        ],
    )

    # Section 3.9: identical paired seeds, no retraining.
    for pattern in ("nominal", "chirp", "multisine", "jink"):
        add(
            f"generalization_{pattern}",
            "case2",
            args.case2_model,
            94001,
            ["--assignment_mode", "fixed", "--attack_pattern", pattern],
        )

    # Section 1.6: fixed and IDBO assignment share each perturbed episode.
    for case, model in (
        ("case1", args.case1_model),
        ("case2", args.case2_model),
    ):
        for mode in ("fixed", "idbo"):
            add(
                f"end_to_end_{case}_{mode}",
                case,
                model,
                95001,
                [
                    "--assignment_mode",
                    mode,
                    "--initial_perturbation_scale",
                    "1.0",
                    "--idbo_population",
                    "30",
                    "--idbo_iterations",
                    "80",
                ],
            )

    # Independent reviewer Case 3: the Case-2-trained policy is frozen and
    # transferred without fine-tuning.  Each episode first solves the actual
    # transformed engagement snapshot with the paper-faithful IDBO, then
    # executes decentralized ART-MAPPO cooperative interception against the
    # unseen heterogeneous switching maneuver.
    add(
        "end_to_end_case3_idbo_hybrid",
        "case3",
        args.case2_model,
        96001,
        [
            "--assignment_mode",
            "idbo",
            "--attack_pattern",
            "case3_hybrid",
            "--idbo_population",
            "30",
            "--idbo_iterations",
            "80",
        ],
    )

    manifest = {
        "episodes_per_condition": args.episodes,
        "workers": args.workers,
        "training_performed": False,
        "optimizer_steps": 0,
        "backpropagation_performed": False,
        "jobs": jobs,
    }
    (outroot / "supplementary_matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    failures = []
    for index, job in enumerate(jobs, 1):
        log_path = outroot / job["name"] / "matrix_runner.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                job["command"],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        print(
            f"[{index}/{len(jobs)}] {job['name']} "
            f"returncode={completed.returncode}",
            flush=True,
        )
        if completed.returncode:
            failures.append(
                {"name": job["name"], "returncode": completed.returncode}
            )
    if failures:
        raise RuntimeError(f"supplementary failures: {failures}")


if __name__ == "__main__":
    main()
