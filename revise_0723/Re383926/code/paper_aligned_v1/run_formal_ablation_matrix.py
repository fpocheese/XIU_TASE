#!/usr/bin/env python3
"""Run the fair ART-MAPPO component-ablation training matrix.

Every job receives the same budget and hyperparameters.  Only ``--variant``
changes.  The manifest is written before execution, and interrupted jobs
resume their own scheduler/checkpoint without a learning-rate restart.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd


VARIANTS = (
    "full",
    "no_trust",
    "no_gru",
    "no_attention_residual",
)


def parse_seeds(text: str):
    seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty unique comma-separated list")
    return seeds


def append_status(path: Path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def is_complete(metrics: Path, required_steps: int):
    if not metrics.exists():
        return False
    try:
        data = pd.read_csv(metrics)
    except Exception:
        return False
    return bool(
        len(data)
        and int(data["environment_steps"].max()) >= required_steps
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument("--outroot", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=614_400)
    parser.add_argument(
        "--cases", default="case1,case2", help="Comma-separated case list."
    )
    parser.add_argument("--seeds", default="12001,12002,12003,12004,12005")
    parser.add_argument("--max_parallel", type=int, default=2)
    parser.add_argument("--trust_initial", type=float, default=0.01)
    parser.add_argument("--trust_alpha", type=float, default=0.01)
    parser.add_argument("--trust_omega_pn", type=float, default=0.04)
    parser.add_argument("--trust_omega_probe", type=float, default=0.95)
    parser.add_argument("--trust_omega_random", type=float, default=0.01)
    parser.add_argument("--ppo_epoch", type=int, default=5)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    cases = [item.strip() for item in args.cases.split(",") if item.strip()]
    if any(case not in {"case1", "case2"} for case in cases):
        parser.error("cases must contain only case1/case2")
    seeds = parse_seeds(args.seeds)
    if args.steps < 4096 or args.max_parallel < 1:
        parser.error("invalid steps or max_parallel")
    guide_weights = [
        args.trust_omega_pn,
        args.trust_omega_probe,
        args.trust_omega_random,
    ]
    if (
        any(weight < 0.0 for weight in guide_weights)
        or abs(sum(guide_weights) - 1.0) > 1e-9
        or args.ppo_epoch < 1
    ):
        parser.error("invalid trust mixture or PPO epoch count")

    project_root = args.project_root.resolve()
    outroot = args.outroot.resolve()
    train_script = (
        project_root / "onpolicy/scripts/train_art_mappo_ablation_3d.py"
    )
    if not train_script.is_file():
        raise FileNotFoundError(train_script)
    outroot.mkdir(parents=True, exist_ok=True)
    logroot = outroot / "logs"
    logroot.mkdir(parents=True, exist_ok=True)
    jobs = []
    for case in cases:
        for seed in seeds:
            for variant in VARIANTS:
                runroot = outroot / "training" / variant / case / f"seed{seed}"
                metrics = runroot / "training_metrics.csv"
                checkpoint = runroot / "models" / "checkpoint_latest.pt"
                command = [
                    sys.executable,
                    str(train_script),
                    "--variant",
                    variant,
                    "--case_3d",
                    case,
                    "--seed",
                    str(seed),
                    "--save_dir",
                    str(outroot / "training"),
                    "--compare_steps",
                    str(args.steps),
                    "--episode_length",
                    "1024",
                    "--n_rollout_threads",
                    "4",
                    "--physical_episode_horizon_steps",
                    "1500",
                    "--trust_initial",
                    str(args.trust_initial),
                    "--trust_alpha",
                    str(args.trust_alpha),
                    "--trust_omega_pn",
                    str(args.trust_omega_pn),
                    "--trust_omega_probe",
                    str(args.trust_omega_probe),
                    "--trust_omega_random",
                    str(args.trust_omega_random),
                    "--ppo_epoch",
                    str(args.ppo_epoch),
                    "--save_interval",
                    "5",
                    "--checkpoint_interval",
                    "5",
                ]
                if checkpoint.exists() and not is_complete(metrics, args.steps):
                    command.append("--resume")
                jobs.append(
                    {
                        "case": case,
                        "seed": seed,
                        "variant": variant,
                        "metrics": str(metrics),
                        "log": str(
                            logroot / f"{variant}_{case}_seed{seed}.log"
                        ),
                        "command": command,
                    }
                )

    manifest = {
        "steps": args.steps,
        "cases": cases,
        "seeds": seeds,
        "variants": list(VARIANTS),
        "max_parallel": args.max_parallel,
        "trust_initial": args.trust_initial,
        "trust_alpha": args.trust_alpha,
        "trust_mixture_weights": {
            "pn": args.trust_omega_pn,
            "probe": args.trust_omega_probe,
            "uniform": args.trust_omega_random,
        },
        "ppo_epoch": args.ppo_epoch,
        "rollout_buffer_transitions": 4096,
        "physical_episode_horizon_steps": 1500,
        "engagements_continue_across_rollout_updates": True,
        "fairness_constraint": (
            "identical command except variant; same seed is paired across variants"
        ),
        "jobs": jobs,
    }
    (outroot / "formal_ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    status_path = outroot / "formal_ablation_status.csv"
    status_lock = __import__("threading").Lock()

    def run_job(job):
        metrics = Path(job["metrics"])
        if is_complete(metrics, args.steps):
            return {**job, "status": "already_complete", "returncode": 0}
        start = time.time()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        with Path(job["log"]).open("a", encoding="utf-8") as log:
            completed = subprocess.run(
                job["command"],
                cwd=project_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
        return {
            **job,
            "status": "complete" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "wall_time_s": time.time() - start,
        }

    failures = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            status_row = {
                "case": result["case"],
                "seed": result["seed"],
                "variant": result["variant"],
                "status": result["status"],
                "returncode": result["returncode"],
                "wall_time_s": result.get("wall_time_s", 0.0),
                "metrics": result["metrics"],
                "log": result["log"],
            }
            with status_lock:
                append_status(status_path, status_row)
            print(json.dumps(status_row), flush=True)
            if result["returncode"]:
                failures.append(status_row)
    if failures:
        raise RuntimeError(f"{len(failures)} formal jobs failed")


if __name__ == "__main__":
    main()
