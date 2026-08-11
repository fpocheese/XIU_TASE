#!/usr/bin/env python3
"""Held-out checkpoint selection and exactly-100-episode ablation evaluation.

Each training seed is selected only on a common validation seed set.  Final
test episodes are then split across training seeds, with identical episode
seeds paired across all four variants.  No optimizer is constructed here.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


VARIANTS = (
    "full",
    "no_trust",
    "no_gru",
    "no_attention_residual",
)
CASES = ("case1", "case2")


def parse_ints(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("expected a nonempty list of unique integers")
    return values


def run_logged(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_root", type=Path, required=True)
    parser.add_argument("--outroot", type=Path, required=True)
    parser.add_argument(
        "--seeds", default="12001,12002,12003,12004,12005"
    )
    parser.add_argument("--validation_episodes", type=int, default=10)
    parser.add_argument("--validation_workers", type=int, default=5)
    parser.add_argument("--checkpoint_stride", type=int, default=5)
    parser.add_argument("--test_episodes", type=int, default=100)
    parser.add_argument("--test_workers", type=int, default=5)
    parser.add_argument("--max_parallel", type=int, default=2)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    seeds = parse_ints(args.seeds)
    if (
        args.validation_episodes < 1
        or args.validation_workers < 1
        or args.checkpoint_stride < 1
        or args.test_episodes < len(seeds)
        or args.test_workers < 1
        or args.max_parallel < 1
    ):
        parser.error("invalid episode, worker, stride, or parallel count")

    training_root = args.training_root.resolve()
    outroot = args.outroot.resolve()
    outroot.mkdir(parents=True, exist_ok=True)
    script_root = Path(__file__).resolve().parent
    selector = script_root / "select_checkpoint_frozen_validation.py"
    evaluator = script_root / "run_reviewer_eval_parallel.py"

    quotient, remainder = divmod(args.test_episodes, len(seeds))
    episodes_by_seed = [
        quotient + int(index < remainder) for index in range(len(seeds))
    ]
    offsets = []
    running = 0
    for count in episodes_by_seed:
        offsets.append(running)
        running += count
    assert running == args.test_episodes

    selection_jobs = []
    evaluation_jobs = []
    for case_index, case in enumerate(CASES):
        validation_seed = 97001 + 100 * case_index
        test_seed_base = 98001 + 1000 * case_index
        for seed_index, seed in enumerate(seeds):
            for variant in VARIANTS:
                model_dir = (
                    training_root
                    / variant
                    / case
                    / f"seed{seed}"
                    / "models"
                )
                selection_dir = (
                    outroot
                    / "selection"
                    / variant
                    / case
                    / f"seed{seed}"
                )
                selection_command = [
                    sys.executable,
                    str(selector),
                    "--case",
                    case,
                    "--variant",
                    variant,
                    "--model_dir",
                    str(model_dir),
                    "--outdir",
                    str(selection_dir),
                    "--episodes",
                    str(args.validation_episodes),
                    "--workers",
                    str(args.validation_workers),
                    "--seed",
                    str(validation_seed),
                    "--checkpoint_stride",
                    str(args.checkpoint_stride),
                    "--initial_perturbation_scale",
                    "1.0",
                ]
                selection_jobs.append(
                    {
                        "case": case,
                        "variant": variant,
                        "training_seed": seed,
                        "command": selection_command,
                        "outdir": str(selection_dir),
                        "log": str(selection_dir / "selection_runner.log"),
                    }
                )

                episode_count = episodes_by_seed[seed_index]
                test_seed = test_seed_base + offsets[seed_index]
                evaluation_dir = (
                    outroot
                    / "evaluation"
                    / variant
                    / case
                    / f"seed{seed}"
                )
                evaluation_command = [
                    sys.executable,
                    str(evaluator),
                    "--episodes",
                    str(episode_count),
                    "--workers",
                    str(min(args.test_workers, episode_count)),
                    "--seed",
                    str(test_seed),
                    "--outdir",
                    str(evaluation_dir),
                    "--condition",
                    f"formal_ablation_{variant}_{case}",
                    "--case",
                    case,
                    "--variant",
                    variant,
                    "--model_dir",
                    str(selection_dir / "selected_model"),
                    "--max_steps",
                    "1500",
                    "--assignment_mode",
                    "fixed",
                    "--initial_perturbation_scale",
                    "1.0",
                    "--cpu_eval",
                ]
                evaluation_jobs.append(
                    {
                        "case": case,
                        "variant": variant,
                        "training_seed": seed,
                        "episode_count": episode_count,
                        "test_seed_first": test_seed,
                        "test_seed_last": test_seed + episode_count - 1,
                        "command": evaluation_command,
                        "outdir": str(evaluation_dir),
                        "log": str(evaluation_dir / "evaluation_runner.log"),
                    }
                )

    manifest = {
        "training_root": str(training_root),
        "training_seeds": seeds,
        "variants": list(VARIANTS),
        "cases": list(CASES),
        "validation_episodes_per_training_seed": args.validation_episodes,
        "checkpoint_stride": args.checkpoint_stride,
        "final_test_episodes_per_variant_case": args.test_episodes,
        "final_episode_allocation_by_training_seed": dict(
            zip(map(str, seeds), episodes_by_seed)
        ),
        "selection_uses_test_seeds": False,
        "paired_test_seeds_across_variants": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "backpropagation_performed": False,
        "selection_jobs": selection_jobs,
        "evaluation_jobs": evaluation_jobs,
    }
    (outroot / "formal_evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    def execute(job, phase):
        expected = (
            Path(job["outdir"]) / "selected_checkpoint.json"
            if phase == "selection"
            else Path(job["outdir"]) / "summary.json"
        )
        if expected.is_file():
            return {**job, "phase": phase, "returncode": 0, "skipped": True}
        code = run_logged(job["command"], Path(job["log"]))
        return {
            **job,
            "phase": phase,
            "returncode": code,
            "skipped": False,
        }

    statuses = []
    for phase, jobs in (
        ("selection", selection_jobs),
        ("evaluation", evaluation_jobs),
    ):
        failures = []
        with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
            futures = [pool.submit(execute, job, phase) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                statuses.append(
                    {
                        key: result[key]
                        for key in (
                            "phase",
                            "case",
                            "variant",
                            "training_seed",
                            "returncode",
                            "skipped",
                            "outdir",
                            "log",
                        )
                    }
                )
                print(json.dumps(statuses[-1]), flush=True)
                if result["returncode"]:
                    failures.append(result)
        pd.DataFrame(statuses).to_csv(
            outroot / "formal_evaluation_status.csv", index=False
        )
        if failures:
            raise RuntimeError(f"{len(failures)} {phase} jobs failed")

    combined_root = outroot / "combined"
    combined_root.mkdir(parents=True, exist_ok=True)
    for table in ("episodes", "targets", "assignments"):
        frames = []
        for job in evaluation_jobs:
            path = Path(job["outdir"]) / f"{table}.csv"
            data = pd.read_csv(path)
            data["training_seed"] = int(job["training_seed"])
            frames.append(data)
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(combined_root / f"{table}.csv", index=False)

    episodes = pd.read_csv(combined_root / "episodes.csv")
    counts = (
        episodes.groupby(["variant", "case"])
        .size()
        .rename("episode_count")
        .reset_index()
    )
    expected_rows = len(VARIANTS) * len(CASES)
    if (
        len(counts) != expected_rows
        or not (counts["episode_count"] == args.test_episodes).all()
    ):
        raise RuntimeError(
            "formal evaluation did not produce exactly the declared count"
        )
    counts.to_csv(combined_root / "episode_count_audit.csv", index=False)
    print(counts.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
