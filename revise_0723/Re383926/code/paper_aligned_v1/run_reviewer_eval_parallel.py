#!/usr/bin/env python3
"""Parallel, seed-exact wrapper for frozen-policy reviewer evaluations.

The worker processes only execute ``run_reviewer_supplementary_experiments.py``.
Their CSV files are merged without changing any episode value, and the aggregate
confidence intervals are recomputed from the merged 100-episode sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


RATE_METRICS = (
    "target_coverage_success",
    "all_defenders_hit",
    "cooperative_success",
    "mission_success",
)
MEAN_METRICS = (
    "E_co_time_s",
    "E_n_g",
    "E_miss_m",
    "E_t_s",
    "mean_closest_approach_m",
    "worst_closest_approach_m",
    "mean_agent_return",
    "idbo_runtime_ms",
    "idbo_repaired_cost",
)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wilson_interval(count: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return math.nan, math.nan
    p = count / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = (
        z
        * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def mean_ci(values):
    finite = np.asarray(
        [float(value) for value in values if np.isfinite(float(value))],
        dtype=float,
    )
    if not len(finite):
        return math.nan, math.nan, math.nan, 0
    mean = float(np.mean(finite))
    if len(finite) == 1:
        return mean, mean, mean, 1
    half = 1.959963984540054 * float(
        np.std(finite, ddof=1) / np.sqrt(len(finite))
    )
    return mean, mean - half, mean + half, int(len(finite))


def split_counts(total: int, workers: int):
    base, extra = divmod(total, workers)
    return [base + (index < extra) for index in range(workers)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=91000)
    parser.add_argument("--outdir", required=True)
    args, forwarded = parser.parse_known_args()
    if args.episodes < 1 or args.workers < 1:
        parser.error("episodes and workers must be positive")
    workers = min(args.workers, args.episodes)
    outdir = Path(args.outdir).resolve()
    chunk_root = outdir / "_chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    evaluator = Path(__file__).with_name(
        "run_reviewer_supplementary_experiments.py"
    )

    processes = []
    offset = 0
    for index, count in enumerate(split_counts(args.episodes, workers)):
        chunk_dir = chunk_root / f"chunk_{index:02d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        log_handle = (chunk_dir / "run.log").open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(evaluator),
            *forwarded,
            "--episodes",
            str(count),
            "--episode_offset",
            str(offset),
            "--seed",
            str(args.seed + offset),
            "--outdir",
            str(chunk_dir),
        ]
        worker_env = os.environ.copy()
        worker_env.update(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=worker_env,
        )
        processes.append((process, log_handle, chunk_dir))
        offset += count

    failures = []
    for process, log_handle, chunk_dir in processes:
        code = process.wait()
        log_handle.close()
        if code:
            failures.append((str(chunk_dir), code))
    if failures:
        raise RuntimeError(f"frozen-policy worker failures: {failures}")

    merged = {}
    for filename in ("episodes.csv", "targets.csv", "assignments.csv"):
        rows = []
        for _, _, chunk_dir in processes:
            rows.extend(read_csv(chunk_dir / filename))
        rows.sort(
            key=lambda row: (
                int(row.get("episode", 0)),
                int(row.get("target_id", -1)),
                int(row.get("defender_id", -1)),
            )
        )
        write_csv(outdir / filename, rows)
        merged[filename] = rows

    episodes = merged["episodes.csv"]
    if len(episodes) != args.episodes:
        raise RuntimeError(
            f"expected {args.episodes} episode rows, got {len(episodes)}"
        )
    episode_ids = [int(row["episode"]) for row in episodes]
    seeds = [int(row["seed"]) for row in episodes]
    if episode_ids != list(range(1, args.episodes + 1)):
        raise RuntimeError("merged episode numbering is not contiguous")
    if len(set(seeds)) != args.episodes:
        raise RuntimeError("merged evaluation seeds are not unique")

    first_summary = json.loads(
        (processes[0][2] / "summary.json").read_text(encoding="utf-8")
    )
    summary = {
        key: value
        for key, value in first_summary.items()
        if not any(
            key.startswith(f"{metric}_") for metric in RATE_METRICS + MEAN_METRICS
        )
        and key != "failure_class_counts"
    }
    summary.update(
        {
            "episodes": args.episodes,
            "parallel_workers": workers,
            "seed_first": min(seeds),
            "seed_last": max(seeds),
            "training_performed": False,
            "optimizer_steps": 0,
            "backpropagation_performed": False,
        }
    )
    for metric in RATE_METRICS:
        count = sum(int(float(row[metric])) for row in episodes)
        low, high = wilson_interval(count, args.episodes)
        summary[f"{metric}_count"] = count
        summary[f"{metric}_rate"] = count / args.episodes
        summary[f"{metric}_ci95_low"] = low
        summary[f"{metric}_ci95_high"] = high
    for metric in MEAN_METRICS:
        mean, low, high, n = mean_ci(row[metric] for row in episodes)
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_ci95_low"] = low
        summary[f"{metric}_ci95_high"] = high
        summary[f"{metric}_n"] = n
    class_counts = {}
    for row in episodes:
        name = row["failure_class"]
        class_counts[name] = class_counts.get(name, 0) + 1
    summary["failure_class_counts"] = class_counts

    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    write_csv(
        outdir / "summary.csv",
        [
            {
                key: (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in summary.items()
            }
        ],
    )
    (outdir / "validation.json").write_text(
        json.dumps(
            {
                "episodes_expected": args.episodes,
                "episodes_observed": len(episodes),
                "unique_seeds": len(set(seeds)),
                "episode_ids_contiguous": True,
                "training_performed": False,
                "optimizer_steps": 0,
                "backpropagation_performed": False,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
