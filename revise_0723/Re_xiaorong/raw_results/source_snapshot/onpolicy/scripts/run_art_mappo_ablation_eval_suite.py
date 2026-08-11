#!/usr/bin/env python
"""Evaluate every ART-MAPPO ablation checkpoint on paired held-out seeds."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


VARIANTS = ["full", "no_trust", "no_gru", "no_attention_residual"]


def atomic_json(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--cases", nargs="+", choices=["case1", "case2"], default=["case1", "case2"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--episodes_per_seed_case", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--eval_seed_offset", type=int, default=100_000)
    parser.add_argument("--max_parallel", type=int, default=1)
    args = parser.parse_args()
    if args.max_parallel < 1:
        parser.error("--max_parallel must be at least 1")

    training_root = Path(args.training_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    evaluator = Path(__file__).with_name("eval_art_mappo_ablation_3d.py")
    jobs = [
        (variant, case, seed)
        for case in args.cases
        for seed in args.seeds
        for variant in args.variants
    ]
    state = {
        "started_at": time.time(),
        "jobs_total": len(jobs),
        "max_parallel": args.max_parallel,
        "jobs": {},
    }
    status_path = output_root / "evaluation_status.json"
    atomic_json(status_path, state)
    state_lock = threading.Lock()

    def update_state(key, payload):
        with state_lock:
            state["jobs"].setdefault(key, {}).update(payload)
            atomic_json(status_path, state)

    def run_job(variant, case, seed):
        key = f"{variant}/{case}/seed{seed}"
        model_dir = (
            training_root / variant / case / f"seed{seed}" / "models"
        )
        run_out = output_root / variant / case / f"seed{seed}"
        expected = run_out / case / f"{case}_episode_summary.csv"
        if expected.exists():
            update_state(key, {"status": "complete", "skipped": True})
            return key
        if not (model_dir / "actor.pt").exists():
            raise FileNotFoundError(model_dir / "actor.pt")
        run_out.mkdir(parents=True, exist_ok=True)
        eval_seed = args.eval_seed_offset + seed
        command = [
            sys.executable,
            str(evaluator),
            "--variant",
            variant,
            "--model_dir",
            str(model_dir),
            "--outdir",
            str(run_out),
            f"--{case}",
            "--seed",
            str(eval_seed),
            "--eval_episodes",
            str(args.episodes_per_seed_case),
            "--max_steps",
            str(args.max_steps),
            "--eval_different_seed",
            "--require_all_hit",
        ]
        update_state(
            key,
            {
                "status": "running",
                "command": command,
                "started_at": time.time(),
            },
        )
        with open(run_out / "evaluation.log", "w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=str(evaluator.parents[2]),
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
            )
        update_state(
            key,
            {
                "status": "complete" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "finished_at": time.time(),
            },
        )
        if result.returncode != 0:
            raise RuntimeError(f"evaluation failed: {key}")
        return key

    failures = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_parallel
    ) as executor:
        future_to_job = {
            executor.submit(run_job, variant, case, seed): (
                variant,
                case,
                seed,
            )
            for variant, case, seed in jobs
        }
        for future in concurrent.futures.as_completed(future_to_job):
            try:
                future.result()
            except Exception as exc:
                failures.append((future_to_job[future], repr(exc)))

    with state_lock:
        state["status"] = "failed" if failures else "complete"
        state["finished_at"] = time.time()
        state["failures"] = failures
        atomic_json(status_path, state)
    if failures:
        raise RuntimeError(
            f"{len(failures)} evaluation job(s) failed: {failures}"
        )
    print(f"[EVALUATION DONE] {output_root}")


if __name__ == "__main__":
    main()
