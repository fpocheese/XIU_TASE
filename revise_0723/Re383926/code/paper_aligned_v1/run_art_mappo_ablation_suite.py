#!/usr/bin/env python
"""Fault-tolerant queue for the ART-MAPPO component ablation."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


DEFAULT_VARIANTS = [
    "full",
    "no_trust",
    "no_gru",
    "no_attention_residual",
]


def atomic_json(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def completed(run_dir, expected_updates):
    checkpoint = run_dir / "models" / "checkpoint_latest.pt"
    metrics = run_dir / "training_metrics.csv"
    if not checkpoint.exists() or not metrics.exists():
        return False
    try:
        with open(metrics, "r", encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]
        return len(lines) - 1 >= expected_updates
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument(
        "--case1_steps",
        type=int,
        default=None,
        help="Optional Case 1 budget; defaults to --steps.",
    )
    parser.add_argument(
        "--case2_steps",
        type=int,
        default=None,
        help="Optional Case 2 budget; defaults to --steps.",
    )
    parser.add_argument("--episode_length", type=int, default=1500)
    parser.add_argument("--rollout_threads", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[701, 702, 703, 704, 705])
    parser.add_argument("--cases", nargs="+", choices=["case1", "case2"], default=["case1", "case2"])
    parser.add_argument("--variants", nargs="+", choices=DEFAULT_VARIANTS, default=DEFAULT_VARIANTS)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--stall_minutes", type=float, default=20.0)
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=1,
        help="Number of independent seed/case/variant runs trained concurrently.",
    )
    args = parser.parse_args()
    if args.max_parallel < 1:
        parser.error("--max_parallel must be at least 1")
    if any(
        value is not None and value < args.episode_length
        for value in (args.case1_steps, args.case2_steps)
    ):
        parser.error("case-specific budgets must cover at least one rollout")

    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "suite_status.json"
    train_script = Path(__file__).with_name("train_art_mappo_ablation_3d.py")
    case_steps = {
        "case1": args.case1_steps or args.steps,
        "case2": args.case2_steps or args.steps,
    }
    jobs = [
        {
            "variant": variant,
            "case": case,
            "seed": seed,
            "steps": int(case_steps[case]),
            "expected_updates": max(
                1,
                int(case_steps[case])
                // args.episode_length
                // args.rollout_threads,
            ),
        }
        for case in args.cases
        for seed in args.seeds
        for variant in args.variants
    ]
    state = {
        "started_at": time.time(),
        "steps": args.steps,
        "case_steps": case_steps,
        "jobs_total": len(jobs),
        "max_parallel": args.max_parallel,
        "jobs": {},
    }
    atomic_json(state_path, state)
    state_lock = threading.Lock()

    def update_state(key, payload):
        with state_lock:
            state["jobs"].setdefault(key, {}).update(payload)
            atomic_json(state_path, state)

    def run_job(index, job):
        key = f"{job['variant']}/{job['case']}/seed{job['seed']}"
        run_dir = root / job["variant"] / job["case"] / f"seed{job['seed']}"
        if completed(run_dir, job["expected_updates"]):
            update_state(key, {"status": "complete", "skipped": True})
            return key

        log_dir = root / "suite_logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / (
            f"{job['variant']}_{job['case']}_seed{job['seed']}.log"
        )
        success = False
        for attempt in range(args.max_retries + 1):
            command = [
                sys.executable,
                str(train_script),
                "--variant",
                job["variant"],
                "--case_3d",
                job["case"],
                "--seed",
                str(job["seed"]),
                "--save_dir",
                str(root),
                "--compare_steps",
                str(job["steps"]),
                "--episode_length",
                str(args.episode_length),
                "--n_rollout_threads",
                str(args.rollout_threads),
                "--resume",
            ]
            update_state(
                key,
                {
                    "status": "running",
                    "index": index,
                    "attempt": attempt,
                    "command": command,
                    "started_at": time.time(),
                },
            )
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(
                    f"\n[SUITE] attempt={attempt} command={' '.join(command)}\n"
                )
                log.flush()
                env = os.environ.copy()
                env.setdefault("CUDA_VISIBLE_DEVICES", "0")
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=str(train_script.parents[2]),
                    env=env,
                    start_new_session=True,
                )
                last_progress = time.time()
                last_mtime = 0.0
                while process.poll() is None:
                    time.sleep(30)
                    metrics = run_dir / "training_metrics.csv"
                    if metrics.exists():
                        mtime = metrics.stat().st_mtime
                        if mtime > last_mtime:
                            last_mtime = mtime
                            last_progress = time.time()
                    update_state(key, {"heartbeat": time.time()})
                    if time.time() - last_progress > args.stall_minutes * 60:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                        log.write("[SUITE] stalled; process group terminated\n")
                        break
                return_code = process.poll()
                if return_code is None:
                    return_code = process.wait()
            if return_code == 0 and completed(
                run_dir, job["expected_updates"]
            ):
                success = True
                break
            update_state(
                key,
                {
                    "last_return_code": return_code,
                    "status": "retrying",
                },
            )

        update_state(
            key,
            {
                "status": "complete" if success else "failed",
                "finished_at": time.time(),
            },
        )
        if not success:
            raise RuntimeError(f"job failed after retries: {key}")
        return key

    failures = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_parallel
    ) as executor:
        future_to_job = {
            executor.submit(run_job, index, job): job
            for index, job in enumerate(jobs, start=1)
        }
        for future in concurrent.futures.as_completed(future_to_job):
            try:
                future.result()
            except Exception as exc:  # keep other independent runs alive
                failures.append((future_to_job[future], repr(exc)))

    with state_lock:
        state["status"] = "failed" if failures else "complete"
        state["finished_at"] = time.time()
        state["failures"] = failures
        atomic_json(state_path, state)
    if failures:
        raise RuntimeError(f"{len(failures)} suite job(s) failed: {failures}")
    print(f"[SUITE DONE] {root}")


if __name__ == "__main__":
    main()
