#!/usr/bin/env python3
"""Replay selected partial-failure episodes and retain complete trajectories.

This script performs deterministic frozen-policy inference only.  It reuses the
same evaluator, model weights, case preset, noise, delay, and failure sampling
used by ``run_two_defender_failure_mc.py``; no optimizer or training code is
invoked.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def load_evaluator(code_root: Path, runner_path: Path):
    sys.path.insert(0, str(code_root))
    sys.path.insert(0, str(code_root / "onpolicy" / "scripts"))
    spec = importlib.util.spec_from_file_location("partial_failure_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load evaluator: {runner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("case1", "case2"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--hit-radius", type=float, default=3.0)
    parser.add_argument("--sync-tol", type=float, default=0.5)
    parser.add_argument("--sensor-delay-steps", type=int, default=1)
    parser.add_argument("--position-noise-std", type=float, default=3.0)
    parser.add_argument("--velocity-noise-std", type=float, default=0.3)
    parser.add_argument("--failed-count", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    evaluator = load_evaluator(args.code_root.resolve(), args.runner.resolve())
    completed = []
    for seed in args.seeds:
        episode_dir = args.outdir / f"seed_{seed}"
        run_args = SimpleNamespace(
            case=args.case,
            model_dir=args.model_dir.resolve(),
            preset=args.preset.resolve(),
            outdir=episode_dir,
            episodes=1,
            seed=seed,
            failed_count=args.failed_count,
            max_steps=args.max_steps,
            hit_radius=args.hit_radius,
            sync_tol=args.sync_tol,
            sensor_delay_steps=args.sensor_delay_steps,
            position_noise_std=args.position_noise_std,
            velocity_noise_std=args.velocity_noise_std,
            cpu=args.cpu,
        )
        evaluator.evaluate(run_args)
        trajectory = episode_dir / f"{args.case}_representative_success.npz"
        if not trajectory.exists():
            raise RuntimeError(
                f"seed {seed} did not reproduce a successful interception trajectory"
            )
        completed.append({"case": args.case, "seed": seed, "trajectory": str(trajectory)})

    (args.outdir / "replay_manifest.json").write_text(
        json.dumps(
            {
                "purpose": "deterministic frozen-policy trajectory replay",
                "training_performed": False,
                "optimizer_steps": 0,
                "backpropagation_performed": False,
                "case": args.case,
                "seeds": args.seeds,
                "code_root": str(args.code_root.resolve()),
                "model_dir": str(args.model_dir.resolve()),
                "preset": str(args.preset.resolve()),
                "episodes": completed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
