#!/usr/bin/env python3
"""Select a frozen policy checkpoint on held-out validation seeds only.

No gradients, optimizer updates, or training rollouts are performed.  The
lexicographic selection rule is declared in ``SELECTION_COLUMNS`` and applied
identically to every ablation variant.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch


SELECTION_COLUMNS = [
    ("mission_success_rate", False),
    ("target_coverage_success_rate", False),
    ("mean_coordinated_groups", False),
    ("mean_complete_groups", False),
    ("mean_targets_covered", False),
    ("mean_closest_approach_m", True),
    ("mean_agent_return", False),
]


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_checkpoint(checkpoint: Path, outdir: Path):
    payload = torch.load(str(checkpoint), map_location="cpu")
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(payload["actor"], str(outdir / "actor.pt"))
    torch.save(payload["critic"], str(outdir / "critic.pt"))
    next_update = int(payload["next_update"])
    environment_steps = payload.get("environment_steps")
    if environment_steps is None:
        steps_per_update = payload.get("rollout_steps_per_update")
        if steps_per_update is not None:
            environment_steps = next_update * int(steps_per_update)
    return next_update, (
        int(environment_steps) if environment_steps is not None else None
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["case1", "case2"], required=True)
    parser.add_argument(
        "--variant",
        choices=[
            "full",
            "no_trust",
            "no_gru",
            "no_attention_residual",
        ],
        required=True,
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=97001)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument(
        "--initial_perturbation_scale",
        type=float,
        default=1.0,
        help="Paired physical initial-state perturbation scale.",
    )
    parser.add_argument(
        "--legacy_steps_per_update",
        type=int,
        default=None,
        help=(
            "Required only for legacy checkpoints that predate embedded "
            "environment-step metadata; never guessed from the update index."
        ),
    )
    parser.add_argument(
        "--checkpoint_stride",
        type=int,
        default=1,
        help="Evaluate every nth named checkpoint after sorting.",
    )
    args = parser.parse_args()
    if args.episodes < 1 or args.workers < 1 or args.checkpoint_stride < 1:
        parser.error("episodes, workers, and stride must be positive")

    model_dir = Path(args.model_dir).resolve()
    outdir = Path(args.outdir).resolve()
    all_checkpoints = sorted(model_dir.glob("checkpoint_update_*.pt"))
    checkpoints = all_checkpoints[:: args.checkpoint_stride]
    # A stride is a computationally efficient pre-registered subsampling
    # rule, but it must never silently omit the final trained checkpoint.
    if all_checkpoints and all_checkpoints[-1] not in checkpoints:
        checkpoints.append(all_checkpoints[-1])
    if not checkpoints:
        raise FileNotFoundError(f"no named checkpoints below {model_dir}")
    evaluator = Path(__file__).with_name("run_reviewer_eval_parallel.py")

    rows = []
    for checkpoint in checkpoints:
        tag = checkpoint.stem.replace("checkpoint_", "")
        checkpoint_root = outdir / "checkpoints" / tag
        exported = checkpoint_root / "model"
        next_update, environment_steps = export_checkpoint(
            checkpoint, exported
        )
        if environment_steps is None:
            if args.legacy_steps_per_update is None:
                raise ValueError(
                    f"{checkpoint} lacks environment-step metadata; pass "
                    "--legacy_steps_per_update explicitly"
                )
            environment_steps = (
                next_update * int(args.legacy_steps_per_update)
            )
        evaluation = checkpoint_root / "validation"
        command = [
            sys.executable,
            str(evaluator),
            "--episodes",
            str(args.episodes),
            "--workers",
            str(min(args.workers, args.episodes)),
            "--seed",
            str(args.seed),
            "--outdir",
            str(evaluation),
            "--condition",
            f"checkpoint_validation_{tag}",
            "--case",
            args.case,
            "--variant",
            args.variant,
            "--model_dir",
            str(exported),
            "--max_steps",
            str(args.max_steps),
            "--assignment_mode",
            "fixed",
            "--initial_perturbation_scale",
            str(args.initial_perturbation_scale),
            "--cpu_eval",
        ]
        with (checkpoint_root / "validation.log").open(
            "w", encoding="utf-8"
        ) as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            raise RuntimeError(
                f"validation failed for {checkpoint}: {completed.returncode}"
            )
        summary = json.loads(
            (evaluation / "summary.json").read_text(encoding="utf-8")
        )
        episodes = pd.read_csv(evaluation / "episodes.csv")
        row = {
            "checkpoint": str(checkpoint),
            "checkpoint_tag": tag,
            "next_update": next_update,
            "environment_steps": environment_steps,
            "validation_seed_first": args.seed,
            "validation_seed_last": args.seed + args.episodes - 1,
            "validation_episodes": args.episodes,
            "mission_success_rate": summary["mission_success_rate"],
            "target_coverage_success_rate": summary[
                "target_coverage_success_rate"
            ],
            "mean_coordinated_groups": float(
                episodes["coordinated_groups"].mean()
            ),
            "mean_complete_groups": float(
                episodes["complete_groups"].mean()
            ),
            "mean_targets_covered": float(
                episodes["targets_covered"].mean()
            ),
            "mean_closest_approach_m": float(
                episodes["mean_closest_approach_m"].mean()
            ),
            "worst_closest_approach_m": float(
                episodes["worst_closest_approach_m"].mean()
            ),
            "mean_agent_return": float(
                episodes["mean_agent_return"].mean()
            ),
            "training_performed": False,
            "optimizer_steps": 0,
            "backpropagation_performed": False,
        }
        rows.append(row)
        print(
            f"[{tag}] mission={row['mission_success_rate']:.3f} "
            f"covered={row['mean_targets_covered']:.3f} "
            f"closest={row['mean_closest_approach_m']:.3f}",
            flush=True,
        )

    def ranking_key(row):
        key = []
        for column, minimize in SELECTION_COLUMNS:
            value = float(row[column])
            if not np.isfinite(value):
                value = np.inf if minimize else -np.inf
            key.append(value if minimize else -value)
        key.append(int(row["next_update"]))
        return tuple(key)

    rows.sort(key=lambda row: int(row["next_update"]))
    selected = min(rows, key=ranking_key)
    write_csv(outdir / "checkpoint_validation.csv", rows)
    selected_root = outdir / "selected_model"
    selected_root.mkdir(parents=True, exist_ok=True)
    selected_export = (
        outdir
        / "checkpoints"
        / selected["checkpoint_tag"]
        / "model"
    )
    shutil.copy2(selected_export / "actor.pt", selected_root / "actor.pt")
    shutil.copy2(selected_export / "critic.pt", selected_root / "critic.pt")
    selection = {
        "case": args.case,
        "variant": args.variant,
        "selection_rule": [
            {
                "metric": column,
                "direction": "minimize" if minimize else "maximize",
            }
            for column, minimize in SELECTION_COLUMNS
        ],
        "validation_seed_first": args.seed,
        "validation_seed_last": args.seed + args.episodes - 1,
        "validation_episodes": args.episodes,
        "selected": selected,
        "test_seeds_used_for_selection": False,
        "training_performed": False,
        "optimizer_steps": 0,
        "backpropagation_performed": False,
    }
    (outdir / "selected_checkpoint.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
