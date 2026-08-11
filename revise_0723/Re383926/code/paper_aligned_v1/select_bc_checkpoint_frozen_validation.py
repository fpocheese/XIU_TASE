#!/usr/bin/env python3
"""Select a guide-initialized actor using held-out frozen-policy validation."""

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


SELECTION_COLUMNS = (
    ("mission_success_rate", False),
    ("target_coverage_success_rate", False),
    ("mean_coordinated_groups", False),
    ("mean_complete_groups", False),
    ("mean_targets_covered", False),
    ("mean_closest_approach_m", True),
    ("mean_agent_return", False),
)


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_indices(text: str, available: dict[int, Path]):
    if text.lower() == "all":
        return sorted(available)
    indices = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("checkpoint indices must be nonempty and unique")
    missing = [index for index in indices if index not in available]
    if missing:
        raise FileNotFoundError(f"missing BC checkpoints: {missing}")
    return indices


def ranking_key(row):
    key = []
    for column, minimize in SELECTION_COLUMNS:
        value = float(row[column])
        if not np.isfinite(value):
            value = np.inf if minimize else -np.inf
        key.append(value if minimize else -value)
    key.append(int(row["bc_episode"]))
    return tuple(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["case1", "case2"], required=True)
    parser.add_argument(
        "--variant",
        choices=["full", "no_gru", "no_attention_residual"],
        required=True,
    )
    parser.add_argument("--pretrain_dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--indices",
        default="1,2,5,10,15,20",
        help="Comma-separated BC episode checkpoints or 'all'.",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=97501)
    args = parser.parse_args()

    pretrain_dir = Path(args.pretrain_dir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    critic = pretrain_dir / "model" / "critic.pt"
    if not critic.is_file():
        raise FileNotFoundError(critic)
    available = {}
    for path in sorted(
        (pretrain_dir / "checkpoints").glob("actor_episode_*.pt")
    ):
        available[int(path.stem.rsplit("_", 1)[-1])] = path
    indices = parse_indices(args.indices, available)
    evaluator = Path(__file__).with_name("run_reviewer_eval_parallel.py")
    rows = []

    for index in indices:
        root = outdir / "checkpoints" / f"episode_{index:04d}"
        model = root / "model"
        model.mkdir(parents=True, exist_ok=True)
        shutil.copy2(available[index], model / "actor.pt")
        shutil.copy2(critic, model / "critic.pt")
        evaluation = root / "validation"
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
            f"bc_checkpoint_validation_episode_{index:04d}",
            "--case",
            args.case,
            "--variant",
            args.variant,
            "--model_dir",
            str(model),
            "--assignment_mode",
            "fixed",
            "--cpu_eval",
        ]
        with (root / "validation.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            raise RuntimeError(
                f"BC checkpoint validation failed: episode {index}"
            )
        summary = json.loads(
            (evaluation / "summary.json").read_text(encoding="utf-8")
        )
        episode_data = pd.read_csv(evaluation / "episodes.csv")
        row = {
            "bc_episode": index,
            "actor_checkpoint": str(available[index]),
            "validation_seed_first": args.seed,
            "validation_seed_last": args.seed + args.episodes - 1,
            "validation_episodes": args.episodes,
            "mission_success_rate": summary["mission_success_rate"],
            "target_coverage_success_rate": summary[
                "target_coverage_success_rate"
            ],
            "mean_coordinated_groups": float(
                episode_data["coordinated_groups"].mean()
            ),
            "mean_complete_groups": float(
                episode_data["complete_groups"].mean()
            ),
            "mean_targets_covered": float(
                episode_data["targets_covered"].mean()
            ),
            "mean_closest_approach_m": float(
                episode_data["mean_closest_approach_m"].mean()
            ),
            "worst_closest_approach_m": float(
                episode_data["worst_closest_approach_m"].mean()
            ),
            "mean_agent_return": float(
                episode_data["mean_agent_return"].mean()
            ),
            "training_performed_during_validation": False,
            "optimizer_steps_during_validation": 0,
            "backpropagation_during_validation": False,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    rows.sort(key=lambda row: int(row["bc_episode"]))
    selected = min(rows, key=ranking_key)
    write_csv(outdir / "bc_checkpoint_validation.csv", rows)
    selected_model = outdir / "selected_model"
    selected_model.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        selected["actor_checkpoint"], selected_model / "actor.pt"
    )
    shutil.copy2(critic, selected_model / "critic.pt")
    result = {
        "case": args.case,
        "variant": args.variant,
        "candidate_bc_episodes": indices,
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
        "training_performed_during_validation": False,
        "optimizer_steps_during_validation": 0,
        "backpropagation_during_validation": False,
    }
    (outdir / "selected_bc_checkpoint.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
