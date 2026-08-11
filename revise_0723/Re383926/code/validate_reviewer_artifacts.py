#!/usr/bin/env python3
"""Integrity and completeness checks for reviewer experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


TERMINAL_METRICS = ("E_co_time_s", "E_n_g", "E_miss_m", "E_t_s")
CORE_METRICS = (
    "mean_closest_approach_m",
    "worst_closest_approach_m",
    "mean_agent_return",
)


def merged_files(root: Path, filename: str):
    return [
        path
        for path in sorted(root.glob(f"**/{filename}"))
        if "_chunks" not in path.parts and "checkpoints" not in path.parts
    ]


def validate_episode_file(path: Path, expected_episodes: int, issues, checks):
    frame = pd.read_csv(path)
    checks.append({"file": str(path), "rows": len(frame)})
    group_cols = [
        column
        for column in ("variant", "case")
        if column in frame.columns
    ]
    grouped = (
        frame.groupby(group_cols, dropna=False, sort=True)
        if group_cols and frame[group_cols].drop_duplicates().shape[0] > 1
        else [((), frame)]
    )
    for keys, group in grouped:
        label = (
            dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            if group_cols and keys != ()
            else "file"
        )
        if len(group) != expected_episodes:
            issues.append(
                f"{path} {label}: expected {expected_episodes} episodes, "
                f"got {len(group)}"
            )
        if "seed" in group and group["seed"].duplicated().any():
            issues.append(f"{path} {label}: duplicate Monte Carlo seeds")
        elif "episode" in group and "training_seed" not in group:
            if group["episode"].duplicated().any():
                issues.append(f"{path} {label}: duplicate episode IDs")
    for metric in CORE_METRICS:
        if metric not in frame:
            issues.append(f"{path}: missing core metric {metric}")
        elif not np.isfinite(frame[metric].to_numpy(dtype=float)).all():
            issues.append(f"{path}: nonfinite values in core metric {metric}")
    if "all_defenders_hit" in frame:
        success = frame["all_defenders_hit"].astype(bool)
        for metric in TERMINAL_METRICS:
            if metric not in frame:
                issues.append(f"{path}: missing terminal metric {metric}")
                continue
            values = frame.loc[success, metric].to_numpy(dtype=float)
            if values.size and not np.isfinite(values).all():
                issues.append(
                    f"{path}: successful episodes contain nonfinite {metric}"
                )
    if "training_performed" in frame and frame[
        "training_performed"
    ].astype(bool).any():
        issues.append(f"{path}: training_performed is true in evaluation data")


def validate_summaries(root: Path, issues, checks):
    for path in sorted(root.glob("**/summary.json")):
        if "_chunks" in path.parts or "checkpoints" in path.parts:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        checks.append({"file": str(path), "summary_checked": True})
        if payload.get("training_performed") is not False:
            issues.append(f"{path}: training_performed is not false")
        if int(payload.get("optimizer_steps", -1)) != 0:
            issues.append(f"{path}: optimizer_steps is not zero")
        if payload.get("backpropagation_performed") is not False:
            issues.append(f"{path}: backpropagation_performed is not false")


def validate_training(root: Path, issues, checks):
    required = (
        "environment_steps",
        "mean_episode_return",
        "policy_loss",
        "value_loss",
        "entropy",
        "actor_lr",
    )
    for path in sorted(root.glob("**/training_metrics.csv")):
        frame = pd.read_csv(path)
        checks.append({"file": str(path), "training_rows": len(frame)})
        if frame.empty:
            issues.append(f"{path}: empty training metrics")
            continue
        if frame["environment_steps"].duplicated().any():
            issues.append(f"{path}: duplicate environment_steps")
        for column in required:
            if column not in frame:
                issues.append(f"{path}: missing training column {column}")
            elif not np.isfinite(frame[column].to_numpy(dtype=float)).all():
                issues.append(f"{path}: nonfinite training column {column}")


def validate_figures(root: Path, issues, checks):
    stems = {}
    for suffix in ("pdf", "svg", "png"):
        for path in root.glob(f"*.{suffix}"):
            stems.setdefault(path.stem, set()).add(suffix)
            if path.stat().st_size == 0:
                issues.append(f"{path}: empty figure")
    for stem, suffixes in sorted(stems.items()):
        missing = {"pdf", "svg", "png"} - suffixes
        if missing:
            issues.append(f"{stem}: missing figure formats {sorted(missing)}")
            continue
        png = root / f"{stem}.png"
        with Image.open(png) as image:
            dpi = image.info.get("dpi", (0.0, 0.0))
            checks.append(
                {
                    "file": str(png),
                    "pixel_size": list(image.size),
                    "dpi": list(dpi),
                }
            )
            if min(dpi) < 590.0:
                issues.append(f"{png}: PNG dpi below 600 target: {dpi}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_root", type=Path, required=True)
    parser.add_argument("--evaluation_root", type=Path, required=True)
    parser.add_argument("--supplementary_root", type=Path, required=True)
    parser.add_argument("--figures_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected_episodes", type=int, default=100)
    args = parser.parse_args()

    issues = []
    checks = []
    validate_training(args.training_root, issues, checks)
    for root in (args.evaluation_root, args.supplementary_root):
        for path in merged_files(root, "episodes.csv"):
            validate_episode_file(
                path, args.expected_episodes, issues, checks
            )
        validate_summaries(root, issues, checks)
    validate_figures(args.figures_root, issues, checks)
    report = {
        "status": "pass" if not issues else "fail",
        "expected_episodes_per_merged_file": args.expected_episodes,
        "checks": checks,
        "issues": issues,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
