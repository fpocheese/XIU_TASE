#!/usr/bin/env python3
"""Validate the locally mirrored formal ART-MAPPO ablation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


VARIANTS = {
    "full",
    "no_trust",
    "no_gru",
    "no_attention_residual",
}
EXPECTED_UPDATES = {"case1": 100, "case2": 300}
CORE_EVAL_FIELDS = (
    "target_num",
    "target_hit_count",
    "target_sync_count",
    "mean_agent_return",
    "team_return",
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    train = root / "training"
    evaluation = root / "evaluation"
    analysis = root / "analysis"
    errors = []

    train_status_path = train / "suite_status.json"
    eval_status_path = evaluation / "evaluation_status.json"
    if not train_status_path.is_file():
        errors.append("missing training/suite_status.json")
        train_status = {}
    else:
        train_status = load_json(train_status_path)
        if train_status.get("status") != "complete":
            errors.append(
                f"training status is {train_status.get('status')!r}"
            )
        jobs = train_status.get("jobs", {})
        if len(jobs) != 40:
            errors.append(f"training status contains {len(jobs)} jobs")
        bad = {
            key: value.get("status")
            for key, value in jobs.items()
            if value.get("status") != "complete"
        }
        if bad:
            errors.append(f"incomplete training jobs: {bad}")

    if not eval_status_path.is_file():
        errors.append("missing evaluation/evaluation_status.json")
        eval_status = {}
    else:
        eval_status = load_json(eval_status_path)
        if eval_status.get("status") != "complete":
            errors.append(
                f"evaluation status is {eval_status.get('status')!r}"
            )
        jobs = eval_status.get("jobs", {})
        if len(jobs) != 40:
            errors.append(f"evaluation status contains {len(jobs)} jobs")
        bad = {
            key: value.get("status")
            for key, value in jobs.items()
            if value.get("status") != "complete"
        }
        if bad:
            errors.append(f"incomplete evaluation jobs: {bad}")

    metric_paths = sorted(train.glob("*/*/seed*/training_metrics.csv"))
    if len(metric_paths) != 40:
        errors.append(f"found {len(metric_paths)} training metric files")
    training_rows = 0
    for path in metric_paths:
        parts = path.relative_to(train).parts
        variant, case = parts[0], parts[1]
        if variant not in VARIANTS or case not in EXPECTED_UPDATES:
            errors.append(f"unexpected training path {path}")
            continue
        rows = read_csv(path)
        training_rows += len(rows)
        expected = EXPECTED_UPDATES[case]
        updates = [int(row["update"]) for row in rows]
        if updates != list(range(1, expected + 1)):
            errors.append(
                f"{path}: updates are incomplete or duplicated "
                f"({len(updates)} rows, expected {expected})"
            )
        for row_index, row in enumerate(rows, 1):
            for name, value in row.items():
                if not finite_number(value):
                    errors.append(
                        f"{path}:{row_index + 1} non-finite {name}={value!r}"
                    )
                    break

    eval_paths = sorted(
        evaluation.glob("*/*/seed*/case*/case*_episode_summary.csv")
    )
    if len(eval_paths) != 40:
        errors.append(f"found {len(eval_paths)} evaluation episode files")
    evaluation_rows = 0
    for path in eval_paths:
        rows = read_csv(path)
        evaluation_rows += len(rows)
        if len(rows) != 20:
            errors.append(f"{path}: {len(rows)} episodes, expected 20")
        for row_index, row in enumerate(rows, 1):
            for name in CORE_EVAL_FIELDS:
                if not finite_number(row.get(name)):
                    errors.append(
                        f"{path}:{row_index + 1} non-finite "
                        f"{name}={row.get(name)!r}"
                    )

    required_analysis = (
        "analysis_manifest.json",
        "ablation_seed_level_metrics.csv",
        "ablation_aggregate_metrics.csv",
        "ablation_paired_statistics.csv",
        "art_mappo_component_ablation.pdf",
        "art_mappo_component_ablation.svg",
        "art_mappo_component_ablation.png",
    )
    for name in required_analysis:
        path = analysis / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty analysis/{name}")
    analysis_manifest_path = analysis / "analysis_manifest.json"
    if analysis_manifest_path.is_file():
        manifest = load_json(analysis_manifest_path)
        if not manifest.get("input_validation", {}).get("passed", False):
            errors.append("remote analysis input validation did not pass")

    report = {
        "passed": not errors,
        "root": str(root),
        "training_metric_files": len(metric_paths),
        "training_metric_rows": training_rows,
        "evaluation_episode_files": len(eval_paths),
        "evaluation_episodes": evaluation_rows,
        "errors": errors,
    }
    report_path = root / "local_validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
