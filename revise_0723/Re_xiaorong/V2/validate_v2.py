#!/usr/bin/env python3
"""Validate the self-contained V2 ablation package."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


VARIANTS = {
    "full": "Full-ART-MAPPO",
    "no_trust": "No-Trust",
    "no_gru": "No-GRU",
    "no_attention_residual": "No-Attn-Residual",
}
CASES = {"case1": 100, "case2": 300}
FIELDS = {
    "rewards": "mean_episode_return",
    "critic_loss": "value_loss",
    "entropy": "entropy",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("v2_root", type=Path)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    errors: list[str] = []

    for variant, file_label in VARIANTS.items():
        for case, expected_rows in CASES.items():
            for seed in range(701, 706):
                raw_path = (
                    args.raw_root
                    / "training"
                    / variant
                    / case
                    / f"seed{seed}"
                    / "training_metrics.csv"
                )
                raw = rows(raw_path)
                if len(raw) != expected_rows:
                    errors.append(f"{raw_path}: row count")
                    continue
                npy_root = args.v2_root / "data" / "converted_npy" / case
                prefix = f"{file_label}_seed{seed}"
                expected_steps = np.asarray(
                    [float(row["environment_steps"]) for row in raw]
                )
                actual_steps = np.load(npy_root / f"{prefix}_steps.npy")
                if not np.array_equal(expected_steps, actual_steps):
                    errors.append(f"{prefix}: steps NPY mismatch")
                for suffix, field in FIELDS.items():
                    expected = np.asarray([float(row[field]) for row in raw])
                    actual = np.load(npy_root / f"{prefix}_{suffix}.npy")
                    if not np.array_equal(expected, actual):
                        errors.append(f"{prefix}: {suffix} NPY mismatch")

    plot_csvs = sorted((args.v2_root / "data" / "plot_csv").rglob("*.csv"))
    if len(plot_csvs) != 24:
        errors.append(f"expected 24 plot CSVs, found {len(plot_csvs)}")
    for path in plot_csvs:
        case = path.parent.name
        data = rows(path)
        if case not in CASES or len(data) != CASES[case]:
            errors.append(f"{path}: unexpected case/row count")
            continue
        for row in data:
            values = [
                float(row["environment_steps_k"]),
                float(row["mean"]),
                float(row["shadow_lower"]),
                float(row["shadow_upper"]),
            ]
            if not all(math.isfinite(value) for value in values):
                errors.append(f"{path}: non-finite plotted value")
                break
            if not values[2] <= values[1] <= values[3]:
                errors.append(f"{path}: invalid confidence bounds")
                break

    expected_stems = [
        "ablation_training_reward",
        "ablation_critic_loss",
        "ablation_policy_entropy",
        "ablation_training_metrics_combined",
        "monte_carlo_component_ablation",
    ]
    for stem in expected_stems:
        for extension in (".pdf", ".svg", ".png"):
            path = args.v2_root / "figures" / f"{stem}{extension}"
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing/empty figure: {path}")

    try:
        from PIL import Image

        for path in (args.v2_root / "figures").glob("*.png"):
            with Image.open(path) as image:
                image.verify()
                dpi = image.info.get("dpi")
                if path.name.startswith("ablation_") and dpi:
                    if min(dpi) < 590:
                        errors.append(f"{path}: expected 600 dpi, found {dpi}")
    except ImportError:
        pass

    required_docs = [
        "README_V2.md",
        "V2_experiment_analysis.md",
        "reviewer_response_en_V2.md",
        "审稿意见回复_中文_V2.md",
        "manuscript_insert_training_ablation_V2.tex",
        "reference_plot_audit.md",
        "prepare_and_plot_training_ablation_v2.py",
        "V2_AUDIT.json",
    ]
    for name in required_docs:
        path = args.v2_root / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing document: {path}")

    report = {
        "passed": not errors,
        "v2_root": str(args.v2_root.resolve()),
        "raw_root": str(args.raw_root.resolve()),
        "raw_training_runs_checked": 40,
        "raw_npy_arrays_checked": 160,
        "plot_csv_files": len(plot_csvs),
        "figure_files": len(list((args.v2_root / "figures").glob("*"))),
        "errors": errors,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
