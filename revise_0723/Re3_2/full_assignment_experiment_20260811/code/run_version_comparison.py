#!/usr/bin/env python3
"""Paired V1/V2 assignment comparison on identical 3-D snapshots."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from assignment_delay_model import current_scenario, load_idbo, run_idbo_records
from v1_adapter import load_v1_optimizer


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def read(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-dir", type=Path, required=True)
    ap.add_argument("--v2-dir", type=Path, required=True)
    ap.add_argument("--experiment-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    Scenario, v2 = load_idbo(args.v2_dir); v1 = load_v1_optimizer(args.v1_dir)
    rows = []
    for seed in range(30):
        scn = current_scenario(Scenario, 20, 8, 1200 + seed)
        for version, optimizer in [("V1", v1), ("V2", v2)]:
            records, cost, max_load, _ = run_idbo_records(
                scn, optimizer, 32, 30, 9000 + seed)
            a = np.array([r.target for r in records], int)
            load = np.bincount(a, minlength=8)
            rows.append({"version": version, "seed": seed,
                         "assignment_cost": cost,
                         "mean_interception_probability": float(np.mean(scn.p_int[np.arange(20), a])),
                         "mean_adversarial_advantage": float(np.mean(scn.chi_static[np.arange(20), a])),
                         "mean_pair_score": float(np.mean((scn.p_int + scn.lam_A * scn.chi_static)[np.arange(20), a])),
                         "target_load_std": float(np.std(load)),
                         "covered_target_fraction": float(np.mean(load > 0)),
                         "capacity_feasible": int(max_load <= scn.L_max),
                         "maximum_target_load": max_load})
    write(args.out_dir / "paired_quality_raw.csv", rows)

    summary = []
    for version in ["V1", "V2"]:
        subset = [r for r in rows if r["version"] == version]
        row = {"version": version, "n_scenes": len(subset)}
        for metric in ["assignment_cost", "mean_interception_probability",
                       "mean_adversarial_advantage", "mean_pair_score",
                       "target_load_std", "covered_target_fraction", "capacity_feasible"]:
            x = np.array([float(r[metric]) for r in subset])
            row[f"{metric}_mean"] = float(np.mean(x))
            row[f"{metric}_std"] = float(np.std(x, ddof=1))
            row[f"{metric}_ci95"] = float(1.96 * np.std(x, ddof=1) / np.sqrt(len(x)))
        summary.append(row)
    write(args.out_dir / "paired_quality_summary.csv", summary)

    combined = []
    for version in ["V1", "V2"]:
        root = args.experiment_root / version / "data"
        rt = read(root / "runtime_summary.csv")
        dy = read(root / "dynamic_summary.csv")
        ch = read(root / "dynamic_change_rate.csv")
        size20 = next(r for r in rt if r["sweep"] == "problem_size" and int(r["M"]) == 20)
        size160 = next(r for r in rt if r["sweep"] == "problem_size" and int(r["M"]) == 160)
        d100 = next(r for r in dy if int(r["delay_ms"]) == 100)
        d200 = next(r for r in dy if int(r["delay_ms"]) == 200)
        q = next(r for r in summary if r["version"] == version)
        combined.append({"version": version,
                         "runtime_20x8_s": size20["runtime_s_mean"],
                         "runtime_160x64_s": size160["runtime_s_mean"],
                         "static_assignment_cost": q["assignment_cost_mean"],
                         "mean_pair_score": q["mean_pair_score_mean"],
                         "capacity_feasible_rate": q["capacity_feasible_mean"],
                         "dynamic_change_fraction": float(np.mean(
                             [float(r["mean_winner_change_fraction"]) for r in ch])),
                         "agreement_at_100ms": d100["winner_jaccard_mean"],
                         "recovery_at_100ms": d100["recovery_rate_mean"],
                         "agreement_at_200ms": d200["winner_jaccard_mean"],
                         "recovery_at_200ms": d200["recovery_rate_mean"]})
    write(args.out_dir / "version_comparison_summary.csv", combined)
    result = {"paired_scenes": 30,
              "v1_cost_mean": summary[0]["assignment_cost_mean"],
              "v2_cost_mean": summary[1]["assignment_cost_mean"],
              "v2_minus_v1_cost_percent": 100 * (
                  summary[1]["assignment_cost_mean"] / summary[0]["assignment_cost_mean"] - 1)}
    (args.out_dir / "comparison_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
