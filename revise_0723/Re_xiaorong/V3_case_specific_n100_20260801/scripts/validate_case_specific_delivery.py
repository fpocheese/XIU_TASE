#!/usr/bin/env python3
"""Hard-gate validator for the final case-specific ablation bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


VARIANTS = ("full", "no_trust", "no_gru", "no_attention_residual")
CASES = ("case1", "case2")
RATE_COLUMNS = ("target_coverage_success", "all_defenders_hit", "cooperative_success", "mission_success")
TERMINAL_COLUMNS = ("E_co_time_s", "E_n_g", "E_miss_m", "E_t_s")
EXPECTED_UPDATES = {"case1": 585, "case2": 585}
EXPECTED_STEPS = {"case1": 599040, "case2": 599040}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_training(root: Path):
    rows = []
    expected_cells = {(v, c) for v in VARIANTS for c in CASES}
    found = set()
    for variant, case in sorted(expected_cells):
        run = root / variant / case / "seed8303"
        for name in ("training_metrics.csv", "run_manifest.json", "command_config.json"):
            require((run / name).is_file(), f"missing {run/name}")
        require((run / "models" / "checkpoint_latest.pt").is_file(), f"missing latest checkpoint: {run}")
        data = pd.read_csv(run / "training_metrics.csv")
        require(len(data) == EXPECTED_UPDATES[case], f"{variant}/{case}: update rows {len(data)}")
        require(int(data.environment_steps.iloc[-1]) == EXPECTED_STEPS[case], f"{variant}/{case}: final steps")
        require(data.environment_steps.is_monotonic_increasing, f"{variant}/{case}: nonmonotonic steps")
        require(not data.environment_steps.duplicated().any(), f"{variant}/{case}: duplicate steps")
        require(not data["update"].duplicated().any(), f"{variant}/{case}: duplicate updates")
        require(np.isfinite(data.select_dtypes(include=[np.number]).to_numpy()).all(), f"{variant}/{case}: NaN/Inf")
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        require(manifest["variant"] == variant and manifest["case"] == case, f"{variant}/{case}: manifest mismatch")
        found.add((variant, case))
        rows.append({"variant": variant, "case": case, "updates": len(data),
                     "final_steps": int(data.environment_steps.iloc[-1]),
                     "actor_parameters": manifest["parameter_count"]["actor"],
                     "critic_parameters": manifest["parameter_count"]["critic"],
                     "training_csv_sha256": sha256(run / "training_metrics.csv")})
    require(found == expected_cells, "training matrix incomplete")
    return rows


def validate_evaluation(root: Path):
    combined = root / "combined"
    episodes = pd.read_csv(combined / "episodes.csv")
    targets = pd.read_csv(combined / "targets.csv")
    assignments = pd.read_csv(combined / "assignments.csv")
    require(len(episodes) == 800, f"episodes rows {len(episodes)}")
    require(len(targets) == 6400, f"target rows {len(targets)}")
    require(len(assignments) == 16000, f"assignment rows {len(assignments)}")
    counts = episodes.groupby(["variant", "case"]).size()
    require(len(counts) == 8 and (counts == 100).all(), f"episode cell counts: {counts.to_dict()}")
    require(episodes.groupby(["variant", "case"])["seed"].nunique().eq(100).all(), "nonunique episode seeds")
    for case in CASES:
        seed_sets = [set(episodes[(episodes.case == case) & (episodes.variant == v)].seed) for v in VARIANTS]
        require(all(values == seed_sets[0] for values in seed_sets[1:]), f"{case}: test seeds not paired")
    for column in RATE_COLUMNS:
        require(episodes[column].isin([0, 1]).all(), f"{column}: not binary")
    complete = episodes.all_defenders_hit.eq(1)
    require(episodes.loc[complete, list(TERMINAL_COLUMNS)].notna().all(axis=1).all(), "complete episode missing terminal metric")
    finite = episodes[list(TERMINAL_COLUMNS)].to_numpy(float)
    require(np.isfinite(finite[~np.isnan(finite)]).all(), "terminal metric Inf")
    require(targets.groupby(["variant", "case", "seed"]).size().eq(8).all(), "not eight target rows per episode")
    require(assignments.groupby(["variant", "case", "seed"]).size().eq(20).all(), "not twenty assignments per episode")

    manifest = json.loads((root / "formal_evaluation_manifest.json").read_text(encoding="utf-8"))
    require(manifest["selection_uses_test_seeds"] is False, "selection leaked test seeds")
    require(manifest["paired_test_seeds_across_variants"] is True, "unpaired test design")
    require(manifest["training_performed"] is False, "training during evaluation")
    require(manifest["optimizer_steps"] == 0 and manifest["backpropagation_performed"] is False, "optimizer/backprop during evaluation")
    require(len(manifest["selection_jobs"]) == 8 and len(manifest["evaluation_jobs"]) == 8, "evaluation job matrix incomplete")
    status = pd.read_csv(root / "formal_evaluation_status.csv")
    require(len(status) == 16 and status.returncode.eq(0).all(), "evaluation status failure")
    for job in manifest["selection_jobs"]:
        selection = Path(job["outdir"])
        if not selection.is_dir():
            selection = root / "selection" / job["variant"] / job["case"] / f"seed{job['training_seed']}"
        require((selection / "selected_checkpoint.json").is_file(), f"missing selection: {selection}")
    return {"episodes": len(episodes), "targets": len(targets), "assignments": len(assignments),
            "episode_cell_counts": {f"{v}/{c}": int(counts.loc[(v, c)]) for v in VARIANTS for c in CASES}}


def validate_figures(root: Path):
    stems = []
    for case in CASES:
        stems.extend([f"ablation_training_{case}_v10",
                      f"ablation_training_reward_{case}_v10",
                      f"ablation_training_critic_loss_{case}_v10",
                      f"ablation_training_policy_entropy_{case}_v10",
                      f"ablation_monte_carlo_{case}_v10",
                      f"ablation_terminal_metrics_{case}_v10"])
    details = []
    for stem in stems:
        for suffix in ("pdf", "svg", "png"):
            path = root / f"{stem}.{suffix}"
            require(path.is_file() and path.stat().st_size > 5000, f"missing/small figure {path}")
            if suffix == "pdf": require(path.read_bytes()[:4] == b"%PDF", f"invalid PDF {path}")
            elif suffix == "svg": require("<svg" in path.read_text(encoding="utf-8")[:1000], f"invalid SVG {path}")
            else:
                with Image.open(path) as image:
                    require(image.width > 1000 and image.height > 500, f"small PNG dimensions {path}")
                    dpi = image.info.get("dpi", (0, 0))
                    require(min(dpi) >= 590, f"PNG below 600 dpi tolerance {path}: {dpi}")
            details.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = json.loads((root / "plot_manifest.json").read_text(encoding="utf-8"))
    require(manifest["png_dpi"] == 600 and manifest["monte_carlo_values_smoothed"] is False and manifest["missing_values_imputed"] is False, "plot manifest integrity")
    return details


def validate_reports(root: Path):
    required = ("README.md", "reviewer_response_en.tex", "reviewer_response_en.md",
                "审稿意见回复_中文.md", "manuscript_ablation_insertion.tex",
                "experiment_results_analysis_zh.md", "completion_audit_checklist.md")
    for name in required:
        path = root / name
        require(path.is_file() and path.stat().st_size > 200, f"missing report {path}")
    analysis = root / "analysis"
    for name in ("training_effect_summary.csv", "monte_carlo_n100_summary.csv",
                 "paired_test_comparisons.csv", "result_integrity_audit.json"):
        require((analysis / name).is_file(), f"missing analysis {name}")


def validate_tables(root: Path):
    names = ("paper_success_rates_n100.csv", "paper_success_rates_n100.tex",
             "paper_training_effects.csv", "paper_training_effects.tex",
             "paper_terminal_metrics_case1.csv", "paper_terminal_metrics_case1.tex",
             "paper_terminal_metrics_case2.csv", "paper_terminal_metrics_case2.tex",
             "paper_paired_binary_effects.csv", "paper_paired_binary_effects.tex")
    for name in names:
        path = root / name
        require(path.is_file() and path.stat().st_size > 100, f"missing/small table {path}")
    require(len(pd.read_csv(root / "paper_success_rates_n100.csv")) == 8, "success table rows")
    require(len(pd.read_csv(root / "paper_training_effects.csv")) == 8, "training table rows")
    require(len(pd.read_csv(root / "paper_paired_binary_effects.csv")) == 12, "paired table rows")
    return list(names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require_reports", action="store_true")
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    result = {"training": validate_training(root / "training"),
              "evaluation": validate_evaluation(root / "formal_evaluation_n100"),
              "figures": validate_figures(root / "figures_v10"),
              "tables": validate_tables(root / "tables")}
    if args.require_reports:
        validate_reports(root); result["reports"] = "passed"
    result["status"] = "PASS"
    out = args.out or (root / "FINAL_VALIDATION.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
