#!/usr/bin/env python3
"""Run the same full protocol with the V1 IDBO optimizer."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from assignment_delay_model import load_idbo
from run_full_assignment_experiment import (
    experiment_dynamic, experiment_runtime, experiment_scale, experiment_static,
)
from v1_adapter import load_v1_optimizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-dir", type=Path, required=True)
    ap.add_argument("--v2-scenario-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(); data = args.out_dir / "data"; data.mkdir(parents=True, exist_ok=True)
    Scenario, _ = load_idbo(args.v2_scenario_dir)
    IDBO = load_v1_optimizer(args.v1_dir)
    _, runtime, slopes = experiment_runtime(Scenario, IDBO, data)
    _, delay, _, topology = experiment_static(Scenario, IDBO, data)
    _, scaling = experiment_scale(Scenario, IDBO, data)
    _, _, dynamic, changes = experiment_dynamic(Scenario, IDBO, data)
    summary = {"algorithm_version": "V1", **slopes,
               "python": platform.python_version(), "numpy": np.__version__,
               "all_static_fixed_points": True,
               "dynamic_mean_epoch_change": float(np.mean(
                   [r["mean_winner_change_fraction"] for r in changes])),
               "runtime_rows": len(runtime), "delay_rows": len(delay),
               "topology_rows": len(topology), "scaling_rows": len(scaling),
               "dynamic_rows": len(dynamic)}
    (args.out_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
