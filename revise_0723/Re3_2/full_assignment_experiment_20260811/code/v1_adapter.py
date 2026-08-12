#!/usr/bin/env python3
"""Adapter that applies the V1 IDBO optimizer to the current V2 assignment objective."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_v1_optimizer(v1_dir: Path):
    """Return a callable with the same interface as ``IDBO_paper``.

    Only the V1 optimizer is imported.  Its legacy fixed scene is deliberately
    not used; both versions are evaluated on the same current 3-D snapshots and
    the same ``Scenario.assignment_cost`` objective.
    """
    sys.path.insert(0, str(v1_dir))
    spec = importlib.util.spec_from_file_location("reviewer32_idbo_v1", v1_dir / "idbo.py")
    if spec is None or spec.loader is None:
        raise ImportError(v1_dir / "idbo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def wrapped(N, max_iter, scn, schedule="linear", seed=0):
        del schedule  # V1 has its own nonlinear schedule.
        np.random.seed(seed)

        def objective(x):
            assignment = np.clip(np.rint(x).astype(int), 0, scn.N_A - 1)
            return scn.assignment_cost(assignment)

        cost, position, curve = module.IDBO(
            N, max_iter, 0.0, float(scn.N_A - 1), scn.N_D, objective)
        assignment = np.clip(np.rint(position).astype(int), 0, scn.N_A - 1)
        return float(cost), assignment, np.asarray(curve)

    return wrapped
