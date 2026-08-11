#!/usr/bin/env python3
"""Reproducible reviewer study for the many-to-one IDBO formulation.

The decision vector has one target index per interceptor. Feasibility repair
enforces at least one and at most ceil(M/N) interceptors per target. The
objective is the sum of target survival probabilities used in the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Problem:
    probability: np.ndarray
    max_load: int

    @property
    def defenders(self) -> int:
        return int(self.probability.shape[0])

    @property
    def targets(self) -> int:
        return int(self.probability.shape[1])


def make_problem(defenders: int, targets: int, seed: int, zem_weight: float = 0.5) -> Problem:
    rng = np.random.default_rng(seed)
    target_angle = np.linspace(0.0, 2.0 * np.pi, targets, endpoint=False)
    target_radius = rng.uniform(1750.0, 2200.0, targets)
    target_pos = np.column_stack(
        (target_radius * np.cos(target_angle), target_radius * np.sin(target_angle))
    )
    defender_angle = rng.uniform(0.0, 2.0 * np.pi, defenders)
    defender_radius = rng.uniform(20.0, 90.0, defenders)
    defender_pos = np.column_stack(
        (defender_radius * np.cos(defender_angle), defender_radius * np.sin(defender_angle))
    )
    defender_heading = rng.uniform(-np.pi, np.pi, defenders)
    los = target_pos[None, :, :] - defender_pos[:, None, :]
    distance = np.linalg.norm(los, axis=2)
    bearing = np.arctan2(los[:, :, 1], los[:, :, 0])
    heading_error = np.abs(np.angle(np.exp(1j * (bearing - defender_heading[:, None]))))

    # A bounded proxy for the two terms in Eqs. (ZEM probability, angular probability).
    zem_proxy = distance * np.sin(heading_error)
    zem_scale = max(float(np.mean(np.abs(zem_proxy))), 1e-12)
    angle_scale = max(float(np.mean(heading_error)), 1e-12)
    p_zem = np.exp(-0.5 * (zem_proxy / zem_scale) ** 2)
    p_angle = np.exp(-0.5 * (heading_error / angle_scale) ** 2)
    probability = np.clip(
        zem_weight * p_zem + (1.0 - zem_weight) * p_angle, 0.02, 0.98
    )
    return Problem(probability=probability, max_load=math.ceil(defenders / targets))


def objective(problem: Problem, assignment: np.ndarray) -> float:
    survival = np.ones(problem.targets, dtype=float)
    for defender, target in enumerate(assignment):
        survival[int(target)] *= 1.0 - problem.probability[defender, int(target)]
    return float(np.sum(survival))


def repair(problem: Problem, assignment: np.ndarray) -> np.ndarray:
    assignment = np.asarray(assignment, dtype=int).copy()
    assignment = np.clip(assignment, 0, problem.targets - 1)
    counts = np.bincount(assignment, minlength=problem.targets)

    def best_move(source: int | None, receivers: np.ndarray) -> tuple[int, int]:
        survival = np.ones(problem.targets, dtype=float)
        for defender, target in enumerate(assignment):
            survival[int(target)] *= 1.0 - problem.probability[defender, int(target)]
        defenders = (
            np.flatnonzero(assignment == source)
            if source is not None
            else np.flatnonzero(counts[assignment] > 1)
        )
        best_delta = np.inf
        best_pair = (-1, -1)
        for defender in defenders:
            old = int(assignment[defender])
            restored_old = survival[old] / (1.0 - problem.probability[defender, old])
            for target in receivers:
                target = int(target)
                if target == old:
                    continue
                changed_target = survival[target] * (
                    1.0 - problem.probability[defender, target]
                )
                delta = (
                    restored_old
                    + changed_target
                    - survival[old]
                    - survival[target]
                )
                if delta < best_delta:
                    best_delta = float(delta)
                    best_pair = (int(defender), target)
        if best_pair[0] < 0:
            raise RuntimeError("Unable to repair assignment")
        return best_pair

    # Fill empty targets first, choosing the move with the smallest objective increase.
    for empty in np.flatnonzero(counts == 0):
        defender, target = best_move(None, np.array([empty], dtype=int))
        old = int(assignment[defender])
        assignment[defender] = target
        counts[old] -= 1
        counts[empty] += 1

    # Enforce the upper capacity while preserving the nonempty constraint.
    while int(counts.max()) > problem.max_load:
        source = int(np.argmax(counts))
        receiver_candidates = np.flatnonzero(counts < problem.max_load)
        defender, target = best_move(source, receiver_candidates)
        assignment[defender] = target
        counts[source] -= 1
        counts[target] += 1
    return assignment


def initialize_population(problem: Problem, population: int, rng: np.random.Generator) -> np.ndarray:
    individuals = np.empty((population, problem.defenders), dtype=int)
    base = np.arange(problem.defenders) % problem.targets
    for idx in range(population):
        rng.shuffle(base)
        individuals[idx] = repair(problem, base)
    return individuals


def mutate_component(
    problem: Problem,
    current: np.ndarray,
    best: np.ndarray,
    second: np.ndarray,
    role: int,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    candidate = current.copy()
    n_change = max(1, int(round(strength * problem.defenders)))
    dims = rng.choice(problem.defenders, size=min(n_change, problem.defenders), replace=False)
    if role == 0:  # rolling: exploit current global best with occasional random steps
        copy_mask = rng.random(dims.size) < 0.75
        candidate[dims[copy_mask]] = best[dims[copy_mask]]
        candidate[dims[~copy_mask]] = rng.integers(0, problem.targets, np.sum(~copy_mask))
    elif role == 1:  # dancing: controlled random reorientation
        candidate[dims] = rng.integers(0, problem.targets, dims.size)
    elif role == 2:  # breeding: recombine the two elite assignments
        choose_best = rng.random(dims.size) < 0.5
        candidate[dims[choose_best]] = best[dims[choose_best]]
        candidate[dims[~choose_best]] = second[dims[~choose_best]]
    else:  # stealing: transfer entries from the best assignment
        candidate[dims] = best[dims]
    return repair(problem, candidate)


def idbo(
    problem: Problem,
    seed: int,
    iterations: int = 100,
    population: int = 40,
    adaptive: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    individuals = initialize_population(problem, population, rng)
    costs = np.array([objective(problem, x) for x in individuals])
    order = np.argsort(costs)
    best = individuals[order[0]].copy()
    best_cost = float(costs[order[0]])
    history = np.empty(iterations, dtype=float)

    for iteration in range(iterations):
        progress = iteration / max(iterations - 1, 1)
        # Decay exploration while retaining a small nonzero mutation probability.
        strength = 0.32 * (1.0 - progress) + 0.04 if adaptive else 0.18
        order = np.argsort(costs)
        second = individuals[order[min(1, population - 1)]].copy()
        for index in range(population):
            role = min(3, (4 * index) // population)
            candidate = mutate_component(
                problem, individuals[index], best, second, role, strength, rng
            )
            candidate_cost = objective(problem, candidate)
            if candidate_cost <= costs[index]:
                individuals[index] = candidate
                costs[index] = candidate_cost
                if candidate_cost < best_cost:
                    best_cost = float(candidate_cost)
                    best = candidate.copy()
        history[iteration] = best_cost
    return best_cost, best, history


def graph_neighbors(nodes: int) -> list[set[int]]:
    neighbors = [set() for _ in range(nodes)]
    jump = max(2, int(round(math.sqrt(nodes))))
    for node in range(nodes):
        for other in ((node - 1) % nodes, (node + 1) % nodes, (node + jump) % nodes):
            neighbors[node].add(other)
            neighbors[other].add(node)
    return neighbors


def source_propagation_rounds(
    problem: Problem,
    delay: int,
    dropout: float,
    seed: int,
    failed_node: int | None = None,
    max_rounds: int = 500,
) -> tuple[int, bool, int]:
    rng = np.random.default_rng(seed)
    active = [node for node in range(problem.defenders) if node != failed_node]
    neighbors = graph_neighbors(problem.defenders)
    # Each node initially knows only its own bid vector.
    known = {node: {node} for node in active}
    queue: dict[int, list[tuple[int, set[int]]]] = {}
    for round_index in range(max_rounds):
        for receiver, payload in queue.pop(round_index, []):
            if receiver in known:
                known[receiver].update(payload)
        if all(len(known[node]) == len(active) for node in active):
            return round_index, True, len(active)
        for sender in active:
            for receiver in neighbors[sender]:
                if receiver == failed_node or receiver not in known:
                    continue
                if rng.random() < dropout:
                    continue
                jitter = int(rng.integers(0, delay + 1)) if delay else 0
                delivery = round_index + 1 + jitter
                queue.setdefault(delivery, []).append((receiver, set(known[sender])))
    return max_rounds, False, len(active)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def run_study(output: Path, seeds: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    base_problem = make_problem(20, 8, seed=20260718, zem_weight=0.5)

    variants: dict[str, dict] = {}
    convergence: dict[str, list[np.ndarray]] = {"adaptive": [], "fixed": []}
    assignments: dict[str, list[np.ndarray]] = {"adaptive": [], "fixed": []}
    for label, adaptive in (("adaptive", True), ("fixed", False)):
        costs: list[float] = []
        for seed in range(seeds):
            value, assignment, history = idbo(
                base_problem, seed=1000 + seed, adaptive=adaptive
            )
            costs.append(value)
            assignments[label].append(assignment)
            convergence[label].append(history)
        variants[label] = summarize(costs)
        variants[label]["iterations_to_1pct_median"] = float(
            np.median(
                [
                    np.argmax(h <= h[-1] * 1.01) + 1
                    for h in convergence[label]
                ]
            )
        )

    weight_rows: list[dict] = []
    for weight in (0.3, 0.5, 0.7):
        costs = []
        for seed in range(seeds):
            problem = make_problem(20, 8, seed=20260718, zem_weight=weight)
            value, _, _ = idbo(problem, seed=3000 + seed, adaptive=True)
            costs.append(value)
        weight_rows.append({"w_zem": weight, "w_angle": 1.0 - weight, **summarize(costs)})

    scale_rows: list[dict] = []
    for defenders, targets in ((20, 8), (40, 16), (80, 32)):
        runtimes = []
        costs = []
        for seed in range(max(5, seeds // 3)):
            problem = make_problem(defenders, targets, seed=5000 + seed)
            start = time.perf_counter()
            value, _, _ = idbo(
                problem, seed=6000 + seed, iterations=60, population=30, adaptive=True
            )
            runtimes.append(time.perf_counter() - start)
            costs.append(value / targets)
        scale_rows.append(
            {
                "defenders": defenders,
                "targets": targets,
                "normalized_cost_mean": float(np.mean(costs)),
                "runtime_seconds_mean": float(np.mean(runtimes)),
                "runtime_seconds_std": float(np.std(runtimes, ddof=1)),
            }
        )

    propagation_rows: list[dict] = []
    for delay in (0, 1, 2, 4):
        for dropout in (0.0, 0.01):
            rounds = []
            success = []
            for seed in range(seeds):
                value, ok, _ = source_propagation_rounds(
                    base_problem, delay=delay, dropout=dropout, seed=7000 + seed
                )
                rounds.append(value)
                success.append(ok)
            propagation_rows.append(
                {
                    "max_delay_rounds": delay,
                    "dropout_probability": dropout,
                    "success_rate": float(np.mean(success)),
                    "rounds_mean": float(np.mean(rounds)),
                    "rounds_std": float(np.std(rounds, ddof=1)),
                    "rounds_max": int(np.max(rounds)),
                }
            )

    failure_rounds = []
    failure_success = []
    for seed in range(seeds):
        value, ok, active = source_propagation_rounds(
            base_problem,
            delay=2,
            dropout=0.01,
            seed=9000 + seed,
            failed_node=seed % base_problem.defenders,
        )
        failure_rounds.append(value)
        failure_success.append(ok)
    failure = {
        "condition": "one_of_20_source_nodes_removed",
        "active_nodes": active,
        "max_delay_rounds": 2,
        "dropout_probability": 0.01,
        "success_rate": float(np.mean(failure_success)),
        "rounds_mean": float(np.mean(failure_rounds)),
        "rounds_std": float(np.std(failure_rounds, ddof=1)),
        "rounds_max": int(np.max(failure_rounds)),
    }

    best_index = int(np.argmin([objective(base_problem, x) for x in assignments["adaptive"]]))
    best_assignment = assignments["adaptive"][best_index]
    best_counts = np.bincount(best_assignment, minlength=base_problem.targets)

    result = {
        "seed_count": seeds,
        "problem": {
            "defenders": base_problem.defenders,
            "targets": base_problem.targets,
            "max_load": base_problem.max_load,
            "objective": "sum_of_target_survival_probabilities",
        },
        "coefficient_ablation": variants,
        "weight_sensitivity": weight_rows,
        "scalability": scale_rows,
        "source_propagation_delay": propagation_rows,
        "source_node_removal": failure,
        "representative_assignment_1_based": (best_assignment + 1).tolist(),
        "representative_target_loads": best_counts.tolist(),
    }
    (output / "idbo_reviewer_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    with (output / "idbo_reviewer_tables.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "condition", "metric", "value"])
        for label, stats in variants.items():
            for key, value in stats.items():
                writer.writerow(["coefficient_ablation", label, key, value])
        for row in weight_rows:
            condition = f"w_zem={row['w_zem']:.1f}"
            for key, value in row.items():
                if key not in {"w_zem", "w_angle"}:
                    writer.writerow(["weight_sensitivity", condition, key, value])
        for row in scale_rows:
            condition = f"{row['defenders']}x{row['targets']}"
            for key, value in row.items():
                if key not in {"defenders", "targets"}:
                    writer.writerow(["scalability", condition, key, value])
        for row in propagation_rows:
            condition = (
                f"delay={row['max_delay_rounds']},dropout={row['dropout_probability']}"
            )
            for key, value in row.items():
                if key not in {"max_delay_rounds", "dropout_probability"}:
                    writer.writerow(["source_propagation_delay", condition, key, value])

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
    for label, color in (("adaptive", "#176B87"), ("fixed", "#B24C3D")):
        array = np.asarray(convergence[label])
        mean = array.mean(axis=0)
        std = array.std(axis=0, ddof=1)
        iterations = np.arange(1, mean.size + 1)
        axes[0].plot(iterations, mean, label=label.capitalize(), color=color, linewidth=1.5)
        axes[0].fill_between(iterations, mean - std, mean + std, color=color, alpha=0.18)
    axes[0].set_xlabel("IDBO iteration")
    axes[0].set_ylabel("Best survival cost")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    labels = [f"{row['max_delay_rounds']}" for row in propagation_rows if row["dropout_probability"] == 0.01]
    means = [row["rounds_mean"] for row in propagation_rows if row["dropout_probability"] == 0.01]
    errors = [row["rounds_std"] for row in propagation_rows if row["dropout_probability"] == 0.01]
    axes[1].errorbar(
        labels, means, yerr=errors, marker="o", capsize=3, color="#3C6E47", linewidth=1.5
    )
    axes[1].set_xlabel("Maximum link delay (rounds)")
    axes[1].set_ylabel("Source-propagation rounds")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "idbo_ablation_delay.pdf", bbox_inches="tight")
    fig.savefig(output / "idbo_ablation_delay.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()
    result = run_study(args.output, args.seeds)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
