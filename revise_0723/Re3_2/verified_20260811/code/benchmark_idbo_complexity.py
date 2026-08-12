#!/usr/bin/env python3
"""Verified IDBO complexity and delayed-consensus benchmark for Reviewer 3.2.

The benchmark deliberately separates two questions:

1. Local optimizer runtime is measured by executing the user's current
   ``idbo_paper.py`` implementation on scenarios initialized from the current
   manuscript (defenders in a 100-m disk, attackers in a 1500--1600-m annulus).
2. Communication delay is evaluated with an isolated, deterministic Top-L
   winner-set propagation benchmark corresponding to Eq. (cbba_update).  This
   isolates consensus efficiency from changing local bids.  The merge is
   associative, commutative, and idempotent, so its static-snapshot fixed point
   is known and can be verified exactly after every run.

No interception trajectories, policy weights, or manuscript files are changed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_idbo(code_dir: Path):
    sys.path.insert(0, str(code_dir))
    from scenario_paper import Scenario  # type: ignore
    from idbo_paper import IDBO_paper  # type: ignore
    return Scenario, IDBO_paper


def current_scenario(Scenario, m: int, n: int, seed: int):
    """Instantiate the actual Scenario class, then apply current Table-II ranges."""
    scn = Scenario(n_def=m, n_att=n, L_max=3, seed=seed)
    rng = np.random.default_rng(seed)

    r_d = rng.uniform(0.0, 100.0, m)
    phi_d = rng.uniform(0.0, 2.0 * np.pi, m)
    scn.pD = np.column_stack((r_d * np.cos(phi_d), r_d * np.sin(phi_d), np.zeros(m)))
    speed_d = rng.uniform(10.0, 40.0, m)
    gamma_d = rng.uniform(0.0, 2.0 * np.pi, m)
    scn.vD = np.column_stack((speed_d * np.cos(gamma_d),
                              speed_d * np.sin(gamma_d), np.zeros(m)))

    r_a = rng.uniform(1500.0, 1600.0, n)
    phi_a = rng.uniform(0.0, 2.0 * np.pi, n)
    scn.pT = np.column_stack((r_a * np.cos(phi_a), r_a * np.sin(phi_a),
                              np.full(n, 120.0)))
    speed_a = rng.uniform(10.0, 40.0, n)
    q_to_center = np.arctan2(-scn.pT[:, 1], -scn.pT[:, 0])
    gamma_a = q_to_center + rng.uniform(-np.pi / 6.0, np.pi / 6.0, n)
    scn.vT = np.column_stack((speed_a * np.cos(gamma_a),
                              speed_a * np.sin(gamma_a), np.zeros(n)))
    scn._precompute()
    return scn


def ring_graph(m: int, k: int):
    adj = [set() for _ in range(m)]
    for i in range(m):
        for h in range(1, k + 1):
            for j in ((i + h) % m, (i - h) % m):
                adj[i].add(j)
                adj[j].add(i)
    return [sorted(x) for x in adj]


def complete_graph(m: int):
    return [[j for j in range(m) if j != i] for i in range(m)]


def diameter(adj):
    out = 0
    for s in range(len(adj)):
        dist = [-1] * len(adj)
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    q.append(v)
        if min(dist) < 0:
            raise RuntimeError("communication graph is disconnected")
        out = max(out, max(dist))
    return out


def top_l(entries, limit):
    best = {}
    for agent, bid in entries:
        best[agent] = max(float(bid), best.get(agent, -np.inf))
    return tuple(sorted(sorted(best.items(), key=lambda x: (-x[1], x[0]))[:limit]))


def make_local_state(scn, limit=3):
    """Each defender bids for its best target using the manuscript score ingredients."""
    score = scn.p_int + scn.lam_A * scn.chi_static
    state = []
    for i in range(scn.N_D):
        target = int(np.argmax(score[i]))
        per_target = [tuple() for _ in range(scn.N_A)]
        per_target[target] = ((i, float(score[i, target])),)
        state.append(tuple(per_target))
    return state


def merge_state(a, b, limit):
    return tuple(top_l(list(a[j]) + list(b[j]), limit) for j in range(len(a)))


def global_fixed_point(local_states, limit):
    merged = tuple(tuple() for _ in range(len(local_states[0])))
    for state in local_states:
        merged = merge_state(merged, state, limit)
    return merged


def consensus_run(local_states, adj, delay_rounds, limit=3, patience=3,
                  max_rounds=1000):
    """Delayed synchronous propagation; one hop takes delay_rounds+1 exchanges."""
    m = len(adj)
    states = list(local_states)
    queues = {(i, j): [] for i in range(m) for j in adj[i]}
    target = global_fixed_point(local_states, limit)
    packets = 0
    record_entries = 0
    stable = 0

    for r in range(max_rounds):
        # Receive first, update the local replica, and only then transmit the
        # updated state.  Thus delay_rounds=0 gives one-hop propagation per
        # synchronous exchange, while delay_rounds=d gives one hop per d+1 exchanges.
        incoming = [[] for _ in range(m)]
        for (i, j), q in queues.items():
            keep = []
            for due, payload in q:
                if due == r:
                    incoming[j].append(payload)
                else:
                    keep.append((due, payload))
            queues[(i, j)] = keep

        new_states = []
        for i in range(m):
            merged = states[i]
            for payload in incoming[i]:
                merged = merge_state(merged, payload, limit)
            new_states.append(merged)
        states = new_states

        edge_disagreement = sum(states[i] != states[j]
                                for i in range(m) for j in adj[i] if i < j)
        all_equal_target = all(state == target for state in states)
        stable = stable + 1 if edge_disagreement == 0 and all_equal_target else 0
        if stable >= patience:
            return {
                "rounds": r + 1,
                "diameter": diameter(adj),
                "packets": packets,
                "record_entries": record_entries,
                "fixed_point_verified": True,
            }

        arrival = r + delay_rounds + 1
        for i in range(m):
            payload = states[i]
            entries = sum(len(x) for x in payload)
            for j in adj[i]:
                queues[(i, j)].append((arrival, payload))
                packets += 1
                record_entries += entries
    raise RuntimeError("consensus did not converge within max_rounds")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def benchmark_local(Scenario, IDBO_paper, out_dir):
    rows = []
    for population in [8, 16, 32, 64]:
        samples = []
        for seed in range(5):
            scn = current_scenario(Scenario, 20, 8, seed)
            t0 = time.perf_counter()
            IDBO_paper(N=population, max_iter=20, scn=scn,
                       schedule="linear", seed=100 + seed)
            samples.append(time.perf_counter() - t0)
        rows.append({
            "defenders_M": 20,
            "targets_N": 8,
            "population_P": population,
            "iterations_T": 20,
            "runtime_median_s": float(np.median(samples)),
            "runtime_q1_s": float(np.quantile(samples, 0.25)),
            "runtime_q3_s": float(np.quantile(samples, 0.75)),
        })
    write_csv(out_dir / "data" / "local_idbo_runtime.csv", rows)
    return rows


def benchmark_consensus(Scenario, out_dir):
    delay_rows = []
    for d in [0, 1, 2, 4, 8]:
        values = []
        for seed in range(10):
            scn = current_scenario(Scenario, 20, 8, seed)
            values.append(consensus_run(make_local_state(scn), ring_graph(20, 2), d))
        delay_rows.append({
            "delay_rounds": d,
            "graph_diameter_D": values[0]["diameter"],
            "rounds_mean": float(np.mean([v["rounds"] for v in values])),
            "rounds_std": float(np.std([v["rounds"] for v in values])),
            "all_fixed_points_verified": all(v["fixed_point_verified"] for v in values),
        })

    topology_rows = []
    for label, adj in [("complete", complete_graph(20)),
                       ("ring-k4", ring_graph(20, 4)),
                       ("ring-k3", ring_graph(20, 3)),
                       ("ring-k2", ring_graph(20, 2)),
                       ("ring-k1", ring_graph(20, 1))]:
        scn = current_scenario(Scenario, 20, 8, 17)
        value = consensus_run(make_local_state(scn), adj, delay_rounds=1)
        topology_rows.append({
            "topology": label,
            "graph_diameter_D": value["diameter"],
            "mean_degree": float(np.mean([len(x) for x in adj])),
            "consensus_rounds": value["rounds"],
            "fixed_point_verified": value["fixed_point_verified"],
        })

    scale_rows = []
    for m in [10, 20, 40, 80, 160]:
        scn = current_scenario(Scenario, m, 8, 23)
        adj = ring_graph(m, 2)
        value = consensus_run(make_local_state(scn), adj, delay_rounds=1)
        scale_rows.append({
            "defenders_M": m,
            "targets_N": 8,
            "mean_degree": float(np.mean([len(x) for x in adj])),
            "graph_diameter_D": value["diameter"],
            "consensus_rounds": value["rounds"],
            "packets_total": value["packets"],
            "directed_packets_per_exchange": sum(len(x) for x in adj),
            "fixed_point_verified": value["fixed_point_verified"],
        })

    write_csv(out_dir / "data" / "consensus_delay.csv", delay_rows)
    write_csv(out_dir / "data" / "consensus_topology.csv", topology_rows)
    write_csv(out_dir / "data" / "consensus_scaling.csv", scale_rows)
    return delay_rows, topology_rows, scale_rows


def plot_results(local_rows, delay_rows, topology_rows, scale_rows, out_dir):
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                         "axes.linewidth": 0.8})
    fig, ax = plt.subplots(2, 2, figsize=(7.1, 5.0), constrained_layout=True)

    population = np.array([r["population_P"] for r in local_rows], float)
    rt = np.array([r["runtime_median_s"] for r in local_rows], float)
    exponent = float(np.polyfit(np.log(population), np.log(rt), 1)[0])
    ax[0, 0].plot(population, rt, "o-", color="#0072B2",
                  label=f"log--log slope={exponent:.2f}")
    ax[0, 0].set(xlabel="Local population size $P$", ylabel="Local IDBO runtime (s)")
    ax[0, 0].legend(frameon=False)

    ms = np.array([r["defenders_M"] for r in scale_rows])
    packets = np.array([r["directed_packets_per_exchange"] for r in scale_rows])
    ax[0, 1].plot(ms, packets, "s-", color="#009E73")
    ax[0, 1].set(xlabel="Number of defenders $M$", ylabel="Packets per consensus round")

    d = np.array([r["delay_rounds"] for r in delay_rows])
    rounds = np.array([r["rounds_mean"] for r in delay_rows])
    ax[1, 0].plot(d, rounds, "^-", color="#D55E00")
    ax[1, 0].set(xlabel="Per-hop delay (exchange rounds)", ylabel="Rounds to consensus")

    topo = sorted(topology_rows, key=lambda r: r["graph_diameter_D"])
    diam = np.array([r["graph_diameter_D"] for r in topo])
    tr = np.array([r["consensus_rounds"] for r in topo])
    ax[1, 1].plot(diam, tr, "D-", color="#CC79A7")
    ax[1, 1].set(xlabel="Communication-graph diameter $D$ (hops)",
                 ylabel="Rounds to consensus")

    for label, a in zip(["(a)", "(b)", "(c)", "(d)"], ax.flat):
        a.grid(True, alpha=0.22, linewidth=0.5)
        a.text(0.02, 0.96, label, transform=a.transAxes, va="top", fontweight="bold")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "idbo_complexity_delay_verified.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "idbo_complexity_delay_verified.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    return exponent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idbo-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    Scenario, IDBO_paper = load_idbo(args.idbo_dir)

    local_rows = benchmark_local(Scenario, IDBO_paper, args.out_dir)
    delay_rows, topology_rows, scale_rows = benchmark_consensus(Scenario, args.out_dir)
    exponent = plot_results(local_rows, delay_rows, topology_rows, scale_rows, args.out_dir)
    summary = {
        "local_runtime_loglog_exponent_vs_P": exponent,
        "delay_fixed_points_all_verified": all(r["all_fixed_points_verified"] for r in delay_rows),
        "topology_fixed_points_all_verified": all(r["fixed_point_verified"] for r in topology_rows),
        "scaling_fixed_points_all_verified": all(r["fixed_point_verified"] for r in scale_rows),
        "scenario": "current manuscript Table II ranges",
        "local_runtime_code": str(args.idbo_dir / "idbo_paper.py"),
        "consensus_model": "isolated delayed Top-L winner-set propagation",
    }
    with (args.out_dir / "benchmark_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
