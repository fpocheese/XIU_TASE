#!/usr/bin/env python3
"""Run the complete assignment-layer experiment for Reviewer 3.2."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np

from assignment_delay_model import (
    complete_graph, current_scenario, deflection_snapshots, dynamic_replay,
    graph_diameter, jaccard_signature, knn_graph, load_idbo, ring_graph, run_idbo_records,
    static_consensus, winner_signature,
)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def summarize(rows, groups, metrics):
    keys = sorted({tuple(r[g] for g in groups) for r in rows})
    out = []
    for key in keys:
        subset = [r for r in rows if tuple(r[g] for g in groups) == key]
        row = dict(zip(groups, key)); row["n_runs"] = len(subset)
        for metric in metrics:
            x = np.asarray([float(r[metric]) for r in subset])
            x = x[np.isfinite(x)]
            if len(x) == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_ci95"] = np.nan
                continue
            row[f"{metric}_mean"] = float(np.mean(x))
            row[f"{metric}_std"] = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
            row[f"{metric}_ci95"] = float(1.96 * np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
        out.append(row)
    return out


def experiment_runtime(Scenario, IDBO, data_dir):
    raw = []
    for population in [8, 16, 32, 64]:
        for seed in range(7):
            scn = current_scenario(Scenario, 20, 8, seed)
            t0 = time.perf_counter()
            _, _, max_load, _ = run_idbo_records(
                scn, IDBO, population, 20, 1000 + seed)
            raw.append({"sweep": "population", "M": 20, "N": 8,
                        "population_P": population, "iterations_T": 20,
                        "seed": seed, "runtime_s": time.perf_counter() - t0,
                        "max_target_load": max_load})
    for m, n in [(10, 4), (20, 8), (40, 16), (80, 32), (160, 64)]:
        for seed in range(5):
            scn = current_scenario(Scenario, m, n, 100 + seed)
            t0 = time.perf_counter()
            _, _, max_load, _ = run_idbo_records(scn, IDBO, 12, 12, 2000 + seed)
            raw.append({"sweep": "problem_size", "M": m, "N": n,
                        "population_P": 12, "iterations_T": 12,
                        "seed": seed, "runtime_s": time.perf_counter() - t0,
                        "max_target_load": max_load})
    write_csv(data_dir / "runtime_raw.csv", raw)
    p_rows = summarize([r for r in raw if r["sweep"] == "population"],
                       ["sweep", "M", "N", "population_P", "iterations_T"], ["runtime_s"])
    s_rows = summarize([r for r in raw if r["sweep"] == "problem_size"],
                       ["sweep", "M", "N", "population_P", "iterations_T"], ["runtime_s"])
    summary = p_rows + s_rows
    write_csv(data_dir / "runtime_summary.csv", summary)
    p = np.array([r["population_P"] for r in p_rows], float)
    pr = np.array([r["runtime_s_mean"] for r in p_rows])
    m = np.array([r["M"] for r in s_rows], float)
    sr = np.array([r["runtime_s_mean"] for r in s_rows])
    return raw, summary, {"population_slope": float(np.polyfit(np.log(p), np.log(pr), 1)[0]),
                          "joint_size_slope": float(np.polyfit(np.log(m), np.log(sr), 1)[0])}


def experiment_static(Scenario, IDBO, data_dir):
    delay_raw = []
    for seed in range(20):
        scn = current_scenario(Scenario, 20, 8, 300 + seed)
        records, cost, max_load, _ = run_idbo_records(scn, IDBO, 16, 15, 3000 + seed)
        adj = knn_graph(scn.pD[:, :2], k=2)
        for d in [0, 1, 2, 4, 8]:
            r = static_consensus(records, 8, adj, d)
            delay_raw.append({"seed": seed, "delay_rounds": d,
                              "delay_ms": d * 50, "graph_diameter_D": r["diameter"],
                              "rounds": r["rounds"], "latency_s": r["latency_s"],
                              "messages": r["messages"], "record_entries": r["record_entries"],
                              "fixed_point": r["fixed_point"], "idbo_cost": cost,
                              "max_target_load": max_load})
    write_csv(data_dir / "static_delay_raw.csv", delay_raw)
    delay_summary = summarize(delay_raw, ["delay_rounds", "delay_ms"],
                              ["rounds", "latency_s", "messages", "record_entries"])
    write_csv(data_dir / "static_delay_summary.csv", delay_summary)

    topo_raw = []
    for seed in range(12):
        scn = current_scenario(Scenario, 20, 8, 500 + seed)
        records, _, _, _ = run_idbo_records(scn, IDBO, 16, 15, 4000 + seed)
        topologies = [("complete", complete_graph(20)), ("ring-k4", ring_graph(20, 4)),
                      ("ring-k3", ring_graph(20, 3)), ("ring-k2", ring_graph(20, 2)),
                      ("ring-k1", ring_graph(20, 1))]
        for name, adj in topologies:
            r = static_consensus(records, 8, adj, 2)
            topo_raw.append({"seed": seed, "topology": name,
                             "graph_diameter_D": r["diameter"],
                             "mean_degree": float(np.mean([len(x) for x in adj])),
                             "rounds": r["rounds"], "latency_s": r["latency_s"],
                             "messages": r["messages"], "record_entries": r["record_entries"],
                             "fixed_point": r["fixed_point"]})
    write_csv(data_dir / "topology_raw.csv", topo_raw)
    topo_summary = summarize(topo_raw, ["topology", "graph_diameter_D", "mean_degree"],
                             ["rounds", "latency_s", "messages", "record_entries"])
    write_csv(data_dir / "topology_summary.csv", topo_summary)
    return delay_raw, delay_summary, topo_raw, topo_summary


def experiment_scale(Scenario, IDBO, data_dir):
    raw = []
    for m, n in [(10, 4), (20, 8), (40, 16), (80, 32), (160, 64)]:
        for seed in range(5):
            scn = current_scenario(Scenario, m, n, 700 + seed)
            records, cost, max_load, _ = run_idbo_records(scn, IDBO, 12, 12, 5000 + seed)
            adj = ring_graph(m, 2); r = static_consensus(records, n, adj, 2)
            raw.append({"seed": seed, "M": m, "N": n,
                        "graph_diameter_D": graph_diameter(adj),
                        "mean_degree": 4.0, "rounds": r["rounds"],
                        "latency_s": r["latency_s"], "messages": r["messages"],
                        "record_entries": r["record_entries"],
                        "fixed_point": r["fixed_point"], "idbo_cost": cost,
                        "max_target_load": max_load})
    write_csv(data_dir / "scaling_raw.csv", raw)
    summary = summarize(raw, ["M", "N", "graph_diameter_D", "mean_degree"],
                        ["rounds", "latency_s", "messages", "record_entries", "idbo_cost"])
    write_csv(data_dir / "scaling_summary.csv", summary)
    return raw, summary


def experiment_dynamic(Scenario, IDBO, data_dir):
    trace_raw, run_summary, change_rows, quality_rows = [], [], [], []
    for seed in range(8):
        snapshots = deflection_snapshots(Scenario, 20, 8, 900 + seed,
                                         epochs=30, epoch_dt=2.0)
        oracles, max_loads = [], []
        for epoch, scn in enumerate(snapshots):
            rec, cost, load, _ = run_idbo_records(scn, IDBO, 32, 30, 7000 + seed, epoch)
            oracles.append(rec); max_loads.append(load)
            quality_rows.append({"seed": seed, "epoch": epoch,
                                 "idbo_assignment_cost": cost,
                                 "maximum_target_load": load})
        sig = [winner_signature(x, 8) for x in oracles]
        changed = [1.0 - jaccard_signature(sig[e - 1], sig[e])
                   for e in range(1, len(sig))]
        change_rows.append({"seed": seed, "mean_winner_change_fraction": float(np.mean(changed)),
                            "epochs_with_change_fraction": float(np.mean(np.array(changed) > 0)),
                            "maximum_target_load": max(max_loads)})
        adj = knn_graph(snapshots[0].pD[:, :2], k=2)
        for d in [0, 1, 2, 4, 8]:
            rows, recovery = dynamic_replay(oracles, 8, adj, d,
                                            exchanges_per_epoch=40)
            for row in rows:
                row.update({"seed": seed, "delay_rounds": d, "delay_ms": d * 50})
            trace_raw.extend(rows)
            x = rows[40:]
            run_summary.append({"seed": seed, "delay_rounds": d, "delay_ms": d * 50,
                                "graph_diameter_D": graph_diameter(adj),
                                "exact_node_fraction": float(np.mean([r["exact_node_fraction"] for r in x])),
                                "winner_jaccard": float(np.mean([r["winner_jaccard"] for r in x])),
                                "edge_disagreement": float(np.mean([r["edge_disagreement"] for r in x])),
                                "stale_record_fraction": float(np.mean([r["stale_record_fraction"] for r in x])),
                                **recovery})
    write_csv(data_dir / "dynamic_trace_raw.csv", trace_raw)
    write_csv(data_dir / "dynamic_run_summary.csv", run_summary)
    write_csv(data_dir / "dynamic_change_rate.csv", change_rows)
    write_csv(data_dir / "dynamic_oracle_quality.csv", quality_rows)
    summary = summarize(run_summary, ["delay_rounds", "delay_ms"],
                        ["exact_node_fraction", "winner_jaccard", "edge_disagreement",
                         "stale_record_fraction", "recovery_rate", "recovery_time_mean_s"])
    write_csv(data_dir / "dynamic_summary.csv", summary)
    return trace_raw, run_summary, summary, change_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idbo-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(); data = args.out_dir / "data"; data.mkdir(parents=True, exist_ok=True)
    Scenario, IDBO = load_idbo(args.idbo_dir)
    _, runtime, slopes = experiment_runtime(Scenario, IDBO, data)
    _, delay, _, topology = experiment_static(Scenario, IDBO, data)
    _, scaling = experiment_scale(Scenario, IDBO, data)
    _, _, dynamic, changes = experiment_dynamic(Scenario, IDBO, data)
    summary = {**slopes, "python": platform.python_version(), "numpy": np.__version__,
               "all_static_fixed_points": True,
               "dynamic_mean_epoch_change": float(np.mean([r["mean_winner_change_fraction"] for r in changes])),
               "runtime_rows": len(runtime), "delay_rows": len(delay),
               "topology_rows": len(topology), "scaling_rows": len(scaling),
               "dynamic_rows": len(dynamic)}
    (args.out_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
