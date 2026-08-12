#!/usr/bin/env python3
"""Assignment-only IDBO and delayed distributed winner-record model."""
from __future__ import annotations

import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def load_idbo(code_dir: Path):
    sys.path.insert(0, str(code_dir))
    from scenario_paper import Scenario  # type: ignore
    from idbo_paper import IDBO_paper  # type: ignore
    return Scenario, IDBO_paper


def current_scenario(Scenario, m: int, n: int, seed: int):
    """Generate a 3-D assignment snapshot using the active manuscript ranges."""
    scn = Scenario(n_def=m, n_att=n, L_max=3, seed=seed)
    rng = np.random.default_rng(seed)
    rd, pd = rng.uniform(0, 100, m), rng.uniform(0, 2 * np.pi, m)
    scn.pD = np.c_[rd * np.cos(pd), rd * np.sin(pd), np.zeros(m)]
    vd, gd = rng.uniform(10, 40, m), rng.uniform(0, 2 * np.pi, m)
    scn.vD = np.c_[vd * np.cos(gd), vd * np.sin(gd), np.zeros(m)]
    ra, pa = rng.uniform(1500, 1600, n), rng.uniform(0, 2 * np.pi, n)
    scn.pT = np.c_[ra * np.cos(pa), ra * np.sin(pa), np.full(n, 120.0)]
    va = rng.uniform(10, 40, n)
    qa = np.arctan2(-scn.pT[:, 1], -scn.pT[:, 0])
    ga = qa + rng.uniform(-np.pi / 6, np.pi / 6, n)
    scn.vT = np.c_[va * np.cos(ga), va * np.sin(ga), np.zeros(n)]
    scn._precompute()
    return scn


def snapshot_from_arrays(Scenario, p_d, v_d, p_t, v_t, seed=0):
    scn = Scenario(n_def=len(p_d), n_att=len(p_t), L_max=3, seed=seed)
    scn.pD, scn.vD = np.array(p_d, copy=True), np.array(v_d, copy=True)
    scn.pT, scn.vT = np.array(p_t, copy=True), np.array(v_t, copy=True)
    scn._precompute()
    return scn


def deflection_snapshots(Scenario, m: int, n: int, seed: int,
                         epochs=30, epoch_dt=0.5):
    """Moving 3-D snapshots with coordinated alternating lateral deflection."""
    scn = current_scenario(Scenario, m, n, seed)
    p_d, v_d = scn.pD.copy(), scn.vD.copy()
    p_t, v_t = scn.pT.copy(), scn.vT.copy()
    speed_t = np.linalg.norm(v_t, axis=1)
    group = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    out = []
    for e in range(epochs):
        out.append(snapshot_from_arrays(Scenario, p_d, v_d, p_t, v_t, seed + e))
        t = e * epoch_dt
        direction = 1.0 if int(t / 2.5) % 2 == 0 else -1.0
        q_center = np.arctan2(-p_t[:, 1], -p_t[:, 0])
        heading = q_center + 0.42 * direction * group
        v_t = np.c_[speed_t * np.cos(heading), speed_t * np.sin(heading),
                    2.0 * direction * group]
        p_d = p_d + v_d * epoch_dt
        p_t = p_t + v_t * epoch_dt
    return out


def ring_graph(m: int, k: int):
    adj = [set() for _ in range(m)]
    for i in range(m):
        for h in range(1, k + 1):
            for j in ((i - h) % m, (i + h) % m):
                adj[i].add(j); adj[j].add(i)
    return [sorted(x) for x in adj]


def complete_graph(m: int):
    return [[j for j in range(m) if j != i] for i in range(m)]


def knn_graph(xy, k=2):
    m = len(xy); adj = [set() for _ in range(m)]
    dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    for i in range(m):
        for j in np.argsort(dist[i])[:k]:
            adj[i].add(int(j)); adj[int(j)].add(i)
    # connect any residual components by their closest cross-component pair
    while True:
        comps = components(adj)
        if len(comps) == 1:
            break
        a, b = comps[0], set().union(*comps[1:])
        i, j = min(((i, j) for i in a for j in b), key=lambda z: dist[z])
        adj[i].add(j); adj[j].add(i)
    return [sorted(x) for x in adj]


def components(adj):
    unseen, out = set(range(len(adj))), []
    while unseen:
        root = unseen.pop(); comp = {root}; q = [root]
        while q:
            u = q.pop()
            for v in adj[u]:
                if v in unseen:
                    unseen.remove(v); comp.add(v); q.append(v)
        out.append(comp)
    return out


def graph_diameter(adj):
    best = 0
    for source in range(len(adj)):
        d = [-1] * len(adj); d[source] = 0; q = deque([source])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if d[v] < 0:
                    d[v] = d[u] + 1; q.append(v)
        if min(d) < 0:
            raise ValueError("disconnected graph")
        best = max(best, max(d))
    return best


@dataclass(frozen=True)
class Record:
    origin: int
    target: int
    bid: float
    timestamp: int


def run_idbo_records(scn, IDBO_paper, population=16, iterations=15,
                     seed=0, timestamp=0):
    cost, assignment, _ = IDBO_paper(N=population, max_iter=iterations,
                                     scn=scn, schedule="linear", seed=seed)
    score = scn.p_int + scn.lam_A * scn.chi_static
    records = [Record(i, int(j), float(score[i, int(j)]), timestamp)
               for i, j in enumerate(assignment)]
    load = np.bincount(assignment, minlength=scn.N_A)
    return records, float(cost), int(load.max()), score


def merge_db(db, payload):
    out = list(db)
    for rec in payload:
        if rec is None:
            continue
        old = out[rec.origin]
        if old is None or rec.timestamp > old.timestamp or (
                rec.timestamp == old.timestamp and rec.bid > old.bid):
            out[rec.origin] = rec
    return out


def winner_signature(db, n_targets, limit=3):
    groups = [[] for _ in range(n_targets)]
    for rec in db:
        if rec is not None:
            groups[rec.target].append(rec)
    return tuple(tuple(r.origin for r in sorted(g, key=lambda x: (-x.bid, x.origin))[:limit])
                 for g in groups)


def jaccard_signature(a, b):
    sa = {(j, i) for j, group in enumerate(a) for i in group}
    sb = {(j, i) for j, group in enumerate(b) for i in group}
    return 1.0 if not (sa | sb) else len(sa & sb) / len(sa | sb)


def static_consensus(records, n_targets, adj, delay_rounds,
                     link_period=0.05, patience=2, max_rounds=2000):
    m = len(records); dbs = [[None] * m for _ in range(m)]
    for i, rec in enumerate(records):
        dbs[i][i] = rec
    oracle = winner_signature(records, n_targets)
    schedule = defaultdict(list); messages = entries = stable = 0
    for k in range(max_rounds):
        for receiver, payload in schedule.pop(k, []):
            dbs[receiver] = merge_db(dbs[receiver], payload)
        sig = [winner_signature(db, n_targets) for db in dbs]
        ok = all(x == oracle for x in sig)
        stable = stable + 1 if ok else 0
        if stable >= patience:
            return {"rounds": k + 1, "latency_s": (k + 1) * link_period,
                    "messages": messages, "record_entries": entries,
                    "diameter": graph_diameter(adj), "fixed_point": True}
        due = k + delay_rounds + 1
        for i in range(m):
            payload = tuple(dbs[i]); count = sum(x is not None for x in payload)
            for j in adj[i]:
                schedule[due].append((j, payload)); messages += 1; entries += count
    return {"rounds": max_rounds, "latency_s": max_rounds * link_period,
            "messages": messages, "record_entries": entries,
            "diameter": graph_diameter(adj), "fixed_point": False}


def dynamic_replay(oracles, n_targets, adj, delay_rounds, exchanges_per_epoch=10,
                   link_period=0.05):
    """Replay time-varying IDBO records through delayed neighbor gossip."""
    m = len(oracles[0]); dbs = [list(oracles[0]) for _ in range(m)]
    schedule = defaultdict(list); rows = []; recovery = {}
    total_steps = len(oracles) * exchanges_per_epoch
    for k in range(total_steps):
        epoch = min(k // exchanges_per_epoch, len(oracles) - 1)
        if k % exchanges_per_epoch == 0 and epoch > 0:
            for i in range(m):
                dbs[i][i] = oracles[epoch][i]
        for receiver, payload in schedule.pop(k, []):
            dbs[receiver] = merge_db(dbs[receiver], payload)
        oracle = winner_signature(oracles[epoch], n_targets)
        sigs = [winner_signature(db, n_targets) for db in dbs]
        exact = np.mean([s == oracle for s in sigs])
        sim = np.mean([jaccard_signature(s, oracle) for s in sigs])
        edges = [(i, j) for i in range(m) for j in adj[i] if i < j]
        gamma = np.mean([sigs[i] != sigs[j] for i, j in edges]) if edges else 0.0
        stale = np.mean([dbs[node][origin].timestamp < epoch
                         for node in range(m) for origin in range(m)])
        if exact >= 0.95 and epoch not in recovery:
            recovery[epoch] = (k % exchanges_per_epoch) * link_period
        rows.append({"exchange": k, "epoch": epoch, "time_s": k * link_period,
                     "exact_node_fraction": exact, "winner_jaccard": sim,
                     "edge_disagreement": gamma, "stale_record_fraction": stale})
        due = k + delay_rounds + 1
        for i in range(m):
            payload = tuple(dbs[i])
            for j in adj[i]:
                schedule[due].append((j, payload))
    oracle_signatures = [winner_signature(x, n_targets) for x in oracles]
    valid_epochs = [e for e in range(1, len(oracles))
                    if oracle_signatures[e] != oracle_signatures[e - 1]]
    recovered = [e in recovery for e in valid_epochs]
    rec_times = [recovery[e] for e in valid_epochs if e in recovery]
    return rows, {"recovery_rate": float(np.mean(recovered)) if recovered else 1.0,
                  "recovery_time_mean_s": float(np.mean(rec_times)) if rec_times else np.nan,
                  "recovery_time_p95_s": float(np.quantile(rec_times, .95)) if rec_times else np.nan}
