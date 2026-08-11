"""
run_experiments.py -- Complexity, scalability, and communication-delay experiments for
the distributed consensus-based IDBO (reviewer response Re3_2).
=======================================================================================
Produces three raw-data files consumed by the plotting scripts:

  A. COMPLEXITY   : per-round wall-clock and communication volume vs swarm size N_D,
                    to validate the O(N_pop N_A) + O(N_A |N_i|) per-defender scaling.
  B. SCALABILITY  : rounds-to-consensus and solution quality as (N_D, N_A) grow together.
  C. COMM-DELAY   : rounds-to-consensus and Gamma^(k) trajectories vs communication delay
                    and graph diameter -- the core "delay influences consensus efficiency".

All runs use the manuscript's engagement model (scenario_paper.Scenario) with the paper's
initialization ranges, scaled to the requested swarm size.
"""
import numpy as np
import csv
import os
import time
from scenario_paper import Scenario
from consensus_idbo import DistributedIDBO, make_graph

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, '..'))     # Re3_2/


# =====================================================================  A. COMPLEXITY
def exp_complexity(seeds=6):
    """Validate the manuscript per-defender complexity O(N_pop N_A + N_A|N_i|):
    hold N_A and the neighborhood degree BOUNDED (k-nearest ring) and vary only N_D.
    Then the TOTAL per-round cost should be linear in N_D and the PER-DEFENDER cost flat."""
    Ns = [10, 20, 40, 80, 160, 240]
    NA_FIXED = 8                                  # fixed target set
    RING_K = 2                                    # bounded degree = 2*RING_K = 4
    rows = []
    print("== A. Complexity scaling (fixed N_A=%d, bounded degree=%d) ==" % (NA_FIXED, 2 * RING_K))
    for N in Ns:
        per_round_t = []
        comm_per_round = []
        for s in range(seeds):
            scn = Scenario(n_def=N, n_att=NA_FIXED, L_max=3, seed=s)
            nb, E, D = make_graph(N, topo='ring', seed=s, ring_k=RING_K)
            sim = DistributedIDBO(scn, nb, E, delay=0, seed=100 + s)
            R = 10
            t0 = time.perf_counter()
            for k in range(R):
                sim._round(k)
            dt = (time.perf_counter() - t0) / R
            per_round_t.append(dt)
            comm_per_round.append(sim.comm_msgs / R)
        avg_deg = np.mean([len(x) for x in nb])
        rows.append((N, NA_FIXED, np.mean(per_round_t), np.std(per_round_t),
                     np.mean(comm_per_round), avg_deg))
        print(f"  N_D={N:4d} N_A={NA_FIXED:3d}: total/round={np.mean(per_round_t)*1e3:7.2f} ms  "
              f"per-defender={np.mean(per_round_t)/N*1e6:6.1f} us  "
              f"comm/round={np.mean(comm_per_round):7.1f}  deg={avg_deg:.1f}")
    np.save(os.path.join(HERE, 'data_complexity.npy'),
            np.array([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows]))
    return rows


# ====================================================================  B. SCALABILITY
def exp_scalability(seeds=6):
    """Rounds-to-consensus and cost as the whole problem scales up."""
    sizes = [(10, 4), (20, 8), (40, 16), (80, 32), (160, 64)]
    rows = []
    print("\n== B. Scalability ==")
    for ND, NA in sizes:
        rounds, costs, comm, walls = [], [], [], []
        for s in range(seeds):
            scn = Scenario(n_def=ND, n_att=NA, L_max=3, seed=s)
            nb, E, D = make_graph(ND, topo='rgg', seed=s)
            sim = DistributedIDBO(scn, nb, E, delay=2, seed=100 + s)
            r = sim.run(max_rounds=400, patience=5)
            rounds.append(r['rounds']); costs.append(r['cost'])
            comm.append(r['comm_msgs']); walls.append(r['wall'])
        rows.append((ND, NA, np.mean(rounds), np.std(rounds),
                     np.mean(costs), np.mean(comm), np.mean(walls)))
        print(f"  {ND:4d}v{NA:3d}: rounds={np.mean(rounds):6.1f}±{np.std(rounds):4.1f} "
              f"cost={np.mean(costs):7.2f} comm={np.mean(comm):9.0f} wall={np.mean(walls):.2f}s")
    np.save(os.path.join(HERE, 'data_scalability.npy'),
            np.array([(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows]))
    return rows


# ===================================================================  C. COMM-DELAY
def exp_delay(seeds=12, max_rounds=300):
    """Rounds-to-consensus vs delay, and Gamma trajectories; plus rounds vs graph diameter."""
    print("\n== C. Communication delay vs consensus efficiency ==")
    ND, NA = 40, 12
    delays = [0, 1, 2, 4, 8, 16]
    # (C1) rounds vs delay on a fixed RGG
    rounds_by_delay = {d: [] for d in delays}
    gamma_traj = {d: [] for d in delays}          # store one representative trajectory
    for d in delays:
        for s in range(seeds):
            scn = Scenario(n_def=ND, n_att=NA, L_max=3, seed=s)
            nb, E, D = make_graph(ND, topo='rgg', seed=s)
            sim = DistributedIDBO(scn, nb, E, delay=d, seed=100 + s)
            r = sim.run(max_rounds=max_rounds, patience=5)
            rounds_by_delay[d].append(r['rounds'])
            if s == 0:
                gamma_traj[d] = r['gamma']
        print(f"  delay={d:2d}: rounds={np.mean(rounds_by_delay[d]):6.1f} "
              f"± {np.std(rounds_by_delay[d]):4.1f}")

    # (C2) rounds vs graph diameter: vary topology/degree to sweep D at fixed N
    print("  -- rounds vs graph diameter --")
    diam_rows = []
    configs = [('complete', {}), ('rgg', {'radius': 0.55}), ('rgg', {'radius': 0.40}),
               ('rgg', {'radius': 0.30}), ('ring', {'ring_k': 2}), ('ring', {'ring_k': 1})]
    for topo, kw in configs:
        for s in range(seeds):
            scn = Scenario(n_def=ND, n_att=NA, L_max=3, seed=s)
            nb, E, D = make_graph(ND, topo=topo, seed=s, **kw)
            sim = DistributedIDBO(scn, nb, E, delay=2, seed=100 + s)
            r = sim.run(max_rounds=max_rounds, patience=5)
            diam_rows.append((D, r['rounds'], np.mean([len(x) for x in nb])))
    diam_rows = np.array(diam_rows)
    # aggregate rounds by diameter
    print("     diameter -> mean rounds")
    for Dval in sorted(set(diam_rows[:, 0].astype(int))):
        m = diam_rows[:, 0].astype(int) == Dval
        print(f"       D={Dval}: rounds={diam_rows[m,1].mean():.1f}  (n={m.sum()})")

    # save
    np.savez(os.path.join(HERE, 'data_delay.npz'),
             delays=np.array(delays),
             rounds_by_delay=np.array([rounds_by_delay[d] for d in delays]),
             gamma_traj=np.array([_pad(gamma_traj[d], max_rounds) for d in delays]),
             diam_rows=diam_rows)
    return rounds_by_delay, gamma_traj, diam_rows


def _pad(arr, n):
    out = np.full(n, np.nan)
    out[:len(arr)] = arr
    return out


def write_summary(comp, scal, delay_rounds):
    with open(os.path.join(OUT, 'complexity_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['experiment', 'key', 'value'])
        for (N, NA, t, tstd, comm, deg) in comp:
            w.writerow(['complexity', f'ND={N}', f'per_round_ms={t*1e3:.3f};comm={comm:.1f}'])
        for (ND, NA, rd, rdstd, cost, comm, wall) in scal:
            w.writerow(['scalability', f'{ND}v{NA}', f'rounds={rd:.1f};cost={cost:.2f}'])
        for d, rs in delay_rounds.items():
            w.writerow(['delay', f'd={d}', f'rounds={np.mean(rs):.1f}'])
    print('\nsaved complexity_summary.csv')


if __name__ == '__main__':
    comp = exp_complexity()
    scal = exp_scalability()
    dr, gt, diam = exp_delay()
    write_summary(comp, scal, dr)
    print('\nAll experiments done.')
