"""
run_ablation.py -- Coefficient-schedule ablation for the paper-faithful IDBO.
============================================================================
Reviewer response (Section III.B): quantify how the adaptive rolling/dancing/
breeding/stealing coefficients influence convergence STABILITY.

Three coefficient schedules are compared on the SAME engagement snapshot:
    'linear'  : c_k = c0 (1 - k/K)   -- the paper's adaptive decay
    'constant': c_k = 0.5 c0         -- fixed coefficients, no decay
    'none'    : c_k = c0             -- full-strength exploration throughout

For each schedule we run R independent seeds and record the per-iteration ELITE
cost (the working assignment the swarm would act on).  Outputs raw data (.npz) and
summary statistics (.csv) consumed by plot_ablation.py.
"""
import numpy as np
import csv
import os
from scenario_paper import Scenario
from idbo_paper import IDBO_paper

# ------------------------------------------------------------------ config
N_POP = 40
MAX_ITER = 120
N_SCN = 5                    # number of independent engagement snapshots
RUNS_PER_SCN = 8             # seeds per snapshot  (total = N_SCN * RUNS_PER_SCN)
SCHEDULES = ['linear', 'constant', 'none']
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_RUNS = N_SCN * RUNS_PER_SCN


def main():
    print(f"Engagement: 20 defenders vs 8 targets, L_max=3")
    print(f"Snapshots={N_SCN}, runs/snapshot={RUNS_PER_SCN}, total runs={N_RUNS}, "
          f"pop={N_POP}, iters={MAX_ITER}\n")

    curves = {s: np.zeros((N_RUNS, MAX_ITER)) for s in SCHEDULES}   # population-mean cost
    spreads = {s: np.zeros((N_RUNS, MAX_ITER)) for s in SCHEDULES}  # population spread
    gammas = {s: np.zeros((N_RUNS, MAX_ITER)) for s in SCHEDULES}
    best_costs = {s: np.zeros(N_RUNS) for s in SCHEDULES}

    for s in SCHEDULES:
        idx = 0
        for scn_seed in range(N_SCN):
            scn = Scenario(n_def=20, n_att=8, L_max=3, seed=scn_seed)
            for r in range(RUNS_PER_SCN):
                best, assign, conv, hist = IDBO_paper(
                    N=N_POP, max_iter=MAX_ITER, scn=scn,
                    schedule=s, seed=1000 * scn_seed + r, return_history=True)
                curves[s][idx] = conv
                spreads[s][idx] = hist['spread']
                gammas[s][idx] = hist['gamma']
                best_costs[s][idx] = best
                idx += 1
        late_spread = spreads[s][:, -25:].mean(axis=1).mean()
        print(f"[{s:8s}] best-ever cost = {best_costs[s].mean():.4f} "
              f"± {best_costs[s].std():.4f} | mean-final = {curves[s][:, -1].mean():.4f} "
              f"| late-pop-spread = {late_spread:.4f}")

    # ---- also record curves for ONE representative snapshot (seed 0) so panel (a)
    #      shows curves on a single consistent cost scale ----
    rep_curves = {s: curves[s][:RUNS_PER_SCN] for s in SCHEDULES}
    rep_spreads = {s: spreads[s][:RUNS_PER_SCN] for s in SCHEDULES}

    # ---- save raw data ----
    np.savez(os.path.join(OUT_DIR, 'ablation_data.npz'),
             schedules=np.array(SCHEDULES),
             **{f'curve_{s}': curves[s] for s in SCHEDULES},        # pooled mean-cost
             **{f'spread_{s}': spreads[s] for s in SCHEDULES},      # pooled spread
             **{f'repcurve_{s}': rep_curves[s] for s in SCHEDULES}, # single snapshot mean-cost
             **{f'repspread_{s}': rep_spreads[s] for s in SCHEDULES},
             **{f'gamma_{s}': gammas[s] for s in SCHEDULES},
             **{f'best_{s}': best_costs[s] for s in SCHEDULES},
             max_iter=MAX_ITER, n_runs=N_RUNS, n_pop=N_POP,
             runs_per_scn=RUNS_PER_SCN, n_scn=N_SCN)
    print("\nsaved ablation_data.npz")

    # ---- summary csv ----
    with open(os.path.join(OUT_DIR, 'ablation_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['schedule', 'best_cost_mean', 'best_cost_std',
                    'mean_cost_final', 'late_pop_spread',
                    'gamma_final_mean'])
        for s in SCHEDULES:
            w.writerow([s,
                        f"{best_costs[s].mean():.6f}",
                        f"{best_costs[s].std():.6f}",
                        f"{curves[s][:, -1].mean():.6f}",
                        f"{spreads[s][:, -25:].mean(axis=1).mean():.6f}",
                        f"{gammas[s][:, -1].mean():.6f}"])
    print("saved ablation_summary.csv")


if __name__ == '__main__':
    main()
