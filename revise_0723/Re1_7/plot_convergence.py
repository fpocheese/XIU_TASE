"""
plot_convergence.py -- Re1_7 confirmatory figure for the convergence/equilibrium claims.
=========================================================================================
Two panels, both re-plotting data already produced by the paper-faithful harness:

  (a) Swarm-disagreement metric Gamma^(k) vs iteration for the three coefficient
      schedules (data: revise_0723/idbo_paper/ablation_data.npz, key gamma_*).
      The linear-decay schedule (paper) drives Gamma below the epsilon threshold and
      holds it there; constant / no-decay keep it oscillating above threshold. This is
      the exact quantity Eq. (convergence) is defined on -- it validates the
      "settles at a fixed point rather than orbiting it" claim.

  (b) Rounds-to-consensus vs communication-graph diameter D
      (data: revise_0723/Re3_2/code/data_delay.npz, key diam_rows).
      Rounds grow near-linearly with D, empirically confirming the O(N*D) consensus
      bound (CBBA, Choi-Brunet-How 2009) cited in Section III.

Style matches the paper's other reviewer-response figures (Times serif, 300 dpi).
Outputs re1_7_convergence.pdf/.png into revise_0723/Re1_7/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 9.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.7,
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))            # revise_0723/
ABL = os.path.join(ROOT, 'idbo_paper', 'ablation_data.npz')
DELAY = os.path.join(ROOT, 'Re3_2', 'code', 'data_delay.npz')

EPS_CON = 0.05    # convergence threshold epsilon used for Gamma in the manuscript

SCHED = {
    'linear':   ('Linear decay (proposed)', '#1f77b4', '-'),
    'constant': ('Constant coeff.',         '#ff7f0e', '--'),
    'none':     ('No decay',                '#d62728', ':'),
}

def main():
    d = np.load(ABL)
    max_iter = int(d['max_iter'])
    it = np.arange(1, max_iter + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))

    # ---------------- panel (a): Gamma^(k) vs iteration ----------------
    for s, (lab, col, ls) in SCHED.items():
        g = d[f'gamma_{s}']                    # (n_runs, max_iter)
        mean = g.mean(axis=0)
        std = g.std(axis=0)
        ax1.plot(it, mean, ls, color=col, label=lab)
        ax1.fill_between(it, np.maximum(mean - std, 0), mean + std,
                         color=col, alpha=0.15, linewidth=0)
    ax1.axhline(EPS_CON, color='0.35', lw=1.1, dash_capstyle='round',
                dashes=(4, 3))
    ax1.text(max_iter * 0.60, EPS_CON * 1.25, r'threshold $\epsilon$',
             color='0.30', fontsize=9.5)
    ax1.set_xlabel('Iteration $k$')
    ax1.set_ylabel(r'Swarm disagreement $\Gamma^{(k)}$')
    ax1.set_title('(a) Consensus convergence')
    ax1.set_xlim(1, max_iter)
    ax1.set_ylim(bottom=0)
    ax1.legend(frameon=False, loc='upper right')
    ax1.grid(True, alpha=0.25)

    # ---------------- panel (b): rounds vs graph diameter --------------
    dd = np.load(DELAY)
    diam_rows = dd['diam_rows']                # (n, 3): [diameter, rounds, *]
    diam = diam_rows[:, 0].astype(int)
    rounds = diam_rows[:, 1].astype(float)
    Ds = np.array(sorted(set(diam)))
    mean_r = np.array([rounds[diam == D].mean() for D in Ds])
    std_r = np.array([rounds[diam == D].std() for D in Ds])

    ax2.errorbar(Ds, mean_r, yerr=std_r, fmt='o', color='#1f77b4',
                 ms=5, capsize=3, lw=1.4, label='Measured rounds')
    # linear reference fit through the origin: rounds ~ a * D
    a = np.sum(Ds * mean_r) / np.sum(Ds * Ds)
    ax2.plot(Ds, a * Ds, '--', color='0.4',
             label=r'Linear reference $\propto D$')
    ax2.set_xlabel('Communication-graph diameter $D$')
    ax2.set_ylabel('Rounds to consensus')
    ax2.set_title(r'(b) Consensus rounds scale as $O(N\!\cdot\!D)$')
    ax2.set_xlim(0, Ds.max() * 1.05)
    ax2.set_ylim(bottom=0)
    ax2.legend(frameon=False, loc='upper left')
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        out = os.path.join(HERE, f're1_7_convergence.{ext}')
        fig.savefig(out)
        print('saved', out)

    # console summary for the response letter
    print('\n--- panel (a) final Gamma (mean over runs) ---')
    for s in SCHED:
        g = d[f'gamma_{s}']
        print(f'  {s:8s}: Gamma_final = {g[:,-1].mean():.4f}')
    print('\n--- panel (b) rounds vs diameter ---')
    for D, m, sd in zip(Ds, mean_r, std_r):
        print(f'  D={D:2d}: rounds = {m:5.1f} +/- {sd:4.1f}')
    print(f'  linear-through-origin slope a = {a:.2f} rounds per unit D')


if __name__ == '__main__':
    main()
