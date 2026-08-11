"""
plot_ablation.py -- Reviewer-response figure: coefficient-schedule ablation.
===========================================================================
Reads ablation_data.npz (from run_ablation.py) and produces a two-panel figure:

  (a) Elite-cost convergence curves (mean +/- std band) for the three schedules.
      Linear decay (paper) converges smoothly to a stable value; constant and
      no-decay coefficients keep oscillating.
  (b) Late-stage elite-cost oscillation (std over the final 30 iterations) per run,
      as box plots -- a direct quantitative measure of convergence stability.

Style matches the paper's other figures (Times serif, 300 dpi, IEEE look).
Outputs idbo_coeff_ablation.pdf/.png into revise_0723/.
"""
import numpy as np
import os
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
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.6,
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, '..'))    # revise_0723/

LABELS = {
    'linear':   r'Linear decay $c_k=c_0(1-k/K)$ (paper)',
    'constant': r'Constant $c_k=0.5\,c_0$',
    'none':     r'No decay $c_k=c_0$',
}
COLORS = {'linear': '#e91e63', 'constant': '#1f77b4', 'none': '#2ca02c'}
STYLES = {'linear': '-', 'constant': '--', 'none': '-.'}


def main():
    d = np.load(os.path.join(HERE, 'ablation_data.npz'), allow_pickle=True)
    schedules = [s for s in ['linear', 'constant', 'none']]
    K = int(d['max_iter'])
    iters = np.arange(1, K + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    # ---------- Panel (a): swarm-mean cost convergence (single snapshot) ----------
    ckey = 'repcurve_' if 'repcurve_linear' in d.files else 'curve_'
    for s in schedules:
        curve = d[f'{ckey}{s}']                    # (R, K) population-mean cost
        mu = curve.mean(axis=0)
        sd = curve.std(axis=0)
        ax1.plot(iters, mu, color=COLORS[s], linestyle=STYLES[s], label=LABELS[s])
        ax1.fill_between(iters, mu - sd, mu + sd, color=COLORS[s], alpha=0.15, linewidth=0)
    ax1.set_xlabel('IDBO iteration $k$')
    ax1.set_ylabel(r'Swarm-mean assignment cost')
    ax1.set_title('(a) Convergence of the swarm')
    ax1.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
    ax1.legend(loc='upper right', framealpha=0.92, edgecolor='black')

    # ---------- Panel (b): late-stage population contraction (pooled snapshots) ----------
    late = []
    for s in schedules:
        spread = d[f'spread_{s}']                  # (R,K) population cost spread
        late.append(spread[:, -25:].mean(axis=1))  # per-run mean spread over last 25 iters
    bp = ax2.boxplot(late, labels=['Linear\n(paper)', 'Constant', 'No decay'],
                     patch_artist=True, widths=0.6, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='white',
                                    markeredgecolor='black', markersize=5))
    for patch, s in zip(bp['boxes'], schedules):
        patch.set_facecolor(COLORS[s])
        patch.set_alpha(0.55)
    for med in bp['medians']:
        med.set_color('black')
    ax2.set_ylabel(r'Late-stage swarm cost spread')
    ax2.set_title('(b) Convergence stability (final 25 iterations)')
    ax2.grid(True, axis='y', linestyle='--', linewidth=0.4, alpha=0.6)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'idbo_coeff_ablation.{ext}'))
    print('saved idbo_coeff_ablation.pdf/.png to', OUT)

    # print numbers for the caption
    print('\n--- summary for caption ---')
    for s in schedules:
        curve = d[f'curve_{s}']
        spread = d[f'spread_{s}']
        best = d[f'best_{s}']
        print(f"{s:8s}: best-ever={best.mean():.3f}±{best.std():.3f}  "
              f"mean-final={curve[:,-1].mean():.3f}  "
              f"late-spread={spread[:,-25:].mean(axis=1).mean():.4f}")


if __name__ == '__main__':
    main()
