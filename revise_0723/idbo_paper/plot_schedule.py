"""
plot_schedule.py -- Reviewer-response mechanism figure: coefficient schedule.
============================================================================
Illustrates WHY the adaptive coefficients stabilize convergence, independent of
any run:

  (a) The linear-decay multiplier s(k)=1-k/K applied to every coefficient
      {alpha,beta,gamma,delta,delta',eta} and the dancing noise std sigma_d,
      versus the constant and no-decay policies.
  (b) The resulting stochastic-perturbation magnitude of the update operators.
      As the exploratory coefficients decay to zero, the random part of the four
      operators vanishes and the IDBO update degenerates into a deterministic,
      advantage-weighted local refinement -- a Robbins--Monro-type sufficient
      condition for a stable fixed point.

Outputs idbo_coeff_schedule.pdf/.png into revise_0723/.
"""
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 11,
    'axes.labelsize': 13, 'axes.titlesize': 13, 'legend.fontsize': 10,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 1.0, 'lines.linewidth': 1.8,
})

OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

K = 120
k = np.arange(K + 1)
lin = np.maximum(1 - k / K, 0.0)
const = np.full_like(k, 0.5, dtype=float)
none = np.ones_like(k, dtype=float)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

# ---- (a) schedule multiplier ----
ax1.plot(k, lin, '-', color='#e91e63', label=r'Linear decay $s(k)=1-k/K$ (paper)')
ax1.plot(k, const, '--', color='#1f77b4', label=r'Constant $s(k)=0.5$')
ax1.plot(k, none, '-.', color='#2ca02c', label=r'No decay $s(k)=1$')
ax1.set_xlabel('IDBO iteration $k$')
ax1.set_ylabel(r'Coefficient multiplier $s(k)$')
ax1.set_title('(a) Adaptive coefficient schedule')
ax1.set_ylim(-0.03, 1.08)
ax1.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
ax1.legend(loc='center left', framealpha=0.92, edgecolor='black')

# ---- (b) stochastic perturbation magnitude of the operators ----
# Rolling random probe ~ beta0*s(k)*exp(-k/K); Dancing noise ~ sigma_d0*s(k).
beta0, sd0 = 0.50, 0.50
roll_lin = beta0 * lin * np.exp(-k / K)
roll_const = beta0 * const * np.exp(-k / K)
roll_none = beta0 * none * np.exp(-k / K)
dance_lin = sd0 * lin
dance_const = sd0 * const
dance_none = sd0 * none

ax2.plot(k, roll_lin + dance_lin, '-', color='#e91e63', label='Linear decay (paper)')
ax2.plot(k, roll_const + dance_const, '--', color='#1f77b4', label='Constant')
ax2.plot(k, roll_none + dance_none, '-.', color='#2ca02c', label='No decay')
ax2.axhspan(0, 0.03, color='gray', alpha=0.15)
ax2.text(K * 0.3, 0.05, 'deterministic refinement regime', color='gray', fontsize=9)
ax2.set_xlabel('IDBO iteration $k$')
ax2.set_ylabel('Stochastic perturbation magnitude')
ax2.set_title('(b) Vanishing exploration $\\Rightarrow$ stable fixed point')
ax2.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
ax2.legend(loc='upper right', framealpha=0.92, edgecolor='black')

plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, f'idbo_coeff_schedule.{ext}'))
print('saved idbo_coeff_schedule.pdf/.png to', OUT)
