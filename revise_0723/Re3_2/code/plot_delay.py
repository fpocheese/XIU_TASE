"""
plot_delay.py -- Reviewer-response figure: communication delay vs consensus efficiency.
======================================================================================
Delay and consensus latency are reported in TIME. A synchronous consensus round takes one
communication period T_COMM (nominal data-link period); a message delay of d rounds is the
latency tau = d * T_COMM, and the consensus time is (rounds) * T_COMM.

Three panels:
  (a) Swarm disagreement Gamma vs consensus time, for several communication delays tau.
  (b) Time-to-consensus vs communication delay tau (mean +/- std) -- roughly linear.
  (c) Time-to-consensus vs communication-graph diameter D (validates the ~N*D claim).
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
    'axes.labelsize': 12, 'axes.titlesize': 12, 'legend.fontsize': 9,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 1.0, 'lines.linewidth': 1.6,
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, '..'))

# Nominal communication period: one synchronous consensus round == one data-link period.
# 50 ms == a 20 Hz inter-agent data link.  Delay and consensus latency are shown in TIME.
T_COMM_MS = 50.0

d = np.load(os.path.join(HERE, 'data_delay.npz'))
delays = d['delays']                        # message delay in rounds -> convert to ms
tau_ms = delays * T_COMM_MS                  # communication delay (ms)
rounds_by_delay = d['rounds_by_delay']      # (n_delays, seeds)
gamma_traj = d['gamma_traj']                # (n_delays, max_rounds) padded with nan
diam_rows = d['diam_rows']                  # (n, 3): diameter, rounds, avg_deg

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.0))
cmap = plt.cm.viridis(np.linspace(0, 0.85, len(delays)))

# ---- (a) Gamma vs consensus time ----
for i, dd in enumerate(delays):
    g = gamma_traj[i]
    g = g[~np.isnan(g)]
    tms = np.arange(1, len(g) + 1) * T_COMM_MS / 1000.0        # seconds
    ax1.plot(tms, g, color=cmap[i], label=fr'$\tau={int(dd*T_COMM_MS)}$ ms')
ax1.set_xlabel('Consensus time (s)')
ax1.set_ylabel(r'Swarm disagreement $\Gamma$')
ax1.set_title('(a) Consensus under communication delay')
ax1.set_xlim(0, min(120, gamma_traj.shape[1]) * T_COMM_MS / 1000.0)
ax1.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
ax1.legend(title='comm. delay', framealpha=0.92, edgecolor='black', ncol=2)

# ---- (b) time-to-consensus vs communication delay ----
mu = rounds_by_delay.mean(axis=1) * T_COMM_MS / 1000.0        # seconds
sd = rounds_by_delay.std(axis=1) * T_COMM_MS / 1000.0
ax2.errorbar(tau_ms, mu, yerr=sd, marker='o', color='#e91e63',
             capsize=3, markeredgecolor='black', markeredgewidth=0.5)
# linear fit (time-to-consensus [s] vs delay [ms])
coef = np.polyfit(tau_ms, mu, 1)
xs = np.linspace(tau_ms.min(), tau_ms.max(), 50)
ax2.plot(xs, np.polyval(coef, xs), '--', color='gray',
         label=f'linear fit (slope {coef[0]:.2f} s/ms)')
ax2.set_xlabel(r'Communication delay $\tau$ (ms)')
ax2.set_ylabel('Time to consensus (s)')
ax2.set_title('(b) Consensus time grows linearly with delay')
ax2.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
ax2.legend(framealpha=0.92, edgecolor='black')

# ---- (c) time-to-consensus vs diameter ----
Dvals = sorted(set(diam_rows[:, 0].astype(int)))
means = [diam_rows[diam_rows[:, 0].astype(int) == Dv, 1].mean() * T_COMM_MS / 1000.0 for Dv in Dvals]
stds = [diam_rows[diam_rows[:, 0].astype(int) == Dv, 1].std() * T_COMM_MS / 1000.0 for Dv in Dvals]
ax3.errorbar(Dvals, means, yerr=stds, marker='s', color='#1f77b4',
             capsize=3, markeredgecolor='black', markeredgewidth=0.5)
coef3 = np.polyfit(Dvals, means, 1)
xs3 = np.linspace(min(Dvals), max(Dvals), 50)
ax3.plot(xs3, np.polyval(coef3, xs3), '--', color='gray',
         label=f'linear fit (slope {coef3[0]:.2f} s/hop)')
ax3.set_xlabel('Communication-graph diameter $D$ (hops)')
ax3.set_ylabel('Time to consensus (s)')
ax3.set_title('(c) Consensus time scales with graph diameter')
ax3.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
ax3.legend(framealpha=0.92, edgecolor='black')

plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, f'idbo_comm_delay.{ext}'))
print('saved idbo_comm_delay.pdf/.png  (T_COMM = %.0f ms)' % T_COMM_MS)
print('delay slope   =', round(coef[0], 3), 's per ms of delay')
print('diameter slope =', round(coef3[0], 3), 's per hop')
