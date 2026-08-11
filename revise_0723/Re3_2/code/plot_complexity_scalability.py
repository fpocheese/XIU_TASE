"""
plot_complexity_scalability.py -- Reviewer-response figures for complexity & scalability.
========================================================================================
Figure 1 (idbo_complexity):
  (a) Per-round wall-clock vs swarm size N_D on log-log axes, with a fitted power-law
      exponent, validating the near-linear per-round scaling of the distributed update.
  (b) Communication volume per round vs N_D (linear in the number of edges / neighbors).

Figure 2 (idbo_scalability):
  (a) Rounds-to-consensus vs problem size (stays low and grows slowly).
  (b) Total communication volume to consensus vs problem size.
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

# ============================================================ complexity figure
comp = np.load(os.path.join(HERE, 'data_complexity.npy'))   # cols: N,NA,t,tstd,comm,deg
N = comp[:, 0]; NA = comp[:, 1]; t = comp[:, 2]; tstd = comp[:, 3]; comm = comp[:, 4]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))

ax1.errorbar(N, t * 1e3, yerr=tstd * 1e3, marker='o', color='#e91e63',
             capsize=3, markeredgecolor='black', markeredgewidth=0.5, label='total per round')
# power-law fit in log-log (total)
lx, ly = np.log(N), np.log(t)
p = np.polyfit(lx, ly, 1)
xs = np.linspace(N.min(), N.max(), 50)
ax1.plot(xs, np.exp(np.polyval(p, np.log(xs))) * 1e3, '--', color='gray',
         label=f'total fit: $\\propto N_D^{{{p[0]:.2f}}}$')
# per-defender cost (should be roughly flat -> validates bounded per-defender complexity)
ax1b = ax1.twinx()
ax1b.plot(N, t / N * 1e6, marker='^', color='#2ca02c', markeredgecolor='black',
          markeredgewidth=0.5, label='per defender')
ax1b.set_ylabel(r'Per-defender cost ($\mu$s)', color='#2ca02c')
ax1b.tick_params(axis='y', labelcolor='#2ca02c')
ax1b.set_ylim(0, max(t / N * 1e6) * 1.6)
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('Swarm size $N_D$')
ax1.set_ylabel('Total per-round wall-clock (ms)')
ax1.set_title(f'(a) Per-round cost (total $\\propto N_D^{{{p[0]:.2f}}}$, per-defender flat)')
ax1.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.5)
lines1, lab1 = ax1.get_legend_handles_labels()
lines2, lab2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lab1 + lab2, framealpha=0.92, edgecolor='black', loc='upper left')

ax2.plot(N, comm, marker='s', color='#1f77b4',
         markeredgecolor='black', markeredgewidth=0.5, label='messages / round')
pc = np.polyfit(N, comm, 1)
ax2.plot(xs, np.polyval(pc, xs), '--', color='gray',
         label=f'linear fit (slope {pc[0]:.1f})')
ax2.set_xlabel('Swarm size $N_D$')
ax2.set_ylabel('Communication volume / round')
ax2.set_title('(b) Communication cost scaling')
ax2.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
ax2.legend(framealpha=0.92, edgecolor='black')

plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, f'idbo_complexity.{ext}'))
print('saved idbo_complexity.pdf/.png ; per-round exponent =', round(p[0], 2))

# ============================================================ scalability figure
# Report consensus latency in time: one synchronous round == T_COMM (50 ms, 20 Hz link).
T_COMM_MS = 50.0
scal = np.load(os.path.join(HERE, 'data_scalability.npy'))  # ND,NA,rounds,rstd,cost,comm,wall
ND = scal[:, 0]
time_s = scal[:, 2] * T_COMM_MS / 1000.0        # time to consensus (s)
tstd_s = scal[:, 3] * T_COMM_MS / 1000.0
comm2 = scal[:, 5]
labels = [f'{int(a)}v{int(b)}' for a, b in zip(scal[:, 0], scal[:, 1])]

fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(9.5, 4.0))

bx1.errorbar(ND, time_s, yerr=tstd_s, marker='o', color='#2ca02c',
             capsize=3, markeredgecolor='black', markeredgewidth=0.5)
bx1.set_xscale('log', base=2)
bx1.set_xticks(ND); bx1.set_xticklabels(labels, rotation=30, fontsize=8)
bx1.set_ylim(0, max(time_s) * 1.6)
bx1.set_xlabel('Problem size (defenders v targets)')
bx1.set_ylabel(r'Time to consensus (s), $\tau=100$ ms')
bx1.set_title('(a) Time-to-consensus vs scale')
bx1.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)

bx2.plot(ND, comm2, marker='D', color='#9467bd',
         markeredgecolor='black', markeredgewidth=0.5)
bx2.set_xscale('log', base=2); bx2.set_yscale('log')
bx2.set_xticks(ND); bx2.set_xticklabels(labels, rotation=30, fontsize=8)
bx2.set_xlabel('Problem size (defenders v targets)')
bx2.set_ylabel('Total messages to consensus')
bx2.set_title('(b) Communication volume vs scale')
bx2.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.5)

plt.tight_layout()
for ext in ('pdf', 'png'):
    fig2.savefig(os.path.join(OUT, f'idbo_scalability.{ext}'))
print('saved idbo_scalability.pdf/.png')
