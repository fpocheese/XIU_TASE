"""
Main script – UAV Task Assignment Optimisation
================================================
20 fixed-wing UAVs → 8 targets
Compare: PSO, GWO, DBO, SSA, BOA, IDBO (ours)

Usage:
    python main.py                     # default: 50 agents, 300 iter, 30 runs
    python main.py --N 50 --iter 300 --runs 30
"""

import os, sys, argparse, time, warnings
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch
import matplotlib.ticker as ticker

# ── local modules ─────────────────────────────────────────
from scenario import (get_task_assignment_problem, decode_solution,
                       print_assignment, N_UAV, N_TARGET,
                       UAV_POS, UAV_SPEED, UAV_HEADING,
                       TARGET_POS, TARGET_THREAT, cost_function)
from pso import PSO
from gwo import GWO
from dbo import DBO
from ssa import SSA
from boa import BOA
from idbo import IDBO

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# Publication-quality Matplotlib settings (IEEE Trans. style)
# ═══════════════════════════════════════════════════════════
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 9,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
})

ALGO_LIST = ['PSO', 'GWO', 'DBO', 'SSA', 'BOA', 'IDBO']

COLORS = {
    'PSO':  '#1f77b4',
    'GWO':  '#7f7f7f',
    'DBO':  '#2ca02c',
    'SSA':  '#ff7f0e',
    'BOA':  '#d62728',
    'IDBO': '#e91e63',
}
MARKERS = {
    'PSO': '^', 'GWO': 'p', 'DBO': 'D',
    'SSA': 'o', 'BOA': 's', 'IDBO': '*',
}

ALGO_FUNCS = {
    'PSO':  PSO,
    'GWO':  GWO,
    'DBO':  DBO,
    'SSA':  SSA,
    'BOA':  BOA,
    'IDBO': IDBO,
}


# ═══════════════════════════════════════════════════════════
# Run algorithm multiple times
# ═══════════════════════════════════════════════════════════
def run_algorithm(name, N, max_iter, lb, ub, dim, fobj, n_runs=1):
    scores, curves, times_list, best_positions = [], [], [], []
    best_global_score = np.inf
    best_global_pos = None
    for r in range(n_runs):
        t0 = time.time()
        score, pos, curve = ALGO_FUNCS[name](N, max_iter, lb, ub, dim, fobj)
        elapsed = time.time() - t0
        scores.append(score)
        curves.append(curve)
        times_list.append(elapsed)
        best_positions.append(pos)
        if score < best_global_score:
            best_global_score = score
            best_global_pos = pos.copy()
        if n_runs > 1:
            print(f'    {name} run {r+1}/{n_runs}  cost={score:.6f}  time={elapsed:.2f}s')
    return {
        'scores': np.array(scores),
        'curves': np.array(curves),
        'times': np.array(times_list),
        'best_pos': best_global_pos,
    }


# ═══════════════════════════════════════════════════════════
# Figure 1 – Convergence curves
# ═══════════════════════════════════════════════════════════
def plot_convergence(results, max_iter, out_dir, show_title=True):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    iters = np.arange(1, max_iter + 1)
    CNT = 14                                         # marker 显示间隔个数
    k_idx = list(np.round(np.linspace(0, max_iter - 1, CNT)).astype(int))

    # 为每条线添加不同 linestyle，使黑白打印也能区分
    LINE_STYLES = {
        'PSO': '-', 'GWO': '--', 'DBO': '-.',
        'SSA': ':', 'BOA': '-', 'IDBO': '-',
    }

    for algo in ALGO_LIST:
        curve = results[algo]['curves'].mean(axis=0)
        ax.plot(iters, curve,
                color=COLORS[algo],
                linestyle=LINE_STYLES[algo],
                linewidth=1.6,
                marker=MARKERS[algo],          # marker 合入同一条线
                markevery=k_idx,               # 仅在指定位置显示 marker
                markersize=6,
                markeredgecolor='black',
                markeredgewidth=0.4,
                label=algo)

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Average Cost')
    ax.legend(loc='upper right', ncol=2, columnspacing=0.8,
              handletextpad=0.4, handlelength=2.8)   # handlelength 加长，线型+marker 同时可见
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    _save(fig, out_dir, 'convergence')


# ═══════════════════════════════════════════════════════════
# Figure 2 – Box-plot
# ═══════════════════════════════════════════════════════════
def plot_boxplot(results, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    data = [results[a]['scores'] for a in ALGO_LIST]
    bp = ax.boxplot(data, labels=ALGO_LIST, patch_artist=True,
                    widths=0.5, showfliers=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='white',
                                   markeredgecolor='black', markersize=4),
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    for patch, algo in zip(bp['boxes'], ALGO_LIST):
        patch.set_facecolor(COLORS[algo])
        patch.set_alpha(0.65)
    ax.set_ylabel('Total Cost')
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    _save(fig, out_dir, 'boxplot')


# ═══════════════════════════════════════════════════════════
# Figure 3 – Bar chart with error bars
# ═══════════════════════════════════════════════════════════
def plot_bar_errorbar(results, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    means = [results[a]['scores'].mean() for a in ALGO_LIST]
    stds  = [results[a]['scores'].std()  for a in ALGO_LIST]
    x = np.arange(len(ALGO_LIST))
    ax.bar(x, means, yerr=stds, width=0.55, capsize=4,
           color=[COLORS[a] for a in ALGO_LIST], edgecolor='black',
           linewidth=0.6, alpha=0.8, error_kw=dict(lw=1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(ALGO_LIST)
    ax.set_ylabel('Total Cost (mean ± std)')
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    _save(fig, out_dir, 'bar_comparison')


# ═══════════════════════════════════════════════════════════
# Figure 4 – Violin plot
# ═══════════════════════════════════════════════════════════
def plot_violin(results, out_dir, show_title=True):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    data = [results[a]['scores'] for a in ALGO_LIST]
    parts = ax.violinplot(data, positions=range(len(ALGO_LIST)),
                          showmeans=True, showmedians=True)
    for idx, (pc, algo) in enumerate(zip(parts['bodies'], ALGO_LIST)):
        pc.set_facecolor(COLORS[algo])
        pc.set_alpha(0.6)
    ax.set_xticks(range(len(ALGO_LIST)))
    ax.set_xticklabels(ALGO_LIST)
    ax.set_ylabel('Total Cost')
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    _save(fig, out_dir, 'violin')


# ═══════════════════════════════════════════════════════════
# Figure 5 – UAV-Target Assignment Map (best solution, IDBO)
# ═══════════════════════════════════════════════════════════
def plot_assignment_map(results, out_dir):
    """Scatter map showing UAV positions, target positions, and assignment links."""
    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot all UAVs (grey dots for unassigned, coloured for assigned)
    ax.scatter(UAV_POS[:, 0], UAV_POS[:, 1],
               c='#5dade2', s=60, marker='^', edgecolors='black',
               linewidths=0.6, zorder=5, label='UAV')
    for i in range(N_UAV):
        ax.annotate(f'U{i+1}', (UAV_POS[i, 0]+1, UAV_POS[i, 1]+1),
                    fontsize=6, color='#2c3e50')

    # Heading arrows for each UAV
    arrow_len = 3.0
    for i in range(N_UAV):
        rad = np.radians(UAV_HEADING[i])
        dx = arrow_len * np.sin(rad)
        dy = arrow_len * np.cos(rad)
        ax.annotate('', xy=(UAV_POS[i, 0]+dx, UAV_POS[i, 1]+dy),
                    xytext=(UAV_POS[i, 0], UAV_POS[i, 1]),
                    arrowprops=dict(arrowstyle='->', color='#5dade2',
                                   lw=0.8, alpha=0.6))

    # Targets
    ax.scatter(TARGET_POS[:, 0], TARGET_POS[:, 1],
               c='#e74c3c', s=100, marker='*', edgecolors='black',
               linewidths=0.6, zorder=5, label='Target')
    for j in range(N_TARGET):
        ax.annotate(f'T{j+1}', (TARGET_POS[j, 0]+1, TARGET_POS[j, 1]-3),
                    fontsize=7, fontweight='bold', color='#c0392b')

    # Draw assignment lines for IDBO best solution
    best_x = results['IDBO']['best_pos']
    assignment = decode_solution(best_x)
    for j in range(N_TARGET):
        uav_id = assignment[j]
        ax.plot([UAV_POS[uav_id, 0], TARGET_POS[j, 0]],
                [UAV_POS[uav_id, 1], TARGET_POS[j, 1]],
                '--', color=COLORS['IDBO'], linewidth=1.2, alpha=0.8)

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    _save(fig, out_dir, 'assignment_map')


# ═══════════════════════════════════════════════════════════
# Figure 6 – Multi-metric Radar chart
# ═══════════════════════════════════════════════════════════
def plot_radar(results, out_dir):
    """Radar chart with 5 metrics: mean cost, std, best, worst, avg time."""
    metrics = ['Mean Cost', 'Std', 'Best Cost', 'Worst Cost', 'Avg Time (s)']
    N_m = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N_m, endpoint=False).tolist()
    angles += angles[:1]

    raw = {}
    for algo in ALGO_LIST:
        s = results[algo]['scores']
        raw[algo] = [s.mean(), s.std(), s.min(), s.max(), results[algo]['times'].mean()]

    # Normalise each metric to [0,1] (lower = better → inverted)
    norm = {}
    for algo in ALGO_LIST:
        norm[algo] = []
    for m_idx in range(N_m):
        vals = [raw[a][m_idx] for a in ALGO_LIST]
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx != mn else 1
        for algo in ALGO_LIST:
            norm[algo].append((raw[algo][m_idx] - mn) / rng)

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    for algo in ALGO_LIST:
        vals = norm[algo] + norm[algo][:1]
        ax.plot(angles, vals, color=COLORS[algo], marker=MARKERS[algo],
                linewidth=1.3, markersize=4, label=algo)
        ax.fill(angles, vals, color=COLORS[algo], alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, 'radar_chart')


# ═══════════════════════════════════════════════════════════
# Figure 7 – Cost sub-component breakdown (stacked bar)
# ═══════════════════════════════════════════════════════════
def plot_cost_breakdown(results, out_dir):
    """Stacked bar chart showing distance / heading / balance / threat costs
    for the best solution found by each algorithm."""
    from scenario import (_distance_cost, _heading_cost,
                           _balance_cost, _threat_cost,
                           W_DIST, W_HEADING, W_BALANCE, W_THREAT)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    comp_names = ['Distance', 'Heading', 'Balance', 'Threat']
    comp_colors = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c']
    x = np.arange(len(ALGO_LIST))

    bottoms = np.zeros(len(ALGO_LIST))
    for ci, (cfun, w, cname, cc) in enumerate(zip(
            [_distance_cost, _heading_cost, _balance_cost, _threat_cost],
            [W_DIST, W_HEADING, W_BALANCE, W_THREAT],
            comp_names, comp_colors)):
        vals = []
        for algo in ALGO_LIST:
            assignment = decode_solution(results[algo]['best_pos'])
            vals.append(w * cfun(assignment))
        vals = np.array(vals)
        ax.bar(x, vals, bottom=bottoms, width=0.55,
               color=cc, edgecolor='black', linewidth=0.5, alpha=0.8, label=cname)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(ALGO_LIST)
    ax.set_ylabel('Weighted Cost Components')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    _save(fig, out_dir, 'cost_breakdown')


# ═══════════════════════════════════════════════════════════
# Figure 8 – Execution time comparison
# ═══════════════════════════════════════════════════════════
def plot_time_comparison(results, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    means = [results[a]['times'].mean() for a in ALGO_LIST]
    stds = [results[a]['times'].std() for a in ALGO_LIST]
    x = np.arange(len(ALGO_LIST))
    ax.bar(x, means, yerr=stds, width=0.55, capsize=4,
           color=[COLORS[a] for a in ALGO_LIST], edgecolor='black',
           linewidth=0.6, alpha=0.8, error_kw=dict(lw=1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(ALGO_LIST)
    ax.set_ylabel('Execution Time (s)')
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    _save(fig, out_dir, 'time_comparison')


# ═══════════════════════════════════════════════════════════
# Utility: save pdf + png
# ═══════════════════════════════════════════════════════════
def _save(fig, out_dir, name):
    path = os.path.join(out_dir, f'{name}.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f'  [saved] {path}')


# ═══════════════════════════════════════════════════════════
# Statistical table
# ═══════════════════════════════════════════════════════════
def print_table(results):
    print(f'\n{"="*76}')
    print(f'  Statistical Results – 20 UAVs → 8 Targets Task Assignment')
    print(f'{"="*76}')
    print(f'{"Algorithm":<10} {"Best":>12} {"Worst":>12} {"Mean":>12} '
          f'{"Std":>12} {"AvgTime(s)":>12}')
    print('-'*76)
    means = {}
    for algo in ALGO_LIST:
        s = results[algo]['scores']
        t = results[algo]['times']
        means[algo] = s.mean()
        print(f'{algo:<10} {s.min():>12.6f} {s.max():>12.6f} {s.mean():>12.6f} '
              f'{s.std():>12.6f} {t.mean():>12.3f}')
    ranked = sorted(means, key=means.get)
    print(f'\n  Ranking (lower cost = better): {" > ".join(ranked)}')
    print(f'{"="*76}\n')


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='UAV Task Assignment – IDBO vs baselines')
    parser.add_argument('--N', type=int, default=50, help='Population size')
    parser.add_argument('--iter', type=int, default=300, help='Max iterations')
    parser.add_argument('--runs', type=int, default=30, help='Independent runs')
    parser.add_argument('--outdir', type=str, default='results', help='Output directory')
    args = parser.parse_args()

    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)

    lb, ub, dim, fobj = get_task_assignment_problem()

    print(f'\n{"#"*60}')
    print(f'  UAV Task Assignment Optimisation')
    print(f'  {N_UAV} UAVs → {N_TARGET} Targets')
    print(f'  N={args.N}  iter={args.iter}  runs={args.runs}')
    print(f'  dim={dim}  lb={lb}  ub={ub}')
    print(f'{"#"*60}')

    results = {}
    for algo in ALGO_LIST:
        print(f'\n  >> Running {algo} ...')
        results[algo] = run_algorithm(algo, args.N, args.iter,
                                       lb, ub, dim, fobj, args.runs)

    # ── Print stats ──
    print_table(results)

    # ── Print best assignment (IDBO) ──
    print("  ── IDBO Best Assignment ──")
    print_assignment(results['IDBO']['best_pos'])

    # ── Save results (pickle) so figures can be regenerated from data later ──
    results_path = os.path.join(out_dir, 'results.pkl')
    with open(results_path, 'wb') as fh:
        pickle.dump(results, fh)
    print(f'  [saved] {results_path}')

    # ── Generate all figures ──
    print(f'\n  Generating publication-quality figures ...')
    plot_convergence(results, args.iter, out_dir)
    if args.runs >= 3:
        plot_boxplot(results, out_dir)
        plot_bar_errorbar(results, out_dir)
        plot_violin(results, out_dir)
    plot_assignment_map(results, out_dir)
    plot_radar(results, out_dir)
    plot_cost_breakdown(results, out_dir)
    plot_time_comparison(results, out_dir)

    print(f'\n✅ All done! Figures saved in: {os.path.abspath(out_dir)}')


if __name__ == '__main__':
    main()
