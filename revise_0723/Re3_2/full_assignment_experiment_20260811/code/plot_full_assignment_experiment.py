#!/usr/bin/env python3
"""Generate the reviewer figure and LaTeX tables directly from experiment CSVs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fcol(rows, name):
    return np.array([float(r[name]) for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(); data = args.out_dir / "data"
    runtime = read(data / "runtime_summary.csv")
    delay = read(data / "static_delay_summary.csv")
    topology = read(data / "topology_summary.csv")
    dynamic = read(data / "dynamic_summary.csv")
    trace = read(data / "dynamic_trace_raw.csv")
    scaling = read(data / "scaling_summary.csv")

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8.2,
                         "axes.labelsize": 8.5, "legend.fontsize": 7.2,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "axes.linewidth": 0.8, "lines.linewidth": 1.5,
                         "lines.markersize": 4.5})
    blue, orange, green, purple = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
    fig, ax = plt.subplots(2, 2, figsize=(7.15, 5.15), constrained_layout=True)

    size = [r for r in runtime if r["sweep"] == "problem_size"]
    xs, ys = fcol(size, "M"), fcol(size, "runtime_s_mean")
    ss = np.polyfit(np.log(xs), np.log(ys), 1)[0]
    ax[0, 0].semilogy(xs, ys, "s-", color=orange,
                      label=rf"Joint $M$--$N$ sweep, slope={ss:.2f}")
    nominal = next(r for r in size if int(float(r["M"])) == 20)
    ax[0, 0].plot(20, float(nominal["runtime_s_mean"]), "*", color=blue,
                  markersize=9, label="Nominal $20\\times8$")
    ax[0, 0].set(xlabel="Number of defenders $M$ ($M/N=2.5$)",
                 ylabel="IDBO runtime (s)")
    ax[0, 0].legend(frameon=False, loc="upper left")

    xd = fcol(delay, "delay_ms"); yd = fcol(delay, "latency_s_mean")
    ed = fcol(delay, "latency_s_ci95")
    ax[0, 1].errorbar(xd, yd, yerr=ed, fmt="o-", capsize=2.5, color=green)
    ax[0, 1].set(xlabel="Additional per-hop delay (ms)",
                 ylabel="Static consensus latency (s)")

    topo = sorted(topology, key=lambda r: float(r["graph_diameter_D"]))
    xt = fcol(topo, "graph_diameter_D"); yt = fcol(topo, "latency_s_mean")
    ax[1, 0].plot(xt, yt, "^-", color=orange)
    ax[1, 0].set(xlabel="Communication-graph diameter $D$ (hops)",
                 ylabel="Consensus latency (s)")

    xq = fcol(dynamic, "delay_ms")
    jq, je = fcol(dynamic, "winner_jaccard_mean"), fcol(dynamic, "winner_jaccard_ci95")
    rr, re = fcol(dynamic, "recovery_rate_mean"), fcol(dynamic, "recovery_rate_ci95")
    ax[1, 1].errorbar(xq, jq, yerr=je, fmt="D-", capsize=2.5, color=purple,
                      label="Winner-set agreement")
    ax[1, 1].errorbar(xq, rr, yerr=re, fmt="s--", capsize=2.5, color=blue,
                      label="Recovery within 2-s interval")
    ax[1, 1].set(xlabel="Additional per-hop delay (ms)",
                 ylabel="Dynamic assignment metric", ylim=(-0.03, 1.08))
    ax[1, 1].legend(frameon=False, loc="lower left")

    for label, a in zip(["(a)", "(b)", "(c)", "(d)"], ax.flat):
        a.grid(True, alpha=0.22, linewidth=0.45)
        a.text(0.0, 1.02, label, transform=a.transAxes, va="bottom", fontweight="bold")
        a.tick_params(direction="in", top=True, right=True)
    figdir = args.out_dir / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    for ext, opts in [("pdf", {}), ("svg", {}), ("png", {"dpi": 600})]:
        fig.savefig(figdir / f"idbo_assignment_complexity_delay.{ext}",
                    bbox_inches="tight", **opts)
    plt.close(fig)

    fig2, bx = plt.subplots(1, 2, figsize=(7.15, 2.55), constrained_layout=True)
    colors = {0: blue, 100: green, 200: orange, 400: purple}
    for delay_ms in [0, 100, 200, 400]:
        subset = [r for r in trace if int(float(r["delay_ms"])) == delay_ms
                  and int(float(r["epoch"])) > 0]
        offset = np.array([int(float(r["exchange"])) % 40 for r in subset])
        x = np.arange(40) * 0.05
        for axis, metric in [(bx[0], "winner_jaccard"),
                             (bx[1], "stale_record_fraction")]:
            means = np.array([np.mean([float(r[metric]) for r, o in zip(subset, offset) if o == k])
                              for k in range(40)])
            axis.plot(x, means, color=colors[delay_ms], label=rf"$\tau={delay_ms}$ ms")
    bx[0].set(xlabel="Time after reassignment update (s)",
              ylabel="Winner-set agreement", ylim=(0.35, 1.02))
    bx[1].set(xlabel="Time after reassignment update (s)",
              ylabel="Stale-record fraction", ylim=(-0.02, 1.02))
    for label, a in zip(["(a)", "(b)"], bx):
        a.grid(True, alpha=0.22, linewidth=0.45)
        a.tick_params(direction="in", top=True, right=True)
        a.text(0.0, 1.02, label, transform=a.transAxes, va="bottom", fontweight="bold")
    bx[0].legend(frameon=False, ncol=2, loc="lower right")
    for ext, opts in [("pdf", {}), ("svg", {}), ("png", {"dpi": 600})]:
        fig2.savefig(figdir / f"idbo_dynamic_consensus_tracking.{ext}",
                     bbox_inches="tight", **opts)
    plt.close(fig2)

    tables = args.out_dir / "tables"; tables.mkdir(parents=True, exist_ok=True)
    delay_lines = []
    for s, d in zip(delay, dynamic):
        delay_lines.append(
            f"{int(float(s['delay_ms']))} & "
            f"{float(s['latency_s_mean']):.2f} $\\pm$ {float(s['latency_s_std']):.2f} & "
            f"100 & {float(d['winner_jaccard_mean']):.3f} $\\pm$ {float(d['winner_jaccard_std']):.3f} & "
            f"{100*float(d['recovery_rate_mean']):.1f} \\\\")
    delay_tex = r"""\begin{table}[!t]
\centering
\footnotesize
\setlength{\tabcolsep}{1.8pt}
\caption{Effect of communication delay on the distributed IDBO assignment.}
\label{tab:idbo_delay_scalability}
\begin{tabular}{@{}ccccc@{}}
\hline\hline
\shortstack{Add. delay\\(ms)} & \shortstack{Static time\\(s)} &
\shortstack{Fixed point\\(\%)} & \shortstack{Dynamic\\agreement} &
\shortstack{Recovery\\(\%)} \\
\hline
""" + "\n".join(delay_lines) + r"""
\hline
\end{tabular}
\end{table}
"""
    (tables / "table_delay_dynamic.tex").write_text(delay_tex, encoding="utf-8")

    scale_lines = []
    for rt, sc in zip(size, scaling):
        nominal_open = "\\textbf{" if int(float(sc["M"])) == 20 else ""
        nominal_close = "}" if nominal_open else ""
        scale_lines.append(
            f"{nominal_open}{int(float(sc['M']))}$\\times${int(float(sc['N']))}{nominal_close} & "
            f"{float(rt['runtime_s_mean']):.3f} $\\pm$ {float(rt['runtime_s_std']):.3f} & "
            f"{int(float(sc['graph_diameter_D']))} & {float(sc['latency_s_mean']):.2f} & "
            f"{float(sc['messages_mean'])/1000:.1f} \\\\")
    scale_tex = r"""\begin{table}[!t]
\centering
\small
\setlength{\tabcolsep}{3.5pt}
\caption{Computational and communication scalability of the IDBO assignment.}
\label{tab:idbo_assignment_scale}
\begin{tabular}{ccccc}
\hline\hline
$M\!\times\!N$ & IDBO time (s) & $D$ & Consensus (s) & Messages ($10^3$) \\
\hline
""" + "\n".join(scale_lines) + r"""
\hline
\end{tabular}
\end{table}
"""
    (tables / "table_scalability.tex").write_text(scale_tex, encoding="utf-8")


if __name__ == "__main__":
    main()
