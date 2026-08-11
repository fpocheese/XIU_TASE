#!/usr/bin/env python3
"""IEEE-style reward-weight sensitivity figure from the REAL reward-composition data.

Plot-only: reads the archived reward_sensitivity_data.csv (100 real frozen-policy
episodes: 50 Case-1 + 50 Case-2; each reward weight scaled by
s in {0.50,0.75,1.00,1.25,1.50}) and the audit JSON with the per-weight sensitivity
indices. Nothing is recomputed, smoothed, or retrained.

The quantity plotted is the normalized variation of the cumulative training reward when
one weight is scaled and the rest are held at nominal:
  (a) all four weights on a common axis -- the sparse terminal-hit weight w_hit sets the
      overall reward scale (near-linear, |variation| up to ~0.5 at +/-50%);
  (b) the three dense shaping weights (coordination, energy, distance-shaping alpha_d)
      magnified -- their cumulative-reward variation stays below ~1e-3, i.e. the learning
      objective is insensitive to shaping-weight choice over +/-50%.

Output: Re3_5/sensitivity_reward_composition.pdf / .png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                          # Re3_5/
REV = ROOT.parent                           # revise_0723/
CSV = REV / "reward_sensitivity_data.csv"
AUDIT = REV / "reward_sensitivity_audit.json"

PARAMS = ["w_hit", "w_coord", "w_energy", "alpha_dist"]
LABEL = {
    "w_hit": r"$w_3$ (terminal hit)",
    "w_coord": r"$w_4$ (coordination)",
    "w_energy": r"$w_5$ (energy)",
    "alpha_dist": r"$\alpha_d$ (distance shaping)",
}
COLOR = {"w_hit": "#0072B2", "w_coord": "#D55E00", "w_energy": "#009E73", "alpha_dist": "#7030A0"}
MARK = {"w_hit": "o", "w_coord": "s", "w_energy": "D", "alpha_dist": "^"}
STYLE = {"w_hit": "-", "w_coord": "--", "w_energy": "-.", "alpha_dist": ":"}


def ieee_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Tinos", "Times", "Nimbus Roman"],
        "mathtext.fontset": "stix",
        "font.size": 8.0, "axes.labelsize": 8.5, "axes.linewidth": 0.8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.8,
        "lines.linewidth": 1.3, "xtick.direction": "in", "ytick.direction": "in",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def main() -> None:
    d = pd.read_csv(CSV)
    ts = d[d.data_type == "trajectory_summary"].copy()
    audit = json.loads(AUDIT.read_text())
    sidx = audit["sensitivity_index"]

    ieee_style()
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(3.5, 3.7))

    # ---- panel (a): all four weights on a common axis -----------------------------
    for p in PARAMS:
        sub = ts[ts.parameter == p].sort_values("scaling")
        s = sub.scaling.to_numpy(float)
        mean = sub["mean"].to_numpy(float)
        ci = sub.ci95.to_numpy(float)
        axa.plot(s, mean, STYLE[p], marker=MARK[p], color=COLOR[p], ms=3.0,
                 mfc="white", mew=0.9, label=LABEL[p])
        axa.fill_between(s, mean - ci, mean + ci, color=COLOR[p], alpha=0.18, lw=0)
    axa.axhline(0.0, color="0.4", lw=0.7, ls="--")
    axa.axvline(1.0, color="0.55", lw=0.7, ls=":")
    axa.set_ylabel("normalized cumulative-\nreward variation")
    axa.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5])
    axa.grid(True, color="0.85", lw=0.4, ls="--", dashes=(2.5, 2.5))
    axa.set_axisbelow(True)
    axa.text(0.03, 0.06, "(a)", transform=axa.transAxes, fontsize=9, fontweight="bold")
    axa.legend(loc="upper left", frameon=False, ncol=1, handlelength=2.4,
               labelspacing=0.25, borderpad=0.2)

    # ---- panel (b): magnified shaping weights (exclude the dominant w_hit) ---------
    shaping = ["w_coord", "alpha_dist", "w_energy"]
    for p in shaping:
        sub = ts[ts.parameter == p].sort_values("scaling")
        s = sub.scaling.to_numpy(float)
        mean = sub["mean"].to_numpy(float) * 1e3          # x10^-3 units
        ci = sub.ci95.to_numpy(float) * 1e3
        axb.plot(s, mean, STYLE[p], marker=MARK[p], color=COLOR[p], ms=3.0,
                 mfc="white", mew=0.9, label=LABEL[p])
        axb.fill_between(s, mean - ci, mean + ci, color=COLOR[p], alpha=0.18, lw=0)
    axb.axhline(0.0, color="0.4", lw=0.7, ls="--")
    axb.axvline(1.0, color="0.55", lw=0.7, ls=":")
    axb.set_xlabel(r"weight scaling factor $s = w_k / w_k^{\mathrm{nom}}$")
    axb.set_ylabel(r"variation ($\times 10^{-3}$)," + "\nshaping weights only")
    axb.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5])
    axb.grid(True, color="0.85", lw=0.4, ls="--", dashes=(2.5, 2.5))
    axb.set_axisbelow(True)
    axb.text(0.03, 0.06, "(b)", transform=axb.transAxes, fontsize=9, fontweight="bold")
    axb.legend(loc="upper right", frameon=False, ncol=1, handlelength=2.4,
               labelspacing=0.25, borderpad=0.2)

    fig.align_ylabels([axa, axb])
    fig.subplots_adjust(left=0.19, right=0.975, bottom=0.11, top=0.985, hspace=0.28)
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"sensitivity_reward_composition.{ext}",
                    dpi=600 if ext == "png" else None, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print("[ok] wrote sensitivity_reward_composition.pdf/.png")
    print(f"     scaling points: {sorted(ts.scaling.unique())}  (100 episodes: 50 C1 + 50 C2)")
    print("     sensitivity index (|dvar/ds| normalized):")
    for p in PARAMS:
        print(f"       {p:11s}: {sidx[p]:.3e}")


if __name__ == "__main__":
    main()
