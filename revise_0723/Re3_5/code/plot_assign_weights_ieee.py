#!/usr/bin/env python3
"""IEEE-style redraw of the REAL assignment-probability weight sweep (Re3_5, Study A).

Plot-only: reads Re3_5/sensitivity_block_prob_results.json (produced by
sensitivity_block_prob.py, 21-point w1 sweep x 12 randomized geometries on real 3-D
kinematics) and never recomputes or smooths the numbers.

Two stacked single-column panels:
  (a) mean selected ZEM (m) and mean 3-D angular error (deg) of the selected
      interceptor-target pairs vs w1  (the tactical trade-off);
  (b) expected number of surviving targets J vs w1  (aggregate effectiveness).
The nominal w1 = 0.55 (min-max balance of the two normalized tactical costs, an actual
sweep point) is marked with a vertical dashed line in both panels.

Output: Re3_5/sensitivity_assign_weights.pdf / .png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # Re3_5/
RES = ROOT / "sensitivity_block_prob_results.json"
NOMINAL_W1 = 0.55


def ieee_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Tinos", "Times", "Nimbus Roman"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "lines.linewidth": 1.3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def main() -> None:
    R = json.loads(RES.read_text())
    w = np.asarray(R["w1_grid"], dtype=float)
    miss = np.asarray(R["metrics_mean"]["miss"], dtype=float)
    aspect = np.asarray(R["metrics_mean"]["aspect"], dtype=float)
    J = np.asarray(R["metrics_mean"]["J"], dtype=float)

    ieee_style()
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(3.5, 3.7), sharex=True)

    # ---- panel (a): tactical trade-off (twin axis) --------------------------------
    c_miss, c_asp = "#0072B2", "#D55E00"
    la = axa.plot(w, miss, "-o", color=c_miss, ms=3.0, mfc="white", mew=0.9,
                  label="mean selected ZEM")
    axa.set_ylabel("mean selected ZEM (m)", color=c_miss)
    axa.tick_params(axis="y", colors=c_miss)
    axa.set_ylim(600, 1200)

    axat = axa.twinx()
    lb = axat.plot(w, aspect, "--s", color=c_asp, ms=3.0, mfc="white", mew=0.9,
                   label="mean 3-D angular error")
    axat.set_ylabel("mean 3-D angular\nerror (deg)", color=c_asp)
    axat.tick_params(axis="y", colors=c_asp)
    axat.set_ylim(55, 100)

    axa.axvline(NOMINAL_W1, ls=":", color="0.35", lw=1.0)
    axa.grid(True, color="0.85", lw=0.4, ls="--", dashes=(2.5, 2.5))
    axa.set_axisbelow(True)
    axa.text(0.03, 0.06, "(a)", transform=axa.transAxes, fontsize=9, fontweight="bold")
    lns = la + lb
    axa.legend(lns, [l.get_label() for l in lns], loc="upper center",
               frameon=False, ncol=1, handlelength=2.2)

    # ---- panel (b): aggregate effectiveness ---------------------------------------
    c_J = "#7030A0"
    axb.plot(w, J, "-^", color=c_J, ms=3.2, mfc="white", mew=0.9,
             label=r"expected surviving targets $J$")
    axb.axvline(NOMINAL_W1, ls=":", color="0.35", lw=1.0)
    axb.set_xlabel(r"assignment weight $w_1$ (ZEM priority),  $w_2 = 1-w_1$")
    axb.set_ylabel(r"expected surviving")
    axb.set_ylim(0.0, 0.22)
    axb.grid(True, color="0.85", lw=0.4, ls="--", dashes=(2.5, 2.5))
    axb.set_axisbelow(True)
    axb.text(0.03, 0.88, "(b)", transform=axb.transAxes, fontsize=9, fontweight="bold")
    axb.legend(loc="lower left", frameon=False, handlelength=2.2)
    # annotate the nominal marker once
    axb.annotate(r"nominal $w_1{=}0.55$", xy=(NOMINAL_W1, 0.02),
                 xytext=(NOMINAL_W1 + 0.02, 0.045), fontsize=6.8, color="0.30")

    axb.set_ylabel(r"targets $J$ (out of $8$)")
    axb.set_xlim(-0.02, 1.02)

    fig.align_ylabels([axa, axb])
    fig.subplots_adjust(left=0.155, right=0.845, bottom=0.115, top=0.985, hspace=0.12)
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"sensitivity_assign_weights.{ext}",
                    dpi=600 if ext == "png" else None, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # console summary of the exact numbers the paper/letter will cite
    drop = (miss[0] - miss[-1]) / miss[0] * 100
    print(f"[ok] wrote sensitivity_assign_weights.pdf/.png")
    print(f"     w1 grid: {w[0]:.2f}:{w[1]-w[0]:.2f}:{w[-1]:.2f}  ({len(w)} points)")
    print(f"     mean ZEM {miss[0]:.0f} -> {miss[-1]:.0f} m  (drop {drop:.1f}%)")
    print(f"     mean 3-D angular error {aspect[0]:.1f} -> {aspect[-1]:.1f} deg")
    print(f"     J in [{J.min():.3f}, {J.max():.3f}] of 8  -> neutralization "
          f"{(1-J.max()/8)*100:.1f}%..{(1-J.min()/8)*100:.1f}%")
    print(f"     J(0.55) = {J[np.isclose(w,0.55)][0]:.3f}")


if __name__ == "__main__":
    main()
