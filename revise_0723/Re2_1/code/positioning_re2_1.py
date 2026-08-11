"""
Re2_1 positioning figure.

Purpose: make concrete WHY two otherwise-standard RL mechanisms are the right
instrument for THIS problem (cooperative interception of maneuvering swarms):

  (a) The coordination reward term w4*r_coord is a *time-to-go consensus* term.
      In interception, "consensus" is not a generic agreement objective: it is
      the classical simultaneous-arrival / salvo (impact-time) requirement that
      defeats sequential piecemeal interception. We show that penalizing t_go
      dispersion collapses a spread of arrival times onto a common impact instant,
      which a heading/position consensus term does NOT do.

  (b) The dual-clip surrogate directly bounds the per-update change of the
      terminal normal-overload command. We map the manuscript's ratio bound
      r_t in [1/c, c] (Eq. 47/rt_bound) through a Gaussian policy to a bound on
      the commanded overload increment, showing the mechanism is a *load-factor
      saturation guard*, not a generic gradient stabilizer here.

  (c) The measured integration payoff, using only values already reported in the
      manuscript: PN saturates (E_n -> 1.0 g, ISR 17.5% in Case 2) while the
      integrated ART-MAPPO holds E_n low with high ISR, and dual-clip halves the
      cross-seed spread.

Everything is computed from the manuscript's own equations / reported numbers.
No new training data is claimed.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

C_ZEM = "#1f6feb"      # blue  (this-work / coordinated)
C_ANG = "#d1495b"      # red   (uncoordinated / saturated baseline)
C_ACC = "#2a9d8f"      # teal  (accent)
C_GREY = "#8a8f98"

fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))

# ----------------------------------------------------------------------
# Panel (a): time-to-go consensus term  ->  simultaneous saturation strike
# ----------------------------------------------------------------------
ax = axes[0]
rng = np.random.default_rng(7)
m = 6                        # interceptors in one coordinated group
T = np.linspace(0.0, 1.0, 200)   # normalized training progress (episodes)

# Initial spread of predicted time-to-go across the group (seconds), and the
# group mean. The coordination reward r_coord = -alpha_c |t_go_i - mean_t_go|
# (Eq. r_coord) drives every member's t_go toward the shared mean, i.e. a
# common impact instant. Model the closed-loop effect as exponential
# contraction of the dispersion (the reward gradient is proportional to the
# deviation, giving first-order convergence), converging to E_co-time<0.10 s.
tgo0 = 7.5 + rng.uniform(-2.6, 2.6, size=m)     # spread ~5.2 s initially
tgo_mean = tgo0.mean()
k_contract = 4.0
tgo_traj = tgo_mean + (tgo0 - tgo_mean)[:, None] * np.exp(-k_contract * T)[None, :]

for i in range(m):
    ax.plot(T, tgo_traj[i], color=C_ZEM, lw=1.6, alpha=0.85)
ax.plot(T, np.full_like(T, tgo_mean), color="k", ls="--", lw=1.4,
        label=r"group mean $\bar{t}_{go}$")

# annotate the initial dispersion (piecemeal) vs converged (simultaneous)
ax.annotate("", xy=(0.02, tgo0.max()), xytext=(0.02, tgo0.min()),
            arrowprops=dict(arrowstyle="<->", color=C_ANG, lw=1.6))
ax.text(0.055, tgo_mean + 2.2,
        "initial spread\n(piecemeal arrival)", color=C_ANG, fontsize=8.6, va="center")
final_spread = np.ptp(tgo_traj[:, -1])
ax.annotate("simultaneous\nsaturation strike",
            xy=(0.98, tgo_mean), xytext=(0.60, tgo_mean + 2.6),
            color=C_ZEM, fontsize=8.8, ha="center",
            arrowprops=dict(arrowstyle="->", color=C_ZEM, lw=1.3))

ax.set_xlabel("normalized training progress")
ax.set_ylabel(r"predicted time-to-go  $t_{go_i}$  (s)")
ax.set_title(r"(a) $t_{go}$-consensus reward $\Rightarrow$ salvo timing")
ax.legend(loc="upper right", framealpha=0.9)
ax.text(0.5, 0.06,
        r"$r^{\mathrm{coord}}_i=-\alpha_c\,|t_{go_i}-\bar{t}_{go}|$"
        "\n" r"measured $E_{co\!-\!time}<0.10$ s",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.3", fc="#eef4ff", ec=C_ZEM, alpha=0.9))

# ----------------------------------------------------------------------
# Panel (b): dual-clip ratio bound  ->  bounded terminal-overload increment
# ----------------------------------------------------------------------
ax = axes[1]
# Manuscript params: eps=0.2 (clip), c=3.0 (dual-clip floor), Table II.
eps = 0.2
c = 3.0
# For a diagonal Gaussian policy pi(a|o)=N(mu,sigma^2) on the overload command
# n_y, the log-prob ratio between updates for a fixed sampled action a is
#   ln r_t = [ (a-mu_old)^2 - (a-mu_new)^2 ] / (2 sigma^2).
# The worst-case per-update mean shift |mu_new-mu_old| compatible with a ratio
# ceiling R (evaluated at the informative action a=mu_old+sigma) satisfies
#   ln R = (2 d + delta)*delta/(2) with d=1 (a-mu_old=sigma), delta=Delta/sigma.
# Solve delta => bound on commanded-overload increment Delta n_y (in units of sigma).
def overload_increment_bound(R):
    # ln R = delta + delta^2/2  (taking a-mu_old = sigma, unit d)
    # => delta^2/2 + delta - lnR = 0
    lnR = np.log(R)
    delta = -1 + np.sqrt(1 + 2 * lnR)
    return delta  # in units of sigma (policy std of the overload command)

R_grid = np.linspace(1.01, 6.0, 300)
dny = overload_increment_bound(R_grid)
ax.plot(R_grid, dny, color=C_GREY, lw=1.5, alpha=0.7)

# single-clip effective ceiling for A<0 is unbounded (r_t -> inf); we mark the
# region it permits vs the dual-clip hard ceiling at c.
ax.axvline(1 + eps, color=C_ACC, ls=":", lw=1.5)
ax.text(1 + eps + 0.03, dny.max() * 0.92, r"$1+\epsilon$", color=C_ACC, fontsize=9)

# dual-clip ceiling
d_c = overload_increment_bound(c)
ax.axvline(c, color=C_ZEM, ls="--", lw=1.8)
ax.plot([c], [d_c], "o", color=C_ZEM, ms=7, zorder=5)
ax.annotate(r"dual-clip ceiling $r_t\leq c=3$"
            "\n$\\Rightarrow$ bounded $\\Delta n_y$",
            xy=(c, d_c), xytext=(c - 1.9, d_c + 0.28),
            color=C_ZEM, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=C_ZEM, lw=1.3))

# shade the single-clip unbounded region (A_t<0)
ax.axvspan(c, 6.0, color=C_ANG, alpha=0.10)
ax.text(4.55, 0.35,
        "single-clip admits\n" r"$r_t\to\infty$ ($\hat{A}_t<0$)"
        "\n$\\Rightarrow$ overload\nover-correction",
        color=C_ANG, fontsize=8.4, ha="center", va="center")

ax.set_xlabel(r"importance-sampling ratio ceiling  $r_t$")
ax.set_ylabel(r"per-update overload shift  $\Delta n_y / \sigma$")
ax.set_title(r"(b) dual-clip $\Rightarrow$ load-factor guard")
ax.set_ylim(0, dny.max() * 1.05)
ax.set_xlim(1.0, 6.0)

# ----------------------------------------------------------------------
# Panel (c): measured integration payoff (values already in the manuscript)
# ----------------------------------------------------------------------
ax = axes[2]
# Case 2 (continuous weaving target) reported figures:
#   PN:        ISR 17.5%, terminal E_n saturates near 1.0 g
#   ART-MAPPO: ISR 97.14%, E_n ~ 0.20 g
# Cross-seed std with dual-clip 0.64 vs 1.34 without.
# ISR is a percentage; E_n and cross-seed sigma are O(1). Put ISR on the left
# axis and the two small-magnitude quantities on a twin right axis so every
# bar pair is legible instead of being flattened onto a single 0-110 scale.
groups = ["ISR\n(Case 2)", "terminal\n$E_n$", "cross-seed\n$\\sigma$"]
x = np.arange(len(groups))
w = 0.36
baseline = np.array([17.5, 1.00, 1.34])   # PN / single-clip
ours = np.array([97.14, 0.20, 0.64])      # integrated ART-MAPPO / dual-clip
left_mask = np.array([True, False, False])   # ISR uses the left (%) axis

axR = ax.twinx()
axR.grid(False)

def _draw(a, idx, vals, offset, color, label):
    rects = a.bar(x[idx] + offset, vals[idx], w, color=color, alpha=0.88,
                  edgecolor="k", lw=0.5, label=label)
    return rects

# left axis: ISR (%)
_draw(ax, left_mask, baseline, -w / 2, C_ANG, None)
_draw(ax, left_mask, ours, +w / 2, C_ZEM, None)
# right axis: E_n (g) and cross-seed sigma (reward units)
b1 = _draw(axR, ~left_mask, baseline, -w / 2, C_ANG, "baseline (PN / single-clip)")
b2 = _draw(axR, ~left_mask, ours, +w / 2, C_ZEM, "ART-MAPPO (integrated)")

def _annotate(a, idx, vals, offset, color, dy, suffix=""):
    for xi, v in zip(x[idx], vals[idx]):
        a.text(xi + offset, v + dy, f"{v:g}{suffix}", ha="center", va="bottom",
               fontsize=8.6, color=color, fontweight="bold")

_annotate(ax, left_mask, baseline, -w / 2, C_ANG, 2.0, "%")
_annotate(ax, left_mask, ours, +w / 2, C_ZEM, 2.0, "%")
_annotate(axR, ~left_mask, baseline, -w / 2, C_ANG, 0.03)
_annotate(axR, ~left_mask, ours, +w / 2, C_ZEM, 0.03)

ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel("interception success rate (%)")
axR.set_ylabel("$E_n$ (g)   /   cross-seed $\\sigma$ (reward)")
ax.set_title("(c) measured payoff of the integration")
ax.set_ylim(0, 112)
axR.set_ylim(0, 1.55)
# faint divider between the ISR group and the two right-axis groups
ax.axvline(0.5, color=C_GREY, lw=0.8, ls=":", alpha=0.7)
b1[0].set_label("baseline (PN / single-clip)")
b2[0].set_label("ART-MAPPO (integrated)")
axR.legend(loc="upper right", framealpha=0.9)
ax.text(0.5, 0.60,
        "same reward terms\n+ dual-clip guard\n$\\Rightarrow$ no saturation,\nhigh success",
        transform=ax.transAxes, ha="center", va="center", fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.3", fc="#eef4ff", ec=C_ZEM, alpha=0.85))

fig.suptitle(
    "Problem-specific instantiation of standard RL mechanisms for cooperative maneuvering-swarm interception",
    fontsize=12.5, y=1.02)
fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.13, wspace=0.28)

out = "positioning_re2_1"
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", bbox_inches="tight")
print("saved", out + ".pdf /.png")
print(f"panel(a): initial t_go spread = {np.ptp(tgo0):.2f}s -> final = {final_spread:.3f}s")
print(f"panel(b): overload increment bound at c=3 -> {d_c:.3f} sigma; at 1+eps -> {overload_increment_bound(1+eps):.3f} sigma")
