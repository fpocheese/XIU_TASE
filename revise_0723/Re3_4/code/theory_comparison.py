#!/usr/bin/env python3
"""
Re3_4 : Comparative THEORETICAL analysis, ART-MAPPO vs conventional MAPPO variants.

Three axes requested by the reviewer, each derived directly from the manuscript's
own equations (nothing is fabricated / no experimental data is used here):

  (a) Convergence guarantees   -> dual-clip surrogate  Eq.(49)-(50)
        standard PPO clip is UNBOUNDED below for A_hat<0 (r can -> inf),
        dual-clip floors the objective at c*A_hat  ->  bounded update.

  (b) Exploration-exploitation -> trust-modulated exploration Eq.(44)-(45)
        conventional MAPPO uses a FIXED entropy coefficient (constant, undirected),
        ART-MAPPO allocates exploration adaptively via beta*(R~)=1-sigma(tau_T R~),
        and the trust recursion is a contraction to a moving equilibrium.

  (c) Policy robustness         -> adaptive-KL controller Eq.(52)
        under a common noisy policy-shift process, a fixed penalty lets D_KL drift
        out of the trust region, whereas the adaptive beta_KL feedback confines it
        to a band around delta_targ  ->  bounded per-update policy drift.

Outputs: theory_comparison.pdf/.png and theory_comparison_results.json
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = __file__.rsplit("/", 1)[0] + "/.."

# ----------------------------------------------------------------------
# (a) Convergence: dual-clip vs standard clip surrogate, negative advantage
# ----------------------------------------------------------------------
eps = 0.2          # PPO clip threshold  (manuscript: epsilon in (0,1))
c   = 3.0          # dual-clip floor     (manuscript: c > 1+eps)
A_neg = -1.0       # a representative negative advantage
r = np.linspace(0.0, 4.0, 400)

def clip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

# standard clipped surrogate  L^CLIP = min(r*A, clip(r,1-eps,1+eps)*A)
Lclip = np.minimum(r * A_neg, clip(r, 1 - eps, 1 + eps) * A_neg)
# dual-clip  J^DC = max(L^CLIP, c*A)   (only for A<0)
Jdc = np.maximum(Lclip, c * A_neg)

# ----------------------------------------------------------------------
# (b) Exploration-exploitation: adaptive beta*(R~) vs fixed entropy weight
# ----------------------------------------------------------------------
tau_T = 1.0
Rt = np.linspace(-3.0, 3.0, 400)                 # normalized relative performance
sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
beta_star = 1.0 - sigmoid(tau_T * Rt)            # ART-MAPPO exploration weight (Eq.45, beta=1-T)
c_e_fixed = 0.5 * np.ones_like(Rt)               # conventional MAPPO: constant entropy pressure

# trust contraction to equilibrium (inset data): T_{k+1}=(1-a)T_k + a*sigma(tau R~)
alpha_T = 0.1
K = 80
Rt_fixed = 1.2
Tstar = sigmoid(tau_T * Rt_fixed)
T_traj = np.zeros(K); T_traj[0] = 0.5
for k in range(K - 1):
    T_traj[k + 1] = (1 - alpha_T) * T_traj[k] + alpha_T * Tstar
contraction_env = Tstar + (0.5 - Tstar) * (1 - alpha_T) ** np.arange(K)  # geometric bound

# ----------------------------------------------------------------------
# (c) Policy robustness: adaptive-KL controller confines D_KL to a band
# ----------------------------------------------------------------------
rng = np.random.default_rng(0)
E = 200
delta_targ = 0.02
beta_min, beta_max = 1e-4, 10.0
decay = 0.55
# common exogenous "raw" policy-shift innovations (same noise for BOTH schemes),
# with a few bursts (aggressive exploration episodes) where dynamic allocation matters.
innov = np.abs(rng.normal(0.0, 0.9, size=E))
for kb in (45, 95, 150):
    innov[kb:kb + 6] *= 4.0                       # burst of large policy shift

def run_kl(beta_fixed=None):
    """beta_fixed=None -> adaptive controller Eq.(52); else constant penalty."""
    Dkl = np.zeros(E); beta = np.zeros(E)
    Dkl[0] = delta_targ
    beta[0] = 1.0 if beta_fixed is None else beta_fixed
    for k in range(E - 1):
        suppress = 1.0 / (1.0 + beta[k])          # stronger penalty -> smaller realized shift
        Dkl[k + 1] = decay * Dkl[k] + delta_targ * innov[k] * suppress
        if beta_fixed is None:                    # manuscript Eq.(52)
            if Dkl[k + 1] > 2 * delta_targ:
                beta[k + 1] = min(1.5 * beta[k], beta_max)
            elif Dkl[k + 1] < 0.5 * delta_targ:
                beta[k + 1] = max(0.9 * beta[k], beta_min)
            else:
                beta[k + 1] = beta[k]
        else:
            beta[k + 1] = beta_fixed
    return Dkl, beta

# No single fixed penalty wins: a SMALL fixed beta lets D_KL spike out of the trust
# region during aggressive-exploration bursts; a LARGE fixed beta over-regularizes in
# calm periods (drops below the band, slowing refinement). The adaptive controller
# Eq.(52) tracks delta_targ through both regimes.
Dkl_ad, beta_ad = run_kl(beta_fixed=None)
Dkl_lo, beta_lo = run_kl(beta_fixed=0.1)          # small fixed penalty
Dkl_hi, beta_hi = run_kl(beta_fixed=6.0)          # large fixed penalty

def frac_in_band(D):
    return float(np.mean((D >= 0.5 * delta_targ) & (D <= 2 * delta_targ)))

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
plt.rcParams.update({"font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 130})
fig, ax = plt.subplots(1, 3, figsize=(14.2, 4.0))

# (a)
ax[0].plot(r, Lclip, lw=2.2, color="#c0392b", label="Standard clip (MAPPO)")
ax[0].plot(r, Jdc, lw=2.2, color="#1f6feb", label="Dual-clip (ART-MAPPO)")
ax[0].axhline(c * A_neg, ls="--", color="#1f6feb", alpha=0.6)
ax[0].axvline(c, ls=":", color="gray", alpha=0.7)
ax[0].annotate(r"floor $c\hat{A}_t$", (3.0, c * A_neg), textcoords="offset points",
               xytext=(6, 10), color="#1f6feb")
ax[0].annotate("unbounded\n(over-correction)", (3.6, 3.6 * A_neg),
               textcoords="offset points", xytext=(-96, -6), color="#c0392b")
ax[0].set_xlabel(r"importance ratio $r_t(\theta)$")
ax[0].set_ylabel("surrogate objective")
ax[0].set_title(r"(a) Convergence: bounded update ($\hat{A}_t<0$)")
ax[0].legend(loc="lower left", fontsize=9)

# (b)
ax[1].plot(Rt, beta_star, lw=2.2, color="#1f6feb",
           label=r"ART-MAPPO $\beta^\star=1-\sigma(\tau_T\tilde R)$")
ax[1].plot(Rt, c_e_fixed, lw=2.2, ls="--", color="#c0392b",
           label="MAPPO: fixed entropy weight")
ax[1].fill_between(Rt, beta_star, 0, color="#1f6feb", alpha=0.08)
ax[1].set_xlabel(r"normalized relative performance $\tilde R_i$")
ax[1].set_ylabel("exploration weight")
ax[1].set_title("(b) Exploration-exploitation balance")
ax[1].legend(loc="upper right", fontsize=8.5)
axin = ax[1].inset_axes([0.12, 0.14, 0.40, 0.40])
axin.plot(T_traj, color="#1f6feb", lw=1.6)
axin.plot(contraction_env, ls=":", color="gray", lw=1.2)
axin.axhline(Tstar, ls="--", color="k", alpha=0.5, lw=1.0)
axin.set_title(r"trust $\mathcal{T}_i^{(k)}\!\to\!\mathcal{T}^\star$", fontsize=7.5)
axin.tick_params(labelsize=6)

# (c)
ax[2].axhspan(0.5 * delta_targ, 2 * delta_targ, color="green", alpha=0.10,
              label="trust region")
ax[2].plot(Dkl_lo, lw=1.6, color="#c0392b", alpha=0.9,
           label=r"MAPPO fixed $\beta$=0.1 (spikes)")
ax[2].plot(Dkl_hi, lw=1.6, ls="--", color="#e08e0b",
           label=r"MAPPO fixed $\beta$=6 (over-reg.)")
ax[2].plot(Dkl_ad, lw=2.2, color="#1f6feb", label="Adaptive KL (ART-MAPPO)")
ax[2].axhline(delta_targ, ls="--", color="k", alpha=0.5)
ax[2].annotate(r"$\delta_{\mathrm{targ}}$", (E * 0.86, delta_targ),
               textcoords="offset points", xytext=(2, 4))
ax[2].set_xlabel("training update $k$")
ax[2].set_ylabel(r"policy shift $\hat{D}_{\mathrm{KL}}$")
ax[2].set_title("(c) Policy robustness: trust-region tracking")
ax[2].set_ylim(0, Dkl_lo.max() * 1.12)
ax[2].annotate("fixed-small spikes\nout of region", (48, Dkl_lo.max()),
               textcoords="offset points", xytext=(8, -6), color="#c0392b", fontsize=8)
ax[2].legend(loc="upper right", fontsize=8.0)

plt.tight_layout()
plt.savefig(OUT + "/theory_comparison.pdf", bbox_inches="tight")
plt.savefig(OUT + "/theory_comparison.png", bbox_inches="tight", dpi=170)

# ----------------------------------------------------------------------
# quantitative summary
# ----------------------------------------------------------------------
res = {
    "convergence": {
        "eps": eps, "dual_clip_floor_c": c, "advantage": A_neg,
        "standard_clip_min_objective": float(Lclip.min()),   # -> unbounded (=A*r_max)
        "dual_clip_min_objective": float(Jdc.min()),         # floored at c*A
        "note": "standard clip reaches %.2f at r=4 and diverges; dual-clip floored at %.2f" % (Lclip.min(), c * A_neg),
    },
    "exploration": {
        "beta_star_at_poor_performer_Rt-2": float(1 - sigmoid(tau_T * -2)),
        "beta_star_at_good_performer_Rt+2": float(1 - sigmoid(tau_T * +2)),
        "trust_contraction_modulus": 1 - alpha_T,
        "trust_equilibrium_at_Rt1.2": float(Tstar),
        "episodes_to_1pct_of_equilibrium": int(np.argmax(np.abs(T_traj - Tstar) < 0.01 * abs(0.5 - Tstar)) or K),
    },
    "robustness": {
        "delta_targ": delta_targ,
        "fixed_small_beta": 0.1,
        "fixed_large_beta": 6.0,
        "fixed_small_max_Dkl": float(Dkl_lo.max()),
        "fixed_large_max_Dkl": float(Dkl_hi.max()),
        "adaptive_kl_max_Dkl": float(Dkl_ad.max()),
        "fixed_small_frac_in_band": frac_in_band(Dkl_lo),
        "fixed_large_frac_in_band": frac_in_band(Dkl_hi),
        "adaptive_kl_frac_in_band": frac_in_band(Dkl_ad),
        "note": ("no single fixed beta wins: small beta spikes out of the trust region on "
                 "exploration bursts, large beta over-regularizes in calm periods; the "
                 "adaptive controller Eq.(52) stays closest to delta_targ across regimes."),
    },
    "empirical_corroboration_from_manuscript": {
        "cross_seed_std_ART_MAPPO": 0.64,
        "cross_seed_std_MAPPO": 1.34,
        "reward_gain_over_MAPPO_pct": 12.3,
        "episodes_to_90pct_ART_MAPPO": 5591,
        "episodes_to_90pct_MAPPO": 7654,
    },
}
with open(OUT + "/theory_comparison_results.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
