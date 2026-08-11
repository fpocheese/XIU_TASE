"""
trust_dynamics.py -- Reviewer-response figure for comment Re2_3.
================================================================
Reviewer's concern: Section IV.B text reads like a *monotonic incremental* trust
process, while Eq. (42) implements a *convergent smoothing* process -- an apparent
logical inconsistency for "maintaining high-trust stability."

We take Eq. (42) as authoritative and illustrate that it is a first-order
exponential-smoothing (contraction) update, NOT a monotonic accumulator:

    T_i^{k+1} = (1 - alpha_T) * T_i^k + alpha_T * sigmoid(tau_T * Rtilde_i^k)      (42)

which is a contraction of modulus (1 - alpha_T) toward the bounded, moving
equilibrium  T* = sigmoid(tau_T * Rtilde)  in (0,1).

Three panels:
  (a) Sustained above-average performance: Eq. (42) converges to and HOLDS a stable
      high-trust equilibrium (<1), whereas a monotonic-increment reading ratchets to
      the ceiling and pins there -- the behavior the reviewer flags as inconsistent.
  (b) Time-varying performance: trust smoothly tracks the moving equilibrium
      sigma(tau_T*Rtilde), rising AND falling -- impossible for a monotonic process.
  (c) Contraction: from any initial T^0 the recursion converges to a unique
      equilibrium; the deviation |T^k - T*| decays geometrically as (1-alpha_T)^k.

Outputs (vector + raster) to the Re2_3/ folder:
  trust_convergent_smoothing.pdf / .png
"""
import os
import numpy as np
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
    'axes.linewidth': 1.0, 'lines.linewidth': 1.8,
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, '..'))

# ---- illustrative parameters (consistent with Eq. (42); values are for exposition)
ALPHA_T = 0.12      # trust update (smoothing) rate  alpha_T in (0,1)
TAU_T = 2.0         # temperature tau_T > 0

C_EQ = '#1f77b4'    # convergent smoothing (Eq. 42)
C_MONO = '#d62728'  # monotonic-increment (mis)reading
C_TGT = '#2ca02c'   # moving equilibrium / target
C_GREY = '#7f7f7f'


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def trust_eq42(Rt, T0=0.5, alpha=ALPHA_T, tau=TAU_T):
    """Eq. (42): first-order exponential smoothing toward sigmoid target."""
    Rt = np.asarray(Rt, dtype=float)
    T = np.empty(len(Rt) + 1)
    T[0] = T0
    for k in range(len(Rt)):
        T[k + 1] = (1.0 - alpha) * T[k] + alpha * sigmoid(tau * Rt[k])
    return T


def trust_monotonic(Rt, T0=0.5, alpha=ALPHA_T, tau=TAU_T):
    """Monotonic-increment (mis)reading: accumulate only the 'above-0.5' surplus,
    never relax -- a ratchet that can only climb and saturates at the ceiling."""
    Rt = np.asarray(Rt, dtype=float)
    T = np.empty(len(Rt) + 1)
    T[0] = T0
    for k in range(len(Rt)):
        inc = alpha * (sigmoid(tau * Rt[k]) - 0.5)
        T[k + 1] = np.clip(T[k] + max(inc, 0.0), 0.0, 1.0)  # only-up ratchet
    return T
def main():
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))

    # ---------------------------------------------------------------
    # (a) Sustained above-average performance: converge & HOLD vs ratchet
    # ---------------------------------------------------------------
    K = 120
    Rt_const = 0.8 * np.ones(K)                      # steady above-average
    Rt_const += 0.05 * rng.standard_normal(K)        # mild episode noise
    T_eq = trust_eq42(Rt_const, T0=0.30)
    T_mono = trust_monotonic(Rt_const, T0=0.30)
    T_star = sigmoid(TAU_T * np.mean(Rt_const))

    ax = axes[0]
    kk = np.arange(K + 1)
    ax.axhline(T_star, color=C_TGT, ls='--', lw=1.4,
               label=r'equilibrium $\mathcal{T}^\star=\sigma(\tau_T\tilde{R})$')
    ax.plot(kk, T_eq, color=C_EQ,
            label=r'Eq.~(42): convergent smoothing')
    ax.plot(kk, T_mono, color=C_MONO, ls='-.',
            label=r'monotonic-increment reading')
    ax.axhspan(0.99, 1.0, color=C_MONO, alpha=0.10)
    ax.text(K * 0.52, 0.955, 'pinned at ceiling', color=C_MONO, fontsize=8.5)
    ax.set_xlabel('episode $k$')
    ax.set_ylabel(r'trust $\mathcal{T}_i^{(k)}$')
    ax.set_title('(a) sustained above-average return')
    ax.set_ylim(0.25, 1.03)
    ax.legend(loc='center right', framealpha=0.92)
    ax.grid(alpha=0.25)

    # ---------------------------------------------------------------
    # (b) Time-varying performance: trust tracks moving equilibrium (up AND down)
    # ---------------------------------------------------------------
    K2 = 260
    Rt = np.zeros(K2)
    Rt[:70] = 1.2          # strong
    Rt[70:130] = -1.0      # slump
    Rt[130:190] = 0.4      # recover, moderate
    Rt[190:] = -0.3        # mild under-performance
    Rt = Rt + 0.06 * rng.standard_normal(K2)
    T_eq2 = trust_eq42(Rt, T0=0.5)
    target = sigmoid(TAU_T * Rt)

    ax = axes[1]
    kk2 = np.arange(K2)
    ax.plot(kk2, target, color=C_TGT, ls='--', lw=1.3, alpha=0.9,
            label=r'moving target $\sigma(\tau_T\tilde{R}_i^{(k)})$')
    ax.plot(np.arange(K2 + 1), T_eq2, color=C_EQ,
            label=r'Eq.~(42): $\mathcal{T}_i^{(k)}$')
    ax.axhline(0.5, color=C_GREY, ls=':', lw=1.0)
    ax.annotate('rises', xy=(35, 0.78), color=C_EQ, fontsize=9)
    ax.annotate('falls', xy=(95, 0.30), color=C_EQ, fontsize=9)
    ax.set_xlabel('episode $k$')
    ax.set_ylabel(r'trust $\mathcal{T}_i^{(k)}$')
    ax.set_title('(b) time-varying return: smooth two-way tracking')
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc='upper right', framealpha=0.92)
    ax.grid(alpha=0.25)

    # ---------------------------------------------------------------
    # (c) Contraction: any T^0 -> unique equilibrium; geometric error decay
    # ---------------------------------------------------------------
    K3 = 90
    Rt_fix = 0.6 * np.ones(K3)
    T_star3 = sigmoid(TAU_T * 0.6)
    ax = axes[2]
    for T0 in [0.02, 0.25, 0.5, 0.75, 0.98]:
        Tk = trust_eq42(Rt_fix, T0=T0)
        ax.plot(np.arange(K3 + 1), Tk, color=C_EQ, alpha=0.55, lw=1.3)
    ax.axhline(T_star3, color=C_TGT, ls='--', lw=1.5,
               label=r'unique equilibrium $\mathcal{T}^\star$')
    # theoretical geometric envelope |T^k - T*| = (1-alpha)^k |T^0 - T*|
    kk3 = np.arange(K3 + 1)
    env = T_star3 + (0.98 - T_star3) * (1 - ALPHA_T) ** kk3
    ax.plot(kk3, env, color=C_MONO, ls=':', lw=1.5,
            label=r'$(1-\alpha_T)^k$ envelope')
    ax.set_xlabel('episode $k$')
    ax.set_ylabel(r'trust $\mathcal{T}_i^{(k)}$')
    ax.set_title(r'(c) contraction from any $\mathcal{T}_i^{(0)}$')
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc='center right', framealpha=0.92)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        p = os.path.join(OUT, f'trust_convergent_smoothing.{ext}')
        fig.savefig(p)
        print('wrote', p)


if __name__ == '__main__':
    main()

