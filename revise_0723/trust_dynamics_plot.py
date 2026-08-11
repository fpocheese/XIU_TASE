#!/usr/bin/env python3
"""
Illustration for the reviewer response (Section IV.B trust mechanism).
Uses EXACTLY the paper's Eq. (42)/trust_update:
    T_i^{k+1} = (1 - a_T) * T_i^{k} + a_T * sigma( tau_T * Rtilde_i^{k} )
    sigma(x) = 1/(1+e^{-x})

The figure demonstrates that this update is a CONVERGENT exponential-smoothing
tracker (fixed point T* = sigma(tau_T * Rtilde)), NOT a monotonic accumulator.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12, "axes.linewidth": 0.9,
    "font.family": "serif", "mathtext.fontset": "cm",
})

def sigma(x):
    return 1.0 / (1.0 + np.exp(-x))

# ---- paper hyper-parameters ----
alpha_T = 0.10   # trust update rate (memory)
tau_T   = 2.0    # temperature (sensitivity)
K       = 120    # training episodes shown

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# =========================================================================
# Panel (a): trajectories T_i^{(k)} for several STATIONARY relative-perf levels
#            -> each converges to its fixed point sigma(tau_T*Rtilde), then STOPS.
# =========================================================================
Rtildes = [+2.0, +1.0, +0.5, 0.0, -1.0, -2.0]
colors  = plt.cm.coolwarm(np.linspace(1, 0, len(Rtildes)))
T0 = 0.5
for Rt, c in zip(Rtildes, colors):
    fp = sigma(tau_T * Rt)          # fixed point
    T = T0
    traj = [T]
    for _ in range(K):
        T = (1 - alpha_T) * T + alpha_T * fp
        traj.append(T)
    ax1.plot(range(K + 1), traj, color=c, lw=2,
             label=rf"$\widetilde{{G}}={Rt:+.1f}\;\Rightarrow\;\mathcal{{T}}^\star={fp:.2f}$")
    ax1.axhline(fp, color=c, ls=":", lw=1, alpha=0.7)

ax1.set_xlabel("Training episode $k$")
ax1.set_ylabel(r"Trust level $\mathcal{T}_i^{(k)}$")
ax1.set_title("(a) Trust converges to a fixed point (not monotone growth)")
ax1.set_ylim(-0.02, 1.02)
ax1.grid(alpha=0.3)
ax1.legend(fontsize=8.5, loc="center right", framealpha=0.9)

# =========================================================================
# Panel (b): the "high-trust stability" paradox.
#   A consistently above-average agent (Rtilde>0) does NOT ratchet to T=1.
#   And when the swarm converges (all agents equal), Rtilde->0 so T->0.5,
#   i.e. a residual exploration floor beta = 1 - T = 0.5 remains during training.
# =========================================================================
# Scenario: agent is strongly above baseline early, then swarm converges (Rtilde->0)
k = np.arange(K + 1)
Rt_traj = 2.0 * np.exp(-k / 25.0)     # relative advantage decays as swarm catches up
T = T0
traj = [T]
beta = [1 - T]
for kk in range(K):
    fp = sigma(tau_T * Rt_traj[kk])
    T = (1 - alpha_T) * T + alpha_T * fp
    traj.append(T)
    beta.append(1 - T)
traj = np.array(traj); beta = np.array(beta)

ax2.plot(k, traj, color="C3", lw=2.2, label=r"Trust $\mathcal{T}_i^{(k)}$")
ax2.plot(k, beta, color="C0", lw=2.2, ls="--",
         label=r"Exploration $\beta_i=1-\mathcal{T}_i$")
ax2.axhline(0.5, color="gray", ls=":", lw=1.2)
ax2.text(K * 0.52, 0.53, r"residual floor $\beta_i=0.5$ as $\widetilde{G}\to0$",
         fontsize=9, color="gray")
ax2.annotate("swarm converges\n" r"($\widetilde{G}_i\to0$)",
             xy=(95, sigma(0)), xytext=(60, 0.80),
             fontsize=9, ha="center",
             arrowprops=dict(arrowstyle="->", color="black", lw=1))
ax2.set_xlabel("Training episode $k$")
ax2.set_ylabel("Value")
ax2.set_title("(b) Relative baseline $\\Rightarrow$ trust self-centers at 0.5")
ax2.set_ylim(-0.02, 1.02)
ax2.grid(alpha=0.3)
ax2.legend(fontsize=9, loc="center right", framealpha=0.9)

plt.tight_layout()
plt.savefig("trust_dynamics_illustration.png", dpi=200, bbox_inches="tight")
plt.savefig("trust_dynamics_illustration.pdf", bbox_inches="tight")
print("saved trust_dynamics_illustration.png / .pdf")

# ---- print the numeric evidence used in the caption ----
print("\nFixed-point check  T* = sigma(tau_T * Rtilde),  alpha_T=%.2f, tau_T=%.1f" % (alpha_T, tau_T))
for Rt in Rtildes:
    fp = sigma(tau_T * Rt)
    T = T0
    for _ in range(500):
        T = (1 - alpha_T) * T + alpha_T * fp
    print(f"  Rtilde={Rt:+.1f}:  sigma={fp:.4f},  T_infty={T:.4f}  (|diff|={abs(fp-T):.2e})")
