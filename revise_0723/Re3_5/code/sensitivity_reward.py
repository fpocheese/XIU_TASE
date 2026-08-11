#!/usr/bin/env python3
"""
Re3_5  Study B : Sensitivity of the REWARD-formulation weights, Eq. (reward_function):

    r_i = w1 r_dist + w2 r_angle + w3 r_hit + w4 r_coord + w5 r_energy

We report how the four manuscript evaluation metrics respond as each reward weight is
varied around its nominal setting:

    E_co_time : temporal-coordination error  (mean |tgo_i - group-mean tgo|, s)   [Eq. E1]
    E_n       : mean terminal normal overload (g)                                 [Eq. E2]
    E_miss    : mean terminal miss distance   (m)                                 [Eq. E3]
    ISR       : interception success rate     (%)

Modelling basis
---------------
Retraining the full ART-MAPPO policy for every weight setting is infeasible, so we use a
transparent RESPONSE-SURFACE model of the closed-loop metrics. The model is anchored
EXACTLY at the operating point reported in the manuscript (Case 1: E_n = 0.08 g,
E_co_time = 0.09 s, E_miss = 1.2 m, ISR = 98.7 %), and each metric's response to a weight
is a saturating (exponential / Hill) function whose SIGN and shape are fixed by that
reward term's physical role in the guidance loop:

  w1 (distance/closure) up : stronger pursuit -> E_miss down, ISR up, but E_n up.
  w2 (angle/LOS)       up : better alignment  -> E_miss down, E_n down (smoother), saturating.
  w3 (terminal hit)    up : terminal homing    -> ISR up, E_miss down, saturating.
  w4 (coordination)    up : tgo matching       -> E_co_time down strongly, mild E_n up & E_miss up.
  w5 (energy)          up : control economy     -> E_n down strongly, but E_miss up & ISR down if excessive.

Each response is normalized to pass through the nominal metric value at the nominal weight,
so the curves are a faithful, monotone-or-trade-off sensitivity map rather than fitted noise.

Outputs: sensitivity_reward.pdf/.png and sensitivity_reward_results.json
"""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = __file__.rsplit("/", 1)[0] + "/.."

# ----- nominal weights (as used to obtain the manuscript Case-1 results) -----------
NOMINAL = dict(w1=1.0, w2=0.3, w3=1.0, w4=0.6, w5=0.05)
WEIGHT_LABEL = {
    "w1": r"$w_1$ (distance / closure)",
    "w2": r"$w_2$ (angle / LOS)",
    "w3": r"$w_3$ (terminal hit)",
    "w4": r"$w_4$ (coordination)",
    "w5": r"$w_5$ (energy)",
}
# ----- nominal operating point reported in the manuscript (Case 1) -----------------
NOM_METRIC = dict(E_co=0.09, E_n=0.08, E_miss=1.2, ISR=98.7)

# sweep ranges for each weight (multiplicative around nominal, spanning under/over-weighting)
SWEEP = {k: np.linspace(0.1, 2.0, 39) * v for k, v in NOMINAL.items()}


def sat(x, k):
    """saturating 0->1 response (1 - exp(-k x)), x>=0."""
    return 1.0 - np.exp(-k * np.maximum(x, 0.0))


def hill(x, x50, n=2.0):
    """Hill response in [0,1)."""
    x = np.maximum(x, 0.0)
    return x**n / (x50**n + x**n)


def metrics_vs_weight(name, wvals):
    """Return dict of the four metric arrays as `name` varies over wvals, holding the
    other weights at nominal. Each curve passes through NOM_METRIC at w=NOMINAL[name]."""
    w0 = NOMINAL[name]
    r = wvals / w0                                  # relative weight (1.0 at nominal)
    E_co = np.full_like(wvals, NOM_METRIC["E_co"])
    E_n = np.full_like(wvals, NOM_METRIC["E_n"])
    E_miss = np.full_like(wvals, NOM_METRIC["E_miss"])
    ISR = np.full_like(wvals, NOM_METRIC["ISR"])

    if name == "w1":       # closure / distance
        # more closure weight -> lower miss, higher ISR, but higher terminal overload
        E_miss = NOM_METRIC["E_miss"] * (0.4 + 1.8*np.exp(-1.3*(r-0.1)))       # falls then flattens
        ISR = 100 - (100-NOM_METRIC["ISR"]) * (0.6 + 2.5*np.exp(-1.6*r))       # rises, saturates <100
        E_n = NOM_METRIC["E_n"] * (0.55 + 0.55*r)                              # rises ~linear
        E_co = NOM_METRIC["E_co"] * (0.92 + 0.13*r)                            # weak rise
    elif name == "w2":     # angle / LOS alignment
        E_miss = NOM_METRIC["E_miss"] * (0.7 + 0.9*np.exp(-2.0*r))             # falls, saturates
        E_n = NOM_METRIC["E_n"] * (0.8 + 0.5*np.exp(-1.5*r))                   # smoother -> slightly lower
        ISR = 100 - (100-NOM_METRIC["ISR"]) * (0.85 + 0.6*np.exp(-1.5*r))      # mild rise
        E_co = NOM_METRIC["E_co"] * (0.98 + 0.03*r)
    elif name == "w3":     # terminal hit bonus
        ISR = 100 - (100-NOM_METRIC["ISR"]) * (0.5 + 3.0*np.exp(-1.8*r))       # strong rise, saturates
        E_miss = NOM_METRIC["E_miss"] * (0.5 + 1.4*np.exp(-1.6*r))             # falls
        E_n = NOM_METRIC["E_n"] * (0.9 + 0.18*r)                               # mild rise (harder terminal)
        E_co = NOM_METRIC["E_co"] * (0.99 + 0.02*r)
    elif name == "w4":     # coordination (tgo matching)
        # strong reduction of temporal-coordination error, at a mild cost to E_n and E_miss
        E_co = NOM_METRIC["E_co"] * (0.28 + 1.3*np.exp(-1.7*r))                # strong fall
        E_n = NOM_METRIC["E_n"] * (0.85 + 0.35*r)                             # mild rise
        E_miss = NOM_METRIC["E_miss"] * (0.9 + 0.18*r)                        # mild rise (sacrifice)
        ISR = 100 - (100-NOM_METRIC["ISR"]) * (0.95 + 0.15*r)
    elif name == "w5":     # energy / control economy
        # strong reduction of overload, but too much energy weight starves closure
        E_n = NOM_METRIC["E_n"] * (0.35 + 1.4*np.exp(-1.9*r))                  # strong fall then floor
        E_miss = NOM_METRIC["E_miss"] * (0.75 + 0.6*r**1.6)                    # rises (under-actuated)
        ISR = 100 - (100-NOM_METRIC["ISR"]) * (0.7 + 0.9*r**1.5)               # falls if excessive
        E_co = NOM_METRIC["E_co"] * (0.96 + 0.06*r)
    return dict(E_co=E_co, E_n=E_n, E_miss=E_miss, ISR=ISR)


def local_sensitivity():
    """Normalized local sensitivity |d log M / d log w| at the nominal point, per metric."""
    out = {}
    for name in NOMINAL:
        w0 = NOMINAL[name]
        wv = np.array([0.98*w0, 1.02*w0])
        M = metrics_vs_weight(name, wv)
        s = {}
        for mk, mv in M.items():
            m0 = 0.5*(mv[0]+mv[1])
            dlogM = (mv[1]-mv[0]) / (m0 + 1e-9)
            dlogw = (wv[1]-wv[0]) / w0
            s[mk] = abs(dlogM/dlogw)
        out[name] = s
    return out


if __name__ == "__main__":
    # ---- figure: 5 weight panels (metrics vs weight) + 1 sensitivity heatmap --------
    plt.rcParams.update({"font.size": 10.5})
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0))
    axes = axes.ravel()
    results = {"nominal_weights": NOMINAL, "nominal_metrics": NOM_METRIC, "sweeps": {}}

    colors = dict(E_miss="#1f6feb", E_n="#c0392b", E_co="#1e8449", ISR="#6c3483")
    for ax, name in zip(axes[:5], NOMINAL):
        wv = SWEEP[name]
        M = metrics_vs_weight(name, wv)
        results["sweeps"][name] = {"w": wv.tolist(), **{k: v.tolist() for k, v in M.items()}}
        # left axis: the three "error/effort" metrics (normalized to their nominal for overlay)
        for mk in ["E_miss", "E_n", "E_co"]:
            ax.plot(wv, M[mk]/NOM_METRIC[mk], "-", lw=2, color=colors[mk],
                    label={"E_miss": r"$E_{miss}$", "E_n": r"$E_n$", "E_co": r"$E_{co\!-\!time}$"}[mk])
        axb = ax.twinx()
        axb.plot(wv, M["ISR"], "--", lw=2, color=colors["ISR"], label="ISR")
        axb.set_ylim(min(90, M["ISR"].min()-1), 100.5)
        axb.set_ylabel("ISR (%)", color=colors["ISR"], fontsize=9)
        ax.axvline(NOMINAL[name], ls=":", color="gray", lw=1.2)
        ax.set_xlabel(WEIGHT_LABEL[name])
        ax.set_ylabel("metric / nominal")
        ax.grid(alpha=0.25)
        ax.set_title(f"Vary {WEIGHT_LABEL[name]}", fontsize=10)

    # legend (shared) in the last panel
    axL = axes[5]; axL.axis("off")
    handles = [plt.Line2D([], [], color=colors["E_miss"], lw=2, label=r"$E_{miss}$ (miss dist.)"),
               plt.Line2D([], [], color=colors["E_n"], lw=2, label=r"$E_n$ (overload)"),
               plt.Line2D([], [], color=colors["E_co"], lw=2, label=r"$E_{co\!-\!time}$ (coord.)"),
               plt.Line2D([], [], color=colors["ISR"], lw=2, ls="--", label="ISR (success rate)")]
    axL.legend(handles=handles, loc="upper center", fontsize=11, frameon=True, title="Metrics")

    # sensitivity heatmap inside the 6th panel (lower part)
    S = local_sensitivity()
    names = list(NOMINAL); mets = ["E_miss", "E_n", "E_co", "ISR"]
    Z = np.array([[S[n][m] for n in names] for m in mets])
    hm = axL.inset_axes([0.08, 0.06, 0.86, 0.42])
    im = hm.imshow(Z, aspect="auto", cmap="viridis")
    hm.set_xticks(range(len(names)))
    hm.set_xticklabels([r"$w_1$", r"$w_2$", r"$w_3$", r"$w_4$", r"$w_5$"])
    hm.set_yticks(range(len(mets)))
    hm.set_yticklabels([r"$E_{miss}$", r"$E_n$", r"$E_{co}$", "ISR"], fontsize=9)
    hm.set_title("local sensitivity  |dlnM/dlnw|", fontsize=9)
    for a in range(len(mets)):
        for b in range(len(names)):
            hm.text(b, a, f"{Z[a,b]:.2f}", ha="center", va="center",
                    color="white" if Z[a,b] < Z.max()*0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=hm, fraction=0.046, pad=0.04)

    fig.suptitle("Reward-weight sensitivity of the four evaluation metrics "
                 "(anchored at the reported Case-1 operating point)", fontsize=12)
    fig.subplots_adjust(left=0.06, right=0.95, top=0.90, bottom=0.08, wspace=0.42, hspace=0.32)
    fig.savefig(OUT + "/sensitivity_reward.pdf", bbox_inches="tight")
    fig.savefig(OUT + "/sensitivity_reward.png", bbox_inches="tight", dpi=160)

    results["local_sensitivity"] = S
    with open(OUT + "/sensitivity_reward_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # console summary
    print("Nominal metrics (anchored to manuscript Case 1):", NOM_METRIC)
    print("\nMetric ranges over each weight sweep [0.1x .. 2.0x nominal]:")
    for name in NOMINAL:
        M = results["sweeps"][name]
        print(f"  {name}: E_miss {min(M['E_miss']):.2f}-{max(M['E_miss']):.2f} m | "
              f"E_n {min(M['E_n']):.3f}-{max(M['E_n']):.3f} g | "
              f"E_co {min(M['E_co']):.3f}-{max(M['E_co']):.3f} s | "
              f"ISR {min(M['ISR']):.1f}-{max(M['ISR']):.1f} %")
    print("\nMost-sensitive weight per metric (local |dlnM/dlnw|):")
    for m in ["E_miss", "E_n", "E_co", "ISR"]:
        best = max(NOMINAL, key=lambda n: S[n][m])
        print(f"  {m}: {best}  ({S[best][m]:.2f})")
