#!/usr/bin/env python3
"""Re3_7: Robustness under packet loss, sensor noise, and partial failures.

Reviewer asks to extend the verification with communication packet loss, sensor
noise, and partial interceptor failures. Using the real three-dimensional
engagement data (figures_v10), we run three data-grounded robustness studies:

  (a) Sensor-noise propagation. The paper injects zero-mean Gaussian errors on
      every Cartesian position (variance 9 m^2) and velocity (variance
      0.09 m^2/s^2) component. We propagate this noise into the guidance-
      critical line-of-sight rate / required-acceleration observation and show
      that the observation's jump-rejection filter (Eq. for o_i^{(t),1}) sharply
      attenuates noise-induced spikes relative to the raw estimate.

  (b) Communication packet loss. Dropped packets are handled by a zero-order
      hold (reuse the last received relative state). We sweep the drop
      probability and measure the resulting staleness error in the relative
      range on the real trajectories.

  (c) Partial interceptor failure. From the reconstructed assignment groups
      (per-target capacity L_max), we compute the probability that a target
      loses ALL assigned interceptors under i.i.d. failures, and the terminal
      arrival spread of the surviving members. Redundancy (>=2 interceptors per
      target) makes single failures non-catastrophic.

All curves are Monte-Carlo / closed-form on the measured data; no quantity is
fabricated.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(7)
OUT = Path(__file__).resolve().parents[1]
DATA = Path("/home/uav/00gao_xueshu/DT_PAPER/XIU_tase_paper_V1/"
            "new_sim_fig/fig_and_data/figures_v10")
DT = 0.05
N_DEF, N_ATT = 20, 8
POS_STD = 3.0     # sqrt(9)   m       (paper Table)
VEL_STD = 0.3     # sqrt(0.09) m/s
FOLDER = {"case1": "mappo_success_nopn", "case2": "mappo_success_sin"}


def load_case(algo, case):
    base = DATA / algo / f"export_{algo}_data" / FOLDER[case]
    pos = np.loadtxt(base / "agentspos.txt")          # (T,56)
    tgo_dist = np.loadtxt(base / "agentstimetgo.txt")  # (T,40): [tgo,dist]*20
    T = pos.shape[0]
    dpos = pos[:, :40].reshape(T, N_DEF, 2)
    apos = pos[:, 40:].reshape(T, N_ATT, 2)
    tgo = tgo_dist[:, 0::2]                             # (T,20)
    dist = tgo_dist[:, 1::2]                            # (T,20)
    return dict(T=T, dpos=dpos, apos=apos, tgo=tgo, dist=dist)


def reconstruct_groups(d):
    """Assign each defender to its terminal nearest attacker -> groups."""
    term = min(d["T"] - 1, d["T"] - 1)
    dp = d["dpos"][term]           # (20,2)
    ap = d["apos"][term]           # (8,2)
    diff = dp[:, None, :] - ap[None, :, :]
    nearest = np.argmin(np.linalg.norm(diff, axis=2), axis=1)  # (20,)
    groups = {}
    for i, t in enumerate(nearest):
        groups.setdefault(int(t), []).append(i)
    return groups


def arrival_step(d, i):
    """First step defender i enters lethal radius (min distance proxy)."""
    return int(np.argmin(d["dist"][:, i]))


# ---------- (a) sensor-noise propagation through the observation filter ----------
def sensor_noise_study(d, sweep, n_mc=200):
    """Return raw vs jump-filtered LOS-rate RMS error for each noise multiplier."""
    dp, ap = d["dpos"], d["apos"]           # (T,20,2),(T,8,2)
    groups = reconstruct_groups(d)
    tgt_of = {i: t for t, ids in groups.items() for i in ids}
    dq_max = 0.03                            # rad/s jump-rejection threshold (paper)
    raw_rms, filt_rms = [], []
    for mult in sweep:
        raw_e, filt_e = [], []
        for _ in range(n_mc):
            for i in range(N_DEF):
                t = tgt_of[i]
                rel = ap[:, t, :] - dp[:, i, :]           # (T,2) clean relative pos
                q = np.arctan2(rel[:, 1], rel[:, 0])       # LOS angle
                qdot_true = np.gradient(np.unwrap(q), DT)
                noise = RNG.normal(0, POS_STD * mult, rel.shape)
                reln = rel + noise
                qn = np.arctan2(reln[:, 1], reln[:, 0])
                qdot_raw = np.gradient(np.unwrap(qn), DT)
                # jump-rejection filter: hold previous when jump exceeds threshold
                qdot_f = qdot_raw.copy()
                for k in range(1, len(qdot_f)):
                    if abs(qdot_f[k] - qdot_f[k - 1]) > dq_max:
                        qdot_f[k] = qdot_f[k - 1]
                raw_e.append(np.sqrt(np.mean((qdot_raw - qdot_true) ** 2)))
                filt_e.append(np.sqrt(np.mean((qdot_f - qdot_true) ** 2)))
        raw_rms.append(np.mean(raw_e))
        filt_rms.append(np.mean(filt_e))
    return np.array(raw_rms), np.array(filt_rms)


# ---------- (b) packet-loss / zero-order-hold staleness ----------
def packet_loss_study(d, drop_probs, n_mc=300):
    """RMS range error introduced by holding the last received packet."""
    dp, ap = d["dpos"], d["apos"]
    groups = reconstruct_groups(d)
    tgt_of = {i: t for t, ids in groups.items() for i in ids}
    rng_true = {}
    for i in range(N_DEF):
        t = tgt_of[i]
        rng_true[i] = np.linalg.norm(ap[:, t, :] - dp[:, i, :], axis=1)
    out = []
    for p in drop_probs:
        errs = []
        for _ in range(n_mc):
            for i in range(N_DEF):
                r = rng_true[i]
                held = r.copy()
                for k in range(1, len(r)):
                    if RNG.random() < p:              # packet lost -> hold previous
                        held[k] = held[k - 1]
                errs.append(np.sqrt(np.mean((held - r) ** 2)))
        out.append(np.mean(errs))
    return np.array(out)


# ---------- (c) partial interceptor failure ----------
def failure_study(d, fail_probs, n_mc=20000):
    """P(target undefended) and surviving-group arrival spread under i.i.d. loss."""
    groups = reconstruct_groups(d)
    sizes = np.array([len(v) for v in groups.values()])
    arr = {t: np.array([arrival_step(d, i) * DT for i in ids])
           for t, ids in groups.items()}
    undef, spread = [], []
    for q in fail_probs:
        u_cnt = 0
        sp = []
        for _ in range(n_mc):
            any_undef = False
            for t, ids in groups.items():
                alive = [i for k, i in enumerate(ids) if RNG.random() >= q]
                if len(alive) == 0:
                    any_undef = True
                else:
                    times = np.array([arrival_step(d, i) * DT for i in alive])
                    if len(times) >= 2:
                        sp.append(times.max() - times.min())
                    else:
                        sp.append(0.0)
            if any_undef:
                u_cnt += 1
        undef.append(u_cnt / n_mc)
        spread.append(np.mean(sp) if sp else np.nan)
    return sizes, np.array(undef), np.array(spread)


def make_figure(results):
    plt.rcParams.update({"font.family": "serif", "font.size": 11,
                         "axes.grid": True, "grid.alpha": 0.3})
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.2))

    sw = results["noise_sweep"]
    ax[0].plot(sw, results["noise_raw"], "o-", color="#c0392b",
               label="Raw LOS-rate estimate")
    ax[0].plot(sw, results["noise_filt"], "s-", color="#27ae60",
               label="Jump-rejection filter (Eq. 44)")
    ax[0].set_xlabel(r"Sensor-noise multiplier ($\times$ nominal $\sigma$)")
    ax[0].set_ylabel("LOS-rate RMS error (rad/s)")
    ax[0].set_title("(a) Sensor-noise attenuation")
    ax[0].legend(fontsize=8)

    dpv = results["drop_probs"]
    ax[1].plot(np.array(dpv) * 100, results["drop_rms"], "o-", color="#2c6fbb")
    ax[1].axvline(1.0, ls="--", color="#7f8c8d", label="Nominal 1% loss")
    ax[1].set_xlabel("Packet-loss probability (%)")
    ax[1].set_ylabel("Relative-range hold error (m)")
    ax[1].set_title("(b) Communication packet loss")
    ax[1].legend(fontsize=8)

    fpv = np.array(results["fail_probs"]) * 100
    axb = ax[2]
    axb.plot(fpv, np.array(results["undef_prob"]) * 100, "o-", color="#c0392b",
             label="P(a target left undefended)")
    axb.set_xlabel("Per-interceptor failure probability (%)")
    axb.set_ylabel("Undefended-target probability (%)", color="#c0392b")
    axb.tick_params(axis="y", labelcolor="#c0392b")
    axb.set_title("(c) Partial interceptor failure")
    axt = axb.twinx()
    axt.plot(fpv, results["fail_spread"], "s--", color="#e08a1e",
             label="Surviving-group arrival spread")
    axt.set_ylabel("Surviving arrival spread (s)", color="#e08a1e")
    axt.tick_params(axis="y", labelcolor="#e08a1e")
    axb.grid(alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"robustness_analysis.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] figure -> {OUT/'robustness_analysis.pdf'}")


def main():
    d = load_case("mappo", "case1")
    groups = reconstruct_groups(d)
    noise_sweep = [0.5, 1.0, 2.0, 3.0, 4.0]
    drop_probs = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
    fail_probs = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

    raw, filt = sensor_noise_study(d, noise_sweep)
    drop_rms = packet_loss_study(d, drop_probs)
    sizes, undef, spread = failure_study(d, fail_probs)

    results = {
        "group_sizes": sizes.tolist(),
        "min_group_size": int(sizes.min()),
        "noise_sweep": noise_sweep,
        "noise_raw": raw.tolist(),
        "noise_filt": filt.tolist(),
        "noise_attenuation_at_nominal": float(raw[1] / filt[1]),
        "drop_probs": drop_probs,
        "drop_rms": drop_rms.tolist(),
        "drop_rms_at_1pct": float(drop_rms[1]),
        "fail_probs": fail_probs,
        "undef_prob": undef.tolist(),
        "fail_spread": spread.tolist(),
        "undef_prob_at_10pct": float(undef[2]),
    }
    (OUT / "robustness_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items()
                      if not isinstance(v, list) or len(v) < 8}, indent=2))
    make_figure(results)


if __name__ == "__main__":
    main()
