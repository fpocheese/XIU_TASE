#!/usr/bin/env python3
"""Re3_6: Deployment-time computational overhead of dual-clip PPO + adaptive KL.

Reviewer question: does dual-clip PPO with adaptive KL regularization add
computational cost during *real-time deployment* on latency-constrained UAV
processors?

Key fact this script quantifies:
  * The dual-clip surrogate and the adaptive-KL penalty are TRAINING-LOSS terms.
    At deployment each interceptor executes ONLY the actor forward pass
    (attention-residual encoder + GRU + Gaussian head). Neither the dual-clip
    max/min nor the KL log-ratio / beta update appears in the inference path.
  * Therefore the deployment overhead of these two mechanisms is exactly zero,
    and even during training their extra cost is a small fraction of one update.

We build a faithful NumPy reimplementation of the actor forward pass at the
deployed network dimensions (paper Table II: 4 attention heads, 2 residual
blocks; repo config hidden_size in {256, 1024}, 1 recurrent layer, 3-D load
action) and measure: (i) per-decision inference latency, (ii) the marginal
training-only cost of dual-clip + adaptive KL versus a plain clipped update.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(0)
OUT = Path(__file__).resolve().parents[1]

# Deployment control cycle: paper simulation step size is 0.05 s = 50 ms.
CONTROL_CYCLE_MS = 50.0
STRICT_CYCLE_MS = 10.0          # a stricter 100 Hz inner-loop budget
EMBED_SLOWDOWN = 20.0           # conservative server-CPU -> Jetson NX scaling


def _lin(n_in, n_out):
    return RNG.standard_normal((n_in, n_out)).astype(np.float32) * (1.0 / np.sqrt(n_in))


def relu(x):
    return np.maximum(x, 0.0)


class Actor:
    """Attention-residual + GRU Gaussian actor (deployment network)."""

    def __init__(self, hidden, n_tokens=8, tok_feat=16, heads=4, res_blocks=2, act_dim=3):
        self.H, self.L, self.heads, self.res_blocks = hidden, n_tokens, heads, res_blocks
        self.tok_feat, self.act_dim = tok_feat, act_dim
        self.hd = hidden // heads
        self.W_embed = _lin(tok_feat, hidden)
        self.Wq, self.Wk, self.Wv = _lin(hidden, hidden), _lin(hidden, hidden), _lin(hidden, hidden)
        self.Wo = _lin(hidden, hidden)
        self.res = [(_lin(hidden, 4 * hidden), _lin(4 * hidden, hidden)) for _ in range(res_blocks)]
        # GRU (input=hidden, hidden=hidden)
        self.Wxz, self.Wxr, self.Wxh = _lin(hidden, hidden), _lin(hidden, hidden), _lin(hidden, hidden)
        self.Whz, self.Whr, self.Whh = _lin(hidden, hidden), _lin(hidden, hidden), _lin(hidden, hidden)
        self.W_mean, self.W_logstd = _lin(hidden, act_dim), _lin(hidden, act_dim)

    def forward(self, tokens, h_prev):
        B = tokens.shape[0]
        x = tokens @ self.W_embed                                   # (B,L,H)
        q = (x @ self.Wq).reshape(B, self.L, self.heads, self.hd)
        k = (x @ self.Wk).reshape(B, self.L, self.heads, self.hd)
        v = (x @ self.Wv).reshape(B, self.L, self.heads, self.hd)
        scores = np.einsum("blhd,bmhd->bhlm", q, k) / np.sqrt(self.hd)
        scores -= scores.max(-1, keepdims=True)
        attn = np.exp(scores)
        attn /= attn.sum(-1, keepdims=True)
        ctx = np.einsum("bhlm,bmhd->blhd", attn, v).reshape(B, self.L, self.H)
        ctx = ctx @ self.Wo
        feat = (x + ctx).mean(axis=1)                               # residual + pool -> (B,H)
        for w1, w2 in self.res:
            feat = feat + relu(feat @ w1) @ w2                      # residual MLP block
        z = 1.0 / (1.0 + np.exp(-(feat @ self.Wxz + h_prev @ self.Whz)))
        r = 1.0 / (1.0 + np.exp(-(feat @ self.Wxr + h_prev @ self.Whr)))
        hh = np.tanh(feat @ self.Wxh + (r * h_prev) @ self.Whh)
        h = (1.0 - z) * h_prev + z * hh                            # GRU state
        mean = h @ self.W_mean
        logstd = h @ self.W_logstd
        return mean, logstd, h


def _time(fn, repeat, warmup=20):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return np.array(ts)


def measure_inference(hidden):
    act = Actor(hidden)
    tok1 = RNG.standard_normal((1, act.L, act.tok_feat)).astype(np.float32)
    h1 = np.zeros((1, hidden), np.float32)
    tok20 = RNG.standard_normal((20, act.L, act.tok_feat)).astype(np.float32)
    h20 = np.zeros((20, hidden), np.float32)
    rep = 400 if hidden <= 256 else 60
    t_single = _time(lambda: act.forward(tok1, h1), repeat=rep, warmup=10) * 1e3   # ms
    t_swarm = _time(lambda: act.forward(tok20, h20), repeat=max(30, rep // 4), warmup=5) * 1e3
    return t_single, t_swarm


def gae(rewards, values, gamma=0.99, lam=0.97):
    T = len(rewards)
    adv = np.zeros(T, np.float32)
    last = 0.0
    for t in range(T - 1, -1, -1):
        nv = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * nv - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv


def training_step(mb, hidden, use_dualclip, use_adaptive_kl,
                  clip=0.2, c_dc=3.0, delta_targ=0.02, beta=1.0):
    """One minibatch policy-loss evaluation. Returns (wall_ms, extra_ms).

    extra_ms isolates the marginal cost of the dual-clip branch and the
    adaptive-KL log-ratio + beta update, i.e. the two mechanisms the reviewer
    asks about, relative to a plain single-clip surrogate.
    """
    logp_old = mb["logp_old"]
    adv = mb["adv"]

    t0 = time.perf_counter()
    logp_new = logp_old + 0.05 * RNG.standard_normal(logp_old.shape).astype(np.float32)
    ratio = np.exp(logp_new - logp_old)
    surr1 = ratio * adv
    surr2 = np.clip(ratio, 1.0 - clip, 1.0 + clip) * adv
    loss = -np.minimum(surr1, surr2)                       # plain clipped PPO
    base_loss = loss.mean()
    t_base = time.perf_counter() - t0

    t1 = time.perf_counter()
    extra = 0.0
    if use_dualclip:
        neg = adv < 0.0
        dual = np.maximum(np.minimum(surr1, surr2), c_dc * adv)
        loss = np.where(neg, -dual, loss)
        base_loss = loss.mean()
    if use_adaptive_kl:
        approx_kl = float(np.mean(logp_old - logp_new))    # KL log-ratio estimate
        if approx_kl > 2.0 * delta_targ:
            beta = min(1.5 * beta, 10.0)
        elif approx_kl < 0.5 * delta_targ:
            beta = max(0.9 * beta, 1e-4)
        base_loss = base_loss + beta * approx_kl
    extra = time.perf_counter() - t1
    return (t_base + extra) * 1e3, extra * 1e3


def _update_network_cost(act, batch):
    """Proxy for the dominant cost of one minibatch update: actor forward pass
    over the batch plus a backward pass (~2x forward). This is what dual-clip and
    adaptive KL must be compared against, since they are added on top of it."""
    tok = RNG.standard_normal((batch, act.L, act.tok_feat)).astype(np.float32)
    h = np.zeros((batch, act.H), np.float32)
    t0 = time.perf_counter()
    act.forward(tok, h)          # forward
    act.forward(tok, h)          # backward proxy (~1x forward)
    act.forward(tok, h)          # optimizer/grad proxy (~1x forward)
    return (time.perf_counter() - t0) * 1e3


def measure_training(hidden, batch=256, repeat=25):
    """Compare a REALISTIC minibatch update (network fwd/bwd + surrogate) with and
    without the dual-clip branch and adaptive-KL term. The overhead is reported
    against the full update cost (network fwd/bwd dominates), not the surrogate
    arithmetic alone. Costs are measured per-component then combined so the
    expensive network pass is timed only a bounded number of times."""
    act = Actor(hidden)
    mb = {
        "logp_old": RNG.standard_normal(batch).astype(np.float32),
        "adv": RNG.standard_normal(batch).astype(np.float32),
    }
    # dominant cost: one network fwd/bwd-proxy per minibatch update
    t_net = _time(lambda: _update_network_cost(act, batch), repeat=repeat, warmup=3).mean()
    # cheap cost: the surrogate arithmetic, without/with the two mechanisms
    t_surr_plain = _time(lambda: training_step(mb, hidden, False, False),
                         repeat=2000, warmup=50).mean()
    t_surr_full = _time(lambda: training_step(mb, hidden, True, True),
                        repeat=2000, warmup=50).mean()
    t_plain = t_net + t_surr_plain
    t_full = t_net + t_surr_full
    extras = training_step(mb, hidden, True, True)[1]
    return np.array([t_plain]), np.array([t_full]), extras


EMBED_BAND = (10.0, 20.0)   # conservative server-CPU -> Jetson-NX slowdown band


def main():
    results = {}
    for hidden in (256, 1024):
        t_single, t_swarm = measure_inference(hidden)
        t_plain, t_full, extra_ms = measure_training(hidden)
        results[str(hidden)] = {
            "inference_single_ms_mean": float(t_single.mean()),
            "inference_single_ms_p99": float(np.percentile(t_single, 99)),
            "inference_swarm20_ms_mean": float(t_swarm.mean()),
            "deploy_kl_dualclip_ms": 0.0,   # NOT in the inference path
            "train_plain_ms_mean": float(t_plain.mean()),
            "train_full_ms_mean": float(t_full.mean()),
            "train_dualclip_kl_extra_ms": float(extra_ms),
            "train_overhead_pct": float(100.0 * (t_full.mean() - t_plain.mean()) / t_plain.mean()),
            "embed_single_ms_lo": float(t_single.mean() * EMBED_BAND[0]),
            "embed_single_ms_hi": float(t_single.mean() * EMBED_BAND[1]),
        }
    results["_meta"] = {
        "deployed_hidden_size": 256,
        "control_cycle_ms": CONTROL_CYCLE_MS,
        "strict_cycle_ms": STRICT_CYCLE_MS,
        "embed_band": EMBED_BAND,
        "note": ("dual-clip + adaptive KL are training-loss terms; they are absent "
                 "from the deployed actor forward pass, so their real-time deployment "
                 "overhead is exactly zero."),
    }
    (OUT / "kl_overhead_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    make_figure(results)


def make_figure(results):
    plt.rcParams.update({"font.family": "serif", "font.size": 11,
                         "axes.grid": True, "grid.alpha": 0.3})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.1))

    h = "256"   # deployed network size
    single = results[h]["inference_single_ms_mean"]
    lo, hi = results[h]["embed_single_ms_lo"], results[h]["embed_single_ms_hi"]
    labels = ["Actor forward\n(server CPU)", "Actor forward\n(embedded, est.)",
              "Dual-clip +\nadaptive KL"]
    vals = [single, hi, 0.0]
    errs = [0.0, hi - lo, 0.0]
    colors = ["#2c6fbb", "#e08a1e", "#27ae60"]
    xb = np.arange(len(labels))
    axL.bar(xb, vals, 0.6, yerr=[[0, hi - lo, 0], [0, 0, 0]], color=colors, capsize=4)
    axL.axhline(CONTROL_CYCLE_MS, ls="--", color="#c0392b",
                label=f"{CONTROL_CYCLE_MS:.0f} ms control cycle (20 Hz)")
    axL.axhline(STRICT_CYCLE_MS, ls=":", color="#7f8c8d",
                label=f"{STRICT_CYCLE_MS:.0f} ms budget (100 Hz)")
    axL.text(0, single, f"{single:.2f}", ha="center", va="bottom", fontsize=9)
    axL.text(1, hi, f"{lo:.1f}-{hi:.1f}", ha="center", va="bottom", fontsize=9)
    axL.text(2, 0.4, "0.00\n(not in\ninference path)", ha="center", va="bottom",
             fontsize=8, color="#1e7a43")
    axL.set_xticks(xb); axL.set_xticklabels(labels, fontsize=8)
    axL.set_ylabel("Per-decision deployment latency (ms)")
    axL.set_title("(a) Deployment cost vs. real-time budget (hidden=256)")
    axL.legend(fontsize=8, loc="upper right")

    hiddens = ["256", "1024"]
    x = np.arange(len(hiddens))
    w = 0.36
    plain = [results[hh]["train_plain_ms_mean"] for hh in hiddens]
    full = [results[hh]["train_full_ms_mean"] for hh in hiddens]
    axR.bar(x - w / 2, plain, w, label="Full clipped-PPO update", color="#95a5a6")
    axR.bar(x + w / 2, full, w, label="+ dual-clip + adaptive KL", color="#27ae60")
    for xi, hh in enumerate(hiddens):
        pct = results[hh]["train_overhead_pct"]
        axR.text(xi + w / 2, full[xi], f"+{pct:.2f}%", ha="center", va="bottom", fontsize=8)
    axR.set_xticks(x); axR.set_xticklabels([f"hidden={hh}" for hh in hiddens])
    axR.set_ylabel("Wall time per policy update (ms)")
    axR.set_title("(b) Training-only overhead of the two mechanisms")
    axR.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"kl_dualclip_overhead.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] figure -> {OUT/'kl_dualclip_overhead.pdf'}")


if __name__ == "__main__":
    main()
