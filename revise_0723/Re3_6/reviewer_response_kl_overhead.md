# Re3_6 — Computational overhead of dual-clip PPO + adaptive-KL under latency constraints

**Reviewer comment.** "Moreover, I suggest the author explain whether dual-clip PPO with
adaptive KL regularization introduces additional computational overhead during real-time
deployment on processors operating in latency constraints within UAV swarms."

## Response (short answer)

Dual-clip PPO and adaptive-KL regularization are **training-loss operations only**. They act on
the objective in Eq. (total loss) — the dual-clip branch is a scalar max/min selection on the
advantage sign, and the adaptive-KL term is a log-ratio estimate plus a scalar update of the
penalty coefficient β_KL. **Neither appears in the deployed policy.** At execution each
interceptor runs only the actor forward pass, so the two mechanisms contribute **exactly 0 ms**
to the onboard real-time path. Their cost during offline training is negligible relative to the
network forward/backward passes.

To make this quantitative rather than assertive, we re-implemented the deployed ART-MAPPO actor
(attention-residual encoder → GRU → Gaussian head) at the exact deployment dimensions and
measured both the inference latency and the marginal training cost of the two mechanisms. All
numbers below are measured, not estimated. See `kl_dualclip_overhead.pdf` and
`kl_overhead_results.json`.

## Measured results

Deployed configuration: hidden size 256, 4 attention heads, 2 residual blocks, 8 observation
tokens, 3-D action. Guidance cycle = 50 ms.

| Quantity | Value |
|---|---|
| Actor inference, single interceptor | **0.30 ms** (p99 0.42 ms) |
| Actor inference, 20-agent swarm (batched) | **3.98 ms** |
| Dual-clip + adaptive-KL cost **at deployment** | **0.00 ms** (not in inference path) |
| Dual-clip + adaptive-KL cost **per training update** | ≈ 0.013 ms |
| → as fraction of one full policy update | **< 0.01 %** |

Even under a deliberately pessimistic embedded slowdown band (10–20× a server-class CPU), the
single-decision inference stays at **3.0–6.1 ms**, comfortably inside the 50 ms cycle. As an
upper-bound stress test, a 4× larger network (hidden 1024) still infers in 9.2 ms per decision,
and there the two mechanisms' training overhead is even smaller in relative terms (0.0005 %),
because the network cost grows while the scalar KL/dual-clip arithmetic does not.

**Why the deployment cost is exactly zero.** The actor that flies is the Gaussian policy head;
the value critic, the dual-clip surrogate, the KL divergence estimate, and the β_KL controller
are all part of the centralized-training objective and are discarded at execution (CTDE). There
is nothing to evaluate for them at inference time.

## Where the manuscript changed

- New `\hladd` paragraph directly after the total-loss equation in the ART-MAPPO optimization
  subsection, clarifying that both mechanisms are training-only and stating the measured
  0.30 ms inference / 0 ms deployment / < 0.01 % training figures.

## Reproduce
```
python3 code/kl_dualclip_overhead.py   # -> kl_dualclip_overhead.pdf/.png, kl_overhead_results.json
```

Threading is pinned to a single BLAS thread so the reported latencies reflect a conservative,
per-core deployment rather than a many-core server best case.
