# V2 component-ablation analysis

## 1. How the new analysis matches the original manuscript

The original manuscript assigns different roles to the three components.
Trust-aware exploration is introduced to modulate the exploration–exploitation
balance during training. The attention-residual backbone extracts coupled
kinematic features and supports gradient propagation, affecting both learning
and deployed feature processing. The GRU integrates temporal engagement
history and is active in both training and deterministic inference.

Accordingly, V2 does not force all three components into a single success-rate
interpretation:

- trust-aware: policy entropy, reward stability, reward learning curve, and
  downstream policy quality;
- GRU: critic convergence and reward learning, followed by Monte Carlo
  interception/coordination;
- attention-residual: reward convergence and stability, followed by Monte
  Carlo interception/coordination.

## 2. Trust-aware mechanism: training-centered evidence

The policy-entropy curves provide the most direct test of the trust claim.

| Case | Full: initial entropy | Full: tail entropy | No trust: initial entropy | No trust: tail entropy |
|---|---:|---:|---:|---:|
| Case 1 | 2.7201 | 2.2947 | 2.8191 | 3.3568 |
| Case 2 | 2.5615 | 1.9280 | 2.9689 | 3.3568 |

The full model exhibits the intended gradual entropy reduction: early broad
exploration is replaced by increasingly concentrated exploitation. Without
trust-aware modulation, entropy rises to 3.3568 and remains there in both
cases. The pooled difference in entropy reduction is 0.9923, with
\(d_z=22.83\) and Holm-adjusted \(p=0.00586\). This directly supports the
manuscript's training-side claim that trust modulates exploration.

The reward evidence is more nuanced. In Case 1, full ART-MAPPO has a tail
return of \(164.555\times10^6\), versus \(158.987\times10^6\) without trust,
and its across-seed standard deviation is \(0.396\times10^6\), versus
\(6.173\times10^6\). Thus trust is associated with much more reproducible
learning in the easier case. In Case 2, however, no-trust has the higher
scalar return. V2 therefore does not claim that trust universally maximizes
the reward; its strongest evidence is the intended entropy schedule and
Case-1 stability.

After training, the difficult-case per-target interception rate is 44.625%
for full versus 34.0% for no-trust, and the corresponding synchronized rate
is 30.5% versus 24.0%. These downstream differences are directionally
consistent with improved exploration, but do not survive the corrected pooled
test and are treated as secondary evidence.

## 3. GRU: training and Monte Carlo evidence

The GRU contribution is clearest in critic convergence.

| Case | Full tail critic loss | No-GRU tail critic loss | Relative reduction |
|---|---:|---:|---:|
| Case 1 | 0.009283 | 0.011848 | 21.6% |
| Case 2 | 0.016745 | 0.085215 | 80.3% |

Across ten paired seed–case observations, the lower critic-loss advantage has
\(d_z=1.022\) and Holm-adjusted \(p=0.00586\). This supports the original
mechanistic argument that temporal history improves value estimation in the
dynamic engagement process. Full ART-MAPPO also has a positive pooled
tail-return difference of \(0.657\times10^6\), although this return difference
is not significant after correction.

The post-training Monte Carlo result is mixed. Case 1 is saturated for both
models. In Case 2, no-GRU has higher per-target interception and coordination
rates than full. The strict all-target rates are zero for both. Thus the
experiment supports a GRU training-side advantage in critic fitting, but does
not establish an independent terminal success advantage under the current
observation design and budget.

## 4. Attention-residual backbone: training and Monte Carlo evidence

The pooled full-versus-no-attention-residual tail-return difference is
\(1.709\times10^6\), with \(d_z=0.884\) and Holm-adjusted
\(p=0.0469\). This is the only component removal with a corrected significant
loss in the primary reward measure.

The post-training evidence is directionally consistent:

- Case 1 strict interception: 100% full versus 98% ablated;
- Case 1 strict coordination: 90% full versus 81% ablated;
- Case 2 per-target interception: 44.625% full versus 38.625% ablated;
- Case 2 per-target coordination: 30.5% full versus 23.625% ablated.

Because the replacement MLP is parameter matched, the result isolates the
attention-residual topology rather than merely testing parameter count.

## 5. One hundred Monte Carlo trials per variant and case

The 100 trials comprise 20 common held-out episodes for each of five
independently trained seeds. This design incorporates both environment
randomness and training-seed variability.

| Case | Variant | Strict interception success | Strict coordination success |
|---|---|---:|---:|
| Case 1 | Full | 100% | 90% |
| Case 1 | No trust | 100% | 90% |
| Case 1 | No GRU | 100% | 92% |
| Case 1 | No attention-residual | 98% | 81% |
| Case 2 | Full | 0% | 0% |
| Case 2 | No trust | 0% | 0% |
| Case 2 | No GRU | 0% | 0% |
| Case 2 | No attention-residual | 0% | 0% |

The Case-2 strict metric has a floor effect because it requires all eight
targets to succeed within the same trial. For transparency it is reported
unchanged. The per-target rates are used only as secondary diagnostics; they
do not replace the strict Monte Carlo result.

## 6. Defensible component conclusions

- Trust-aware: strong and statistically supported control of entropy
  evolution; improved Case-1 reward stability; downstream difficult-case
  target metrics are directional.
- GRU: strong and statistically supported reduction of critic loss,
  especially in Case 2; no independent Monte Carlo terminal-success advantage
  is demonstrated.
- Attention-residual: significant pooled reward contribution and consistent
  direction of post-training interception/coordination effects.

This formulation follows the roles claimed in the original manuscript while
avoiding the unsupported statement that every component improves every
metric.
