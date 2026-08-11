# Reviewer response — V2

**Comment.** “Then, it would be helpful to add an ablation investigation
demonstrating the independent contribution of the trust-aware mechanism, GRU
temporal encoder, and attention-residual backbone within ART-MAPPO
architecture.”

**Response.** Thank you for this constructive suggestion. We added a
controlled four-arm ablation comprising full ART-MAPPO and three
single-component removals: without trust-aware exploration, without the GRU,
and without the attention-residual backbone. The latter is replaced by a
parameter-matched MLP. All environment, reward, optimization, target
assignment, and rollout settings are otherwise fixed. Each variant is trained
with five paired seeds in both cases. Post-training performance is evaluated
over 100 deterministic held-out Monte Carlo trials per variant and case (20
trials for each of five independently trained seeds), for 800 evaluation
episodes in total. Trust-guided exploration is disabled during inference.

We aligned each ablation criterion with the role stated in the manuscript.
Because trust-aware exploration is a training-time mechanism, we assess it
primarily through policy entropy, reward evolution, and cross-seed stability.
Full ART-MAPPO exhibits the intended transition from exploration to
exploitation: its tail entropy is 2.2947 and 1.9280 in Cases 1 and 2, whereas
the no-trust entropy rises to and remains at 3.3568 in both cases. Across the
ten paired seed–case observations, the full model improves entropy reduction
by 0.9923 (\(d_z=22.83\), \(p_{\mathrm{Holm}}=0.00586\)). In Case 1, removing
trust also increases the standard deviation of tail return from
\(0.396\times10^6\) to \(6.173\times10^6\). These results directly support the
claimed exploration–exploitation and stability role. The difficult-case
per-target interception and synchronized rates also increase from 34.0% to
44.625% and from 24.0% to 30.5%, respectively, although these downstream
differences are not significant after correction and are reported as
directional.

The GRU and attention-residual backbone remain active in the deployed policy,
so we examine both training and Monte Carlo effects. Removing the GRU raises
tail critic loss from 0.009283 to 0.011848 in Case 1 and from 0.016745 to
0.085215 in Case 2. The pooled lower-loss advantage is large
(\(d_z=1.022\)) and remains significant after Holm correction
(\(p_{\mathrm{Holm}}=0.00586\)), supporting the value-estimation benefit of
temporal engagement history. However, no-GRU is competitive or better on the
Case-2 per-target terminal metrics; thus we do not claim that recurrence is
universally necessary for terminal success under the present observation
design.

Removing the attention-residual backbone gives the clearest reward
degradation. Full ART-MAPPO improves pooled tail return by
\(1.709\times10^6\), with \(d_z=0.884\) and
\(p_{\mathrm{Holm}}=0.0469\). The Monte Carlo result is directionally
consistent: strict Case-1 interception/coordination decrease from 100%/90% to
98%/81%, while difficult-case per-target interception/coordination decrease
from 44.625%/30.5% to 38.625%/23.625%.

We added three training-side figures—reward, critic loss, and policy
entropy—in the same typography and presentation style as the original
training figure. Curves are generated exclusively from the recorded data,
using a common 5%-horizon moving average and 95% Student-\(t\) confidence
intervals over five seeds. No synthetic curve blending or manual adjustment
is applied. The revised discussion therefore demonstrates the intended
training role of trust, evaluates GRU and attention-residual components on
both learning and held-out interception, and explicitly reports mixed or
non-significant evidence instead of attributing every metric to every module.
