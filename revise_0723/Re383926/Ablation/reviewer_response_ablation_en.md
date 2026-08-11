## Reviewer comment

> Then, it would be helpful to add an ablation investigation demonstrating
> the independent contribution of the trust-aware mechanism, GRU temporal
> encoder, and attention-residual backbone within ART-MAPPO architecture.

## Response

Thank you for this constructive suggestion. We added a controlled
single-component ablation study of ART-MAPPO. Starting from the same
three-dimensional assignment and guidance pipeline, we removed, one at a
time, (i) trust-modulated training exploration, (ii) the GRU temporal encoder,
and (iii) the attention--residual feature extractor. The last component is
replaced by a capacity-matched plain MLP (733,200 versus 732,774 actor
parameters), so the comparison is not confounded by reduced model capacity.
All variants share the same dynamics, reward, target assignment, action
limits, optimizer, rollout budget, checkpoint-selection rule, and random
seeds.

Each variant was trained independently with three seeds for 81,920
environment steps in both Cases 1 and 2. We report episode return, Critic Loss,
and Policy Entropy as three-seed means with standard-deviation bands. After
training, guided exploration was disabled and each frozen actor was evaluated
on exactly 100 paired Monte Carlo episodes per case. No optimizer or
back-propagation operation was performed during evaluation. The primary test
metrics are all-target coverage, all-defender interception, and strict
cooperative success, where every assigned group must be complete and have an
arrival-time spread no greater than 0.5 s. We additionally report
\(E_{\mathrm{co\text{-}time}}\), \(E_n\), \(E_{\mathrm{miss}}\), and
\(E_t\) as diagnostic terminal metrics.

The results separate training-level from deployment-level contributions.
First, the trust-aware mechanism is a training-time device in our
formulation, rather than a deployment-time controller. Removing it increased
the final policy entropy from 2.3773 to 2.4287 in Case 1 and from 2.3722 to
2.6213 in Case 2. Its downstream effect is clearest in Case 1: strict
cooperative success decreased from 53% (95% Wilson CI: 43.29%--62.49%) to
29% (21.01%--38.54%), although both methods retained 100% target coverage and
100% all-defender interception. Thus, the trust-aware mechanism changed the
training distribution in a way that improved final temporal coordination by
24 percentage points. We note that the no-trust variant had a higher final
Case-1 training-return window; consequently, our conclusion is based on the
frozen-policy coordination test and entropy behavior, not on a claim that
trust improves every training statistic.

Second, the attention--residual backbone has a clear independent
deployment-level effect in Case 1. Replacing it by the capacity-matched MLP
reduced cooperative success from 53% to 2% (95% Wilson CI:
0.55%--7.00%), a 51-percentage-point difference, while target coverage and
all-defender interception remained 100%. Its mean
\(E_{\mathrm{co\text{-}time}}\) also increased from 0.1757 s to 0.2395 s.
The nearly equal actor sizes make it unlikely that this difference is merely
a parameter-count effect.

Third, the GRU strongly improved optimization-level value fitting: removing
it increased the final-window Critic Loss by factors of approximately 12.85
in Case 1 (0.25320 versus 0.01970) and 2.84 in Case 2 (0.23335 versus
0.08222). However, the frozen-policy result does not support a universal
deployment benefit under the present setting: the no-GRU variant achieved
99% cooperative success in Case 1, compared with 53% for the full model.
This suggests that the current observations are sufficiently close to
Markovian in Case 1 for a simpler feed-forward policy to fit the nominal
engagement, despite the recurrent critic being substantially easier to fit.
We have revised the discussion accordingly and do not claim that the GRU
improves every final test metric.

Case 2 also provides an important limitation. Target coverage was 79%--80%
for all variants, only 5%--8% of the episodes achieved all-defender
interception, and none satisfied the strict all-group 0.5-s cooperation
criterion. The Case-2 terminal boxplots are therefore based only on the
5--8 complete groups per variant and are presented as diagnostic evidence,
not as proof of superiority. The ablation consequently supports the
independent Case-1 contributions of trust-aware training and structured
attention--residual processing, demonstrates a strong GRU contribution to
critic fitting, and also identifies where those effects do not transfer to
strict deployment success.

We added the protocol, training curves, 100-episode success-rate results,
terminal-metric boxplots, and this qualified interpretation to the revision.
All per-update and per-episode CSV files, model-selection records, and plotting
code are retained for reproducibility.
