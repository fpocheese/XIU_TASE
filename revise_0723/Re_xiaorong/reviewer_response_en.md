# Response to the reviewer

**Comment.** “Then, it would be helpful to add an ablation investigation
demonstrating the independent contribution of the trust-aware mechanism, GRU
temporal encoder, and attention-residual backbone within ART-MAPPO
architecture.”

**Response.** Thank you for this constructive suggestion. We added a
controlled component-ablation experiment that compares the full ART-MAPPO
architecture with three single-component removals: (i) without the
trust-aware training mechanism, (ii) without the GRU temporal encoder, and
(iii) without the attention-residual backbone. The plain-MLP replacement used
in the third variant was capacity matched to avoid conflating architectural
topology with parameter count (actor: 732,774 versus 733,200 parameters;
critic: 749,761 versus 749,932). All environment, reward, optimizer, rollout
budget, target assignment, and guidance settings were otherwise identical.

For each of the two three-dimensional interception cases, we trained every
variant with five paired random seeds (701–705). Case 1 used \(0.6\) million
environment steps and Case 2 used \(1.8\) million steps. Each trained
seed/case model was then evaluated for 20 deterministic held-out episodes
using common evaluation seeds across variants, giving \(4\times2\times5
\times20=800\) episodes. Guided exploration was disabled at evaluation.
Consequently, the trust-aware comparison measures the effect of the
training-time return-based trust memory; it does not give the full method an
extra guidance source at deployment.

The new figure reports the five-seed learning curves and the held-out
per-target interception and synchronized-interception rates, with 95%
Student-\(t\) confidence intervals over independent training seeds. We also
performed paired comparisons by seed and case using two-sided exact sign-flip
tests, with Holm correction across the three component removals within each
metric.

The ablation provides three main findings. First, removing the
attention-residual backbone produces the clearest degradation. Across the ten
matched seed–case pairs, the full architecture improves final training return
by \(1.709\times10^6\), with a large paired effect
\(d_z=0.884\) and a Holm-adjusted \(p=0.0469\). In the difficult Case 2, the
full model also improves per-target interception and synchronization by
6.00 and 6.875 percentage points, respectively, relative to this ablation.

Second, the trust-aware mechanism has a task-aligned but more variable effect.
In Case 2, the full model improves per-target interception from 34.0% to
44.625% and synchronized interception from 24.0% to 30.5%, i.e., gains of
10.625 and 6.5 percentage points. In Case 1, both variants reach 100%
per-target interception and essentially saturated synchronization; however,
removing trust increases the across-seed dispersion of final training return
substantially. The pooled target-level effects do not survive Holm correction
(\(p_{\rm Holm}=0.75\) for interception and \(1.0\) for synchronization), so
we describe this evidence as directional rather than statistically
conclusive.

Third, the GRU ablation shows that temporal recurrence is not independently
indispensable under the present observation design and training budget. The
full model has a positive pooled final-return difference of
\(0.657\times10^6\) with a medium-to-large paired effect
\(d_z=0.678\), but it is not significant after correction
(\(p_{\rm Holm}=0.1289\)). Moreover, the no-GRU model attains higher
target-level interception and synchronization in Case 2. We therefore revised
the manuscript to avoid claiming a universal GRU advantage and instead
interpret the result as evidence that the current 17-dimensional observation
already exposes enough instantaneous time-to-go and engagement geometry for
a feed-forward encoder to remain competitive.

For transparency, we retain the stricter trial-level metrics in all CSV
files. In Case 1, the full method achieves 100% all-target interception and
90% all-target synchronized interception. In Case 2, the all-eight-target
indicator is zero for all variants at this training budget; this floor effect
is why the figure uses the more informative per-target rates rather than
hiding or replacing the strict metric. The revised text explicitly reports
this limitation.

Overall, this investigation substantiates the independent value of the
attention-residual backbone, provides directional evidence for the
trust-aware training mechanism in the difficult scenario, and establishes
that the GRU contribution is conditional rather than universal. We have
updated the manuscript accordingly and provide the complete 800-episode raw
data, paired statistics, source snapshot, and reproducibility records with
the revision.
