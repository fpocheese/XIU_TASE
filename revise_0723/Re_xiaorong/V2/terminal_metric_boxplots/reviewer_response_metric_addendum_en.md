**Additional terminal-metric ablation analysis.** We further evaluated the
trained ablation variants using the four terminal criteria defined in the
simulation section. All quantities were recomputed from the same 800
deterministic held-out episodes (100 episodes per variant and scenario); no
policy was retrained or updated. For Case 1, the boxplots include only episodes
in which every member of all eight assigned interceptor groups reached its
target. The corresponding sample counts for ART-MAPPO, w/o Trust, w/o GRU, and
w/o Attention-Residual are 90, 90, 92, and 81. The full model achieves median
\(E_{\mathrm{co\text{-}time}}=0.0130\) s,
\(E_{\mathrm{miss}}=1.7380\) m, and \(E_t=35.2781\) s. Removing the
attention-residual backbone reduces the strict completion rate from 90% to 81%
and increases the terminal-load median from 0.3002 g to 0.3150 g and the
engagement-duration median from 35.2781 s to 35.4563 s.

Case 2 exposes a more important coverage-versus-conditional-quality tradeoff.
None of the four variants completes all eight assigned groups in any episode;
we therefore do not label its distributions as successful-trial statistics.
Instead, we report a complete-group-conditioned diagnostic together with the
unconditional completion rates. ART-MAPPO completes 37.13% of target groups,
compared with 30.25%, 33.50%, and 29.00% for w/o Trust, w/o GRU, and w/o
Attention-Residual, respectively. This conditioning is essential: for example,
w/o Trust has smaller conditional miss and load values but completes
substantially fewer groups, so interpreting the conditional boxplots alone
would introduce survivor bias. The attention-residual result is the most
consistent across criteria: its removal reduces interception coverage and
increases conditional terminal load and miss distance. The GRU result is a
mixed tradeoff—its removal yields smaller synchronization error among the
groups that do complete, but reduces complete-group coverage and increases
conditional miss distance and engagement duration. These observations support
the manuscript's interpretation that trust chiefly stabilizes learning, GRU
provides temporal representation, and attention-residual modeling improves
spatial interaction, while avoiding an unsupported claim that every module
must monotonically improve every conditional terminal metric.

The event logs contain the resultant load at the first lethal-radius entry but
not the full terminal measurement window for all 800 episodes. We therefore
label the reported load quantity as an \(E_n\) terminal-sample estimator rather
than claiming that it is the exact terminal-window average in the revised
definition. The other three metrics are reconstructed directly from the stored
arrival times and terminal distances.
