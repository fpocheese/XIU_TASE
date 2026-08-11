# Response to the reviewer: failure-case analysis

**Reviewer comment:** “Notably, one of the recommendations is to include
failure-case analysis for unsuccessful interception and delayed cooperative
engagement scenarios because understanding framework limitations remains
essential for practical battlefield deployment reliability.”

**Proposed response:** Thank you for this important recommendation. We have
added a dedicated failure-case analysis for both engagement cases. The purpose
of this experiment is not to replace the Monte Carlo success-rate evaluation,
but to expose a practically relevant boundary between (i) mission-level target
coverage and (ii) strict completion by every assigned interceptor.

We kept the trained policies, target assignment, attacker trajectories, and
3-m hit criterion fixed. No retraining or parameter update was performed. To
represent deployment uncertainty, the frozen-policy evaluation used a
one-sample (0.05-s) observation delay, zero-mean position noise with a 3-m
standard deviation, and zero-mean velocity noise with a 0.3-m/s standard
deviation. We screened 100 reproducible evaluation episodes for each case and
used a fixed, predeclared selection rule: the earliest three seeds in each case
for which all eight attackers were intercepted, exactly 19 of the 20 assigned
interceptors hit, seven of eight assignment groups were complete, and the
arrival-time spread among the observed hits did not exceed the prescribed
0.5-s tolerance. Thus, the six examples were selected by an objective rule
rather than by editing trajectories or choosing visually favorable plots.

In all six boundary cases, the redundant assignment preserved 8/8 target
coverage although one assigned interceptor failed to enter the 3-m hit region.
The missed closest-approach distances were 3.04–3.24 m in Case 1 and
3.10–3.57 m in Case 2. Equivalently, the miss margins above the hit threshold
were only 0.04–0.24 m and 0.10–0.57 m, respectively. These are near-threshold
failures rather than grossly divergent trajectories. The seven completed
assignment groups remained well synchronized: their maximum observed
hit-time spreads were 0.05 s in Case 1 and 0.15–0.35 s in Case 2, all below the
0.5-s criterion. However, the incomplete group's completion time is
right-censored beyond the 75-s evaluation horizon because its missed member
never enters the lethal region. This constitutes an unsuccessful individual
interception and, from the group-completion viewpoint, a delayed cooperative
engagement.

The analysis reveals both a strength and a limitation of the framework.
Redundant target assignment isolates a single near-miss and maintains complete
mission-level target coverage. Nevertheless, the current time-coordination
controller does not include an online recovery mechanism after an interceptor
passes its closest-approach point. Consequently, strict all-member completion
cannot be guaranteed under delayed and noisy observations even when the target
is neutralized by another member. We now state this limitation explicitly and
identify practical extensions: a closing-rate/closest-approach miss detector,
event-triggered target reassignment, a reserve-interceptor policy, and
uncertainty-aware safety constraints. The six examples are reported as
diagnostic boundary cases and are not presented as an estimate of the overall
failure probability.

The revised manuscript includes the six-panel trajectory figure, the
closest-approach diagnostics, and the quantitative table for both cases.

