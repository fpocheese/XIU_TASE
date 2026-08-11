## Reviewer comment 3.8

> Notably, one of the recommendations is to include failure-case analysis for
> unsuccessful interception and delayed cooperative engagement scenarios
> because understanding framework limitations remains essential for
> practical battlefield deployment reliability.

## Response

Thank you for emphasizing this practical issue. We added a failure-mode
analysis that explicitly separates unsuccessful target interception from
successful interception with delayed within-group coordination. The analysis
uses the unmodified frozen-policy records and does not retrain or resample a
model after viewing the failures.

For unsuccessful interception, we use the full ART-MAPPO results in Case 2.
Of 100 paired test episodes, 79 intercepted all eight targets and 21 failed
the all-target-coverage criterion. Seventeen of the 21 failures still
intercepted seven targets, while four intercepted six. The failed episodes
therefore covered 6.810 targets on average (median: seven), indicating a
localized terminal miss rather than complete guidance collapse. The mean and
median worst-target closest approaches were 4.120 and 4.154 m, respectively,
slightly outside the 3-m lethal radius. Targets 4, 6, and 5 accounted for the
largest numbers of uncovered groups (8, 6, and 5 episodes), which localizes
the reliability bottleneck to particular engagement geometries.

For delayed cooperative engagement, we use the unseen-maneuver Case-3
end-to-end experiment. All 100 episodes achieved 20/20 interceptor hits and
8/8 target coverage, but 28 exceeded the strict 0.5-s arrival-spread
requirement in at least one group. These are therefore timing failures, not
interception failures. Their worst-group spread had a mean of 1.275 s, a
median of 1.200 s, and a 95th percentile of 1.950 s. Across the 28 episodes,
40 target groups were delayed; target 7 was involved most frequently
(13 episodes), whereas target 4 had the largest mean delayed spread
(1.742 s).

The two modes suggest different mitigations. The Case-2 misses motivate
additional terminal authority or assignment redundancy for difficult target
geometries. The Case-3 timing failures motivate online group-level
time-to-go resynchronization or stronger cooperative-time adaptation. We
have also revised the presentation to report all-target coverage,
all-defender interception, and strict cooperative success separately, so a
successful kill is not conflated with simultaneous arrival of every assigned
interceptor.
