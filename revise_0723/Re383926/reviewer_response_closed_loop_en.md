# Closed-loop response to reviewer comments 1.6, 3.8, and 3.9

## Comment 1.6: end-to-end verification and ablation

We added both requested elements. First, Case 3 executes the complete
IDBO-to-guidance chain: IDBO receives the actual three-dimensional initial
state, assigns 20 interceptors to eight attackers, and the frozen recurrent
policy immediately executes the assignment. Second, we performed a
single-component ablation of the trust-aware mechanism, GRU, and
attention--residual backbone with three training seeds and 100 paired
frozen-policy tests per variant and case. Detailed results are given in
`Case3/` and `Ablation/`.

## Comment 3.8: failure cases

We now distinguish two failure types. In Case 2, the full model misses complete
target coverage in 21 of 100 episodes; 17 of these still intercept seven of
eight targets and the median worst-target closest approach is 4.154 m,
slightly outside the 3-m lethal radius. In unseen Case 3, all 100 trials
intercept all targets, but 28 violate the 0.5-s all-group timing requirement.
Their median worst-group spread is 1.20 s. Thus, the former is a terminal
coverage limitation whereas the latter is a synchronization-margin
limitation. Raw records and the corresponding V10 figure are provided in
`FailureAnalysis/`.

## Comment 3.9: unseen attacker maneuvers

Case 3 changes both initial geometry and attacker policy. The attackers use
staggered altitude and speed together with a previously unseen three-stage
multi-sine, alternating bang-bang, and chirped maneuver with vertical
excitation. The Case-2 actor is frozen and no Case-3 retraining is performed.
Across 100 formal trials, all-target interception is 100%, while strict
all-group coordination is 72% (Wilson 95% CI approximately 62.5%--79.9%).
This supports robustness of the end-to-end interception chain while
quantifying the residual zero-shot synchronization limitation. Five
reproducible complete successes, including full trajectories and publication
figures, are supplied for representative visualization.

## Qualification of the ablation claim

The ablation does not show that every component improves every metric. Trust
removal reduces Case-1 cooperative success from 53% to 29%, and replacing the
attention--residual backbone by a capacity-matched MLP reduces it to 2%.
Removing the GRU increases final Critic Loss by 12.85 times in Case 1 and
2.84 times in Case 2, but the no-GRU frozen actor reaches 99% cooperation in
Case 1. We therefore claim an optimization contribution for the GRU, not a
universal deployment advantage. Case 2 yields no strict cooperative success
for any ablation variant at the present budget and is retained as a documented
limitation.
