# Case 3 end-to-end reviewer experiment protocol

## Isolation and provenance

- Read-only mother archive: `/home/a2rl/xiu_onpolicy_3d_fix/stable_V2`
- Independent working copy: `/home/a2rl/reviewer_xiu_case3_20260729`
- The archived Case-2 recurrent actor and critic are copied byte-for-byte into
  `models/case2`. They are loaded in evaluation mode; no optimizer,
  back-propagation, or parameter update is invoked.
- The original `simple_world_comm_3d.py` remains byte-identical in the Case-3
  copy. Case-3 geometry, IDBO assignment and maneuver generation are attached
  by the independent runner.
- Exact input and output hashes are recorded with each formal result.

## Unseen Case-3 condition

Case 3 is not a relabeling of Case 1 or Case 2. The Case-2 archived state is
used only as a deterministic base template before applying the following
seeded transformation:

1. the eight attackers are placed on a 2075--approximately 2300 m nonuniform
   shell rather than the original Case-1/2 geometry;
2. altitude is staggered over four levels, with seeded perturbations;
3. attacker speed is heterogeneous (34--45 m/s);
4. alternating biased inbound headings are applied; and
5. attacker lateral motion follows a three-stage hybrid maneuver: multi-sine,
   alternating bang-bang, then chirp, with an additional vertical oscillation.

The condition is reproducible from the episode seed.

## End-to-end target assignment

For every episode, IDBO operates on the actual transformed 3-D snapshot
(positions and velocities). The paper-faithful optimizer uses population 24,
80 iterations, linear coefficient decay and maximum group size 3. A
deterministic feasibility repair is applied only if a decoded assignment
leaves a target uncovered or violates the maximum group size. Both the raw
and repaired group counts, costs, optimizer runtime, final population cost,
and final disagreement are saved.

The resulting 0--7 target indices are mapped to attacker IDs 20--27 and written
to each defender before the first policy action. Thus the reported experiment
is the complete IDBO assignment plus cooperative interception chain.

## Policy and guidance configuration

- Frozen actor: stable_V2 Case-2 `actor.pt`
- Frozen critic: stable_V2 Case-2 `critic.pt` (loaded for provenance; evaluation
  actions are produced by the actor)
- Deterministic recurrent inference
- Guidance gain 2.4, time constant 0.25 s, lead factor 1.40
- Learned residual scale 0.20 (nonzero)
- Time-to-go synchronization speed gain 0.01, mean reference
- Hit radius 3 m
- Coordination tolerance 0.5 s
- Physical step 0.05 s; maximum 1500 steps

These settings were selected on disjoint validation seeds 73101--73503. All
screening folders are retained. Formal Monte Carlo seeds start at 74001 and
were not used in screening.

## Formal evaluation

The formal test comprises 100 independently seeded episodes. The principal
rates are:

- individual interceptor hit rate;
- target coverage rate;
- all-target interception rate; and
- cooperative mission success rate, which requires every assigned defender to
  hit and every target group to satisfy the 0.5 s coordination threshold.

The four paper metrics are computed from successful trials only:

\[
E_{\mathrm{co-time}}^{(j)}
=m_j^{-1}\sum_{i\in\mathcal G_j}
|t_i^{\mathrm{arr}}-\bar t_j^{\mathrm{arr}}|,
\]

\[
E_n^{(j)}
=m_j^{-1}\sum_{i\in\mathcal G_j}\bar n_{i,\mathrm{term}},
\quad
E_{\mathrm{miss}}^{(j)}
=m_j^{-1}\sum_{i\in\mathcal G_j}d_i(t_i^{\mathrm{arr}}),
\quad
E_t^{(j)}=\max_i t_i^{\mathrm{arr}}.
\]

The terminal load window is 1.0 s. Episode-level quantities are the mean of
the eight target-group values. Interception rates retain all 100 trials,
including failures.

