# Case 3: Frozen-policy IDBO--ART-MAPPO end-to-end protocol

Status: preregistered before the first Case-3 evaluation.

## Purpose

Case 3 is an independent reviewer experiment, not a renamed Case-1/Case-2
perturbation. It measures whether the complete target-assignment and guidance
pipeline transfers to a previously unseen attacker strategy without retraining
or fine-tuning the policy.

## Initial engagement

- 20 defenders and 8 attackers; the defender state and all vehicle constraints
  are unchanged from the audited paper environment.
- Attackers occupy a wider shell whose nominal radius is
  `2260 + 42*j` m for attacker index `j`, plus zero-mean 18-m seeded
  perturbations.
- Attacker altitude is staggered over four levels,
  `85 + 30*(j mod 4)` m, plus zero-mean 2.5-m seeded perturbations.
- Attacker speed spans 34--47 m/s and inbound headings have alternating
  cross-track biases of 0.055--0.075 rad.
- Every Monte Carlo episode uses a unique, recorded seed.

## Unseen penetration strategy

The attackers use three heterogeneous roles (`j mod 3`) and two role-dependent
switching times:

1. multisine lateral weaving with vertical oscillation and a small axial pulse;
2. saturated smooth jinks with role-dependent sign/frequency and a signed
   axial acceleration pulse;
3. frequency-swept lateral/vertical terminal penetration with an axial pulse.

The maneuver is clipped by the same attacker load and speed constraints used
in the audited environment. This differs from Case 1 (constant maneuver
transitioning to PN) and Case 2 (fixed-frequency continuous weaving).

## End-to-end execution

1. Reset and construct the seeded Case-3 snapshot.
2. Run the paper-faithful distributed IDBO implementation with population 30,
   80 iterations, linearly decaying coefficients, and target capacity 3.
3. Apply deterministic feasibility repair only if needed; record the raw and
   repaired target loads, assignment cost, disagreement, and runtime.
4. Freeze the selected Case-2-trained Full ART-MAPPO policy and execute the
   assigned cooperative engagements with guided exploration disabled.
5. No gradient, optimizer step, backpropagation, policy update, or Case-3
   fine-tuning is permitted.

## Monte Carlo and metrics

- Exactly 100 episodes, seeds 96001--96100.
- Sensor delay: 50 ms; position noise: 3 m; velocity noise: 0.3 m/s;
  command lag: 0.30 s; hit radius: 3 m; coordination tolerance: 0.5 s.
- Report target-coverage success, all-defender-hit rate, cooperative success,
  and mission success with Wilson 95% confidence intervals.
- For successful complete groups, report
  `E_co-time`, `E_n`, `E_miss`, and `E_t`.
- Retain episode-, target-, and assignment-level CSV files, failure classes,
  IDBO diagnostics, and the frozen checkpoint identity.

