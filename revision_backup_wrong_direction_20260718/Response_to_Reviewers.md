# Response to Reviewers

**Manuscript:** *Capacity-Constrained Target Assignment and Recurrent Multi-Agent
Reinforcement Learning for Cooperative Interception of Maneuvering UAV Swarms*

We thank the Associate Editor and the reviewers for the detailed comments. The
review identified important inconsistencies between the original mathematical claims,
the evaluated policy checkpoints, and the available experimental evidence. We
therefore made a substantial evidence-alignment revision rather than preserving claims
that could not be reproduced.

The principal changes are:

1. We reformulated target assignment as the implemented 20-interceptor/8-target,
   capacity-constrained discrete problem and added a reproducible 30-paired-seed study.
2. We removed unsupported distributed-IDBO consensus, global-convergence, and
   equilibrium claims. A separate fixed-source propagation test is labeled as such and
   is not presented as distributed assignment consensus.
3. We renamed the guidance method R-MAPPO and restricted its architecture to the
   evaluated checkpoint: MLP encoder, GRU, and Gaussian action head. Unsupported
   trust-aware and attention-residual mechanisms, dual clipping, and adaptive-KL claims
   were removed.
4. We replaced the old toy-environment curves and two-dimensional geometry with audited
   three-dimensional settings and representative R-MAPPO/PN engagements.
5. We added separate delay/noise/command-lag and software channel-impairment tests,
   including the observed delayed-coordination failure. The latter is explicitly not
   described as HIL.
6. We added notation and algorithm boxes, clarified the reward and evaluation metrics,
   and stated the experiments that remain outside the evidence: multi-seed reward
   retraining, loss-triggered reassignment, held-out maneuvers, processor timing, and
   genuine HIL or flight testing.

Blue text in the highlighted manuscript marks added or substantively revised material.
The clean manuscript contains the same scientific content without revision coloring.

---

## Reviewer 1

### Comment 1.1: Provide a concise notation table

**Response.** We agree. A concise notation table has been added. It separates the
assignment matrix and vector, IDBO budgets, probability weights, recurrent-policy
variables, three-dimensional load commands, and terminal metrics. Ambiguous reuse of
weights has been removed; for example, the assignment probability weights are now
\(w_{\mathrm{ZEM}}\) and \(w_\sigma\).

**Changes in the manuscript.** See the Appendix, “Notation Table,” Table
“Concise Notation Summary.”

### Comment 1.2: Include pseudocode for IDBO and ART-MAPPO

**Response.** We agree. We added algorithm boxes with explicit inputs, outputs,
feasibility repair, stopping budgets, training operations, and deployment operations.
Because the audited checkpoint does not implement the originally claimed ART modules,
the second algorithm is now correctly named R-MAPPO.

**Changes in the manuscript.** See Algorithm “Capacity-Constrained IDBO and Assignment
Export” in Section III and Algorithm “R-MAPPO Cooperative Interception Guidance” in
Section IV.

### Comment 1.3: Add intuitive explanations around the dense equations

**Response.** We agree. The revised text now explains the physical role of the ZEM and
heading terms before the assignment probability, interprets the survival-probability
objective, explains each discrete IDBO operator, identifies what the GRU retains, and
describes how each reward term affects range, alignment, interception, synchronization,
and control effort. Statements such as “guarantees” are now used only for explicit
properties, such as feasibility after repair and monotonicity of the retained incumbent.

**Changes in the manuscript.** See Sections III-A, III-B, IV-A, and IV-B.

### Comment 1.4: Discuss communication delay, sensing uncertainty, and 3D dynamics

**Response.** We agree and now keep three evidence levels separate.

First, the main evaluation uses three-dimensional fixed-wing mathematical dynamics,
three load-command channels, a 0.05 s step, a 3 m hit criterion, and a 0.5 s group
synchronization threshold. In the four representative runs (\(n=1\) per method/case),
both R-MAPPO and the capacity-matched \(N=4\) PN reference achieve 20/20 interceptor
hits and 8/8 synchronized groups. The maximum group spreads are 0.25/0.30 s for
R-MAPPO and 0.40/0.35 s for PN in Cases 1/2. These are feasibility examples, not
success-rate or superiority estimates.

Second, one separate illustrative run per case applies a 50 ms sensing delay, 3 m
position-noise standard deviation, 0.3 m/s velocity-noise standard deviation, and
a 0.30 s command-lag time constant. Both selected runs retain 20/20 hits and 8/8
synchronized groups, with a 0.25 s maximum spread. Because \(n=1\), no robustness
rate is inferred.

Third, a ten-episode software test applies a 100 ms observation delay, a 50 ms action
delay, normalized observation/action noise standard deviations of 0.003/0.02, and
1% command dropout to five local policy endpoints. It uses a different randomized
geometry and relaxed 12 m/3 s criteria, so it is reported separately and is not
called HIL.

**Changes in the manuscript.** See Section V-A, Table “Revised Three-Dimensional
Evaluation Settings,” Section V-D, and Table “Software Temporal-State and Channel
Diagnostics.”

### Comment 1.5: Discuss reward-weight and hyperparameter sensitivity

**Response.** We agree that sensitivity should be evaluated using invariant mission
metrics rather than returns whose numerical definition changes with the reward. We
therefore explain the direction and tradeoff of every reward weight and identify hit
count, group spread, miss distance, peak load, and control energy as the appropriate
common metrics. The previous one-seed preview was not a completed, controlled
retraining study and has been removed. A full-budget multi-seed reward retraining sweep
was not available; the revised paper states this limitation instead of claiming that
one reward component is empirically dominant.

For the assignment probability, we added a 30-seed sweep over
\(w_{\mathrm{ZEM}}\in\{0.3,0.5,0.7\}\). Because each setting defines a different
objective, the resulting objective magnitudes are not ranked as though they were one
common performance metric.

**Changes in the manuscript.** See Sections IV-C and V-B.

### Comment 1.6: Clarify convergence and equilibrium properties

**Response.** We agree and substantially narrowed the theory. Feasibility repair
ensures coverage and the one-to-three capacity constraint for every evaluated
assignment. Elitist acceptance gives
\(J_{\mathrm{best}}^{(k+1)}\leq J_{\mathrm{best}}^{(k)}\), and the nonnegative
incumbent sequence therefore converges to a finite limit. This limited statement does
not prove a global optimum, population consensus, distributed assignment consensus,
or a Nash equilibrium. PPO clipping is likewise described as an update regularizer,
not a global or game-theoretic convergence guarantee. The unsupported
\(\epsilon\)-Nash statement and potential-game argument were removed.

**Changes in the manuscript.** See Section III-C, “Convergence Scope, Complexity, and
Delay Model,” and Section IV-C.

---

## Reviewer 2

### Comment 2.1: Reward shaping and dual clipping are overstated as innovations

**Response.** We agree. Consensus-oriented reward shaping and PPO clipping are no
longer listed as standalone innovations. The contribution is now framed as a
capacity-constrained assignment formulation, an implementation-matched recurrent
guidance controller, and an auditable assignment-to-guidance data interface. The
evaluated checkpoint uses standard clipped PPO; the unsupported dual-clip/adaptive-KL
description has been removed. Load limits are enforced through the action envelope
and load-related reward terms, not attributed to dual clipping.

**Changes in the manuscript.** See the revised Introduction contributions and
Sections IV-A and IV-B.

### Comment 2.2: The left side of the original Fig. 5 is unclear

**Response.** We agree. The original diagram has been replaced. The new data-flow
figure shows one actor sequence:
local 3D observation \(\rightarrow\) MLP feature \(\rightarrow\) one GRU update from
\(\mathbf h_{i,t-1}\) to \(\mathbf h_{i,t}\) \(\rightarrow\) Gaussian actor head
\(\rightarrow\) bounded loads. The centralized critic and GAE/PPO branch is shown
separately as training-only. Unsupported attention and trust branches are absent.

**Changes in the manuscript.** See Fig. “Revised R-MAPPO data flow” in Section IV-A.

### Comment 2.3: The trust text contradicts its smoothing equation

**Response.** We agree with the mathematical criticism. The original equation
describes smoothing toward a performance-dependent target and is not a monotonic
increase law. More importantly, no matching trust-modulated action-selection module
was found in the evaluated checkpoint. We therefore removed the trust mechanism and
its causal performance claims rather than retaining a corrected but unevaluated
conceptual module. The method is renamed R-MAPPO accordingly.

**Changes in the manuscript.** The trust subsection and trust branch were removed;
Section IV now begins with an explicit checkpoint-to-architecture scope statement.

### Comment 2.4: The environment is too simple for a “high-fidelity” claim

**Response.** We agree. “High-fidelity simulation” has been replaced by
“three-dimensional mathematical simulation.” The revised setup specifies the modeled
kinematics, axial/yaw-normal/pitch-normal commands, command limits, geometry, step,
horizon, hit radius, and synchronization criterion. It also lists unmodeled estimator
bias/drift, actuator dynamics, collision avoidance, terrain/weather, processor
latency/jitter, network transport, and physical hardware. We make no aerodynamic,
sensor-stack, HIL, or flight-fidelity claim.

**Changes in the manuscript.** See Sections V-A and V-D and the Note to Practitioners.

### Comment 2.5: The 25-step horizon and \(D_1,D_2,D_3\) geometry are inconsistent

**Response.** We agree. The 25-step value came from an unrelated toy benchmark and
could not represent the reported engagement. It and the associated old curves were
removed. The audited evaluation uses \(\Delta t=0.05\) s and a 1500-step (75 s)
maximum horizon. Ambiguous \(D_1,D_2,D_3\) notation was replaced by executed radial
and altitude intervals: defender radius 52.63–65.44 m, attacker radius
1987.72–2145.95 m, and defender/attacker altitudes 0/120 m, together with the
executed speed ranges.

**Changes in the manuscript.** See Section V-A and Table “Revised
Three-Dimensional Evaluation Settings.”

### Comment 2.6: End-to-end verification and ablation are missing

**Response.** We agree and corrected the scope. A direct reset audit verifies that the
stored, capacity-feasible 20-entry assignment vector shown in the manuscript is
consumed unchanged by both guidance cases. However, that legacy case artifact was not
regenerated by the new 30-seed IDBO study. We therefore call this an audited data
interface, not a newly executed end-to-end IDBO-to-guidance experiment.

The checkpoint audit also showed that trust and attention-residual modules are absent,
so those components and their causal claims were removed rather than assigned
artificial ablations. The GRU is retained because it exists in the checkpoint; the
added inference audit is limited to resetting recurrent state during evaluation and
is not represented as a substitute for full multi-seed retraining. With full recurrent
history, both cases achieve all-group synchronization in 10/10 episodes. Resetting the
hidden state before every action retains 10/10 in Case 1 but gives 9/10 in Case 2,
where the failed episode has a 39.55 s worst spread despite complete individual
interception.

**Changes in the manuscript.** See Sections III-B, IV-A, V-C, and the explicit
limitations in Sections V-D and VI.

---

## Reviewer 3

### Comment 3.1: Justify IDBO operator coefficients and convergence stability

**Response.** We agree. The ambiguous continuous preference equations were replaced
by executable discrete role operations over the 20-entry assignment vector. Rolling
copies from the incumbent with occasional random moves; dancing performs controlled
random redirection; breeding recombines the two best assignments; stealing copies
incumbent components. A common mutation strength decays from 0.36 to 0.04, and every
candidate is repaired before elitist acceptance.

We added a 30-paired-seed comparison with a fixed strength of 0.18. The decaying
schedule reaches its own 1%-of-final band at median iteration 19.0 versus 21.5, but it
does not improve final objective: it is better/equal/worse in 1/25/4 paired runs.
Accordingly, the manuscript claims a modest stabilization-speed observation, not a
final-quality advantage or global convergence.

**Changes in the manuscript.** See Sections III-B, III-C, Fig. “Capacity-constrained
IDBO study,” and Table “IDBO Coefficient-Schedule Study.”

### Comment 3.2: Add complexity and scalability analysis under communication delay

**Response.** We agree. Direct candidate evaluation and repair cost
\(O(MN)\); population size \(P\) and iteration budget \(K\) give
\(O(KPMN)\) time and \(O(PM+MN)\) memory for this implementation. A scale script
evaluates 20/8, 40/16, and 80/32 cases. The archived audit shows increasing runtime,
but repeated absolute timings varied substantially with shared-host load, so the paper
does not interpret seconds as a hardware-independent guarantee.

The available implementation is not distributed IDBO. We therefore removed the
original distributed-consensus claim. A separate 20-node test measures only
fixed-source identifier propagation with 1% dropout: mean rounds are 4.00, 5.80, 7.07,
and 9.17 for maximum link delays 0, 1, 2, and 4. Removing one source node leaves a
connected 19-node graph and propagation completes in \(7.17\pm0.46\) rounds. These
results are explicitly not described as assignment consensus, task reassignment, or
guidance robustness.

**Changes in the manuscript.** See Sections III-B, III-C, and V-B.

### Comment 3.3: Add trust, GRU, and attention-residual ablations

**Response.** We agree that component claims require component evidence. Checkpoint
inspection found a standard MLP+GRU+Gaussian actor with 533,800 parameters, but no
trust or attention-residual modules. We removed the absent modules and all independent
performance attributions. For the retained GRU, we added an inference-only temporal
state audit that zeros the recurrent state before every action. Full history gives
10/10 all-group synchronization in both cases; reset state gives 10/10 in Case 1 and
9/10 in Case 2, with a 39.55 s worst spread in the failed episode. All-hit outcomes
remain 10/10 under this separate software suite's 12 m/3 s criteria. This does not
retrain a feed-forward baseline and is therefore reported only as a bounded
diagnostic, not a full architectural ablation.

**Changes in the manuscript.** See Sections IV-A, IV-C, and V-D and Table “Software
Temporal-State and Channel Diagnostics.” The limitations state
that a full-budget, multi-seed retraining ablation remains necessary for a causal GRU
claim.

### Comment 3.4: Compare the proposed policy with conventional MAPPO theoretically

**Response.** We agree. After implementation alignment, the evaluated method is
R-MAPPO: recurrent MAPPO with the same PPO-style nonconvex optimization as conventional
MAPPO. The GRU changes the policy function class and provides observation history, but
does not strengthen the general convergence guarantee. Standard PPO clipping
regularizes sampled updates but does not hard-bound every realized probability ratio
and does not imply global or Nash convergence. Since trust-aware exploration is not
implemented, no exploration–exploitation advantage is claimed.

**Changes in the manuscript.** See Sections IV-A and IV-C.

### Comment 3.5: Add probability-weight and reward sensitivity analyses

**Response.** We agree. The 30-seed assignment sweep varies
\(w_{\mathrm{ZEM}}\) over 0.3, 0.5, and 0.7 while setting
\(w_\sigma=1-w_{\mathrm{ZEM}}\). The optimized means are 0.004611, 0.006986, and
0.009956, respectively. These objective values are not ranked because the objective
definition changes across weights. The sweep demonstrates that the tactical weighting
changes the optimized problem, not that one weight is universally best.

The reward section now explains coefficient roles and the necessary invariant mission
metrics. A controlled multi-seed retraining sweep was not available, so the previous
one-seed preview and unsupported influence ranking were removed.

**Changes in the manuscript.** See Sections IV-C and V-B.

### Comment 3.6: Discuss dual-clip/adaptive-KL deployment overhead

**Response.** We agree that training and deployment costs must be separated. Neither
dual clipping nor adaptive KL is present in the evaluated policy described in the
revision. The deployed actor uses only the MLP encoder, GRU, and Gaussian mean head and
contains 533,800 parameters. Standard PPO clipping and critic evaluation are
training-time operations. No processor-specific timing, power, or jitter trace is
available, so we make no onboard deadline or real-time hardware claim.

**Changes in the manuscript.** See Sections IV-A and IV-C and the limitations in
Section V-D.

### Comment 3.7: Test packet loss, sensor noise, and partial interceptor failures

**Response.** We added the two delay/noise tests described in Response 1.4. In the
ten-episode software channel test, both cases retain all-hit outcomes in 10/10
episodes under the impaired configuration; all-group synchronization occurs in 9/10
Case-1 episodes and 10/10 Case-2 episodes. This test uses relaxed 12 m/3 s criteria
and five impaired endpoints, so it is not directly compared with the main 3 m/0.5 s
representative cases.

No final-case interceptor-removal, stuck-actuator, or failure-triggered assignment and
guidance experiment was available. We explicitly state that partial-failure robustness
has not been established.

**Changes in the manuscript.** See Section V-D and Table “Software
Temporal-State and Channel Diagnostics.”

### Comment 3.8: Include unsuccessful or delayed-engagement cases

**Response.** We agree. The revised paper reports the impaired Case-1 episode 6 rather
than averaging it away: all 20 interceptors reach their targets, but only seven of
eight groups meet the 3 s criterion and the worst group spread is 43.65 s. This result
shows that individual interception success does not imply coordinated success. It is a
software channel-emulation failure, not a physical-flight failure.

**Changes in the manuscript.** See the final paragraph of Section V-D and Table
“Software Temporal-State and Channel Diagnostics.”

### Comment 3.9: Evaluate generalization to unseen maneuver patterns

**Response.** We agree that held-out maneuver evaluation is necessary. The available
final evidence covers two predefined three-dimensional profiles only: piecewise
evasion and sinusoidal weaving. No held-out maneuver family or out-of-distribution
protocol was executed. We therefore removed any claim that these two cases establish
generalization and list held-out maneuver testing as future work.

**Changes in the manuscript.** See Sections V-D and VI.

### Comment 3.10: Add HIL and semi-physical validation

**Response.** We agree that genuine hardware validation would materially strengthen
the work. The available endpoint package separates policy objects and emulates delay,
noise, and dropout, but all endpoints execute in one Python process. The repository
contains no processor identity, device execution log, transport trace, measured
latency/jitter record, or simulator-to-hardware I/O trace. We therefore renamed the
experiment “software channel emulation” and do not call it HIL, processor-in-the-loop,
or semi-physical validation. Genuine HIL, flight-controller, and flight tests remain
future work.

**Changes in the manuscript.** See the Note to Practitioners, Section V-D, and
Section VI.

---

## Reproducibility and Scope Statement

The revision package includes the 30-seed assignment study script, its JSON/CSV
outputs, and the generated ablation/propagation figure. The three-dimensional result
figures are generated from archived trajectory artifacts. We deliberately removed the
old training and assignment plots because their dimensions, horizon, or plotting logic
did not match the revised 20-to-8 claims.

The revision does not claim completion of the following requested validations:
multi-seed reward-weight retraining, independent trust/attention ablations,
failure-triggered reassignment and guidance, held-out maneuver generalization,
processor timing, or genuine HIL/semi-physical/flight testing. These limitations are
stated in both the manuscript and this response so that the evidence boundary is
unambiguous.
