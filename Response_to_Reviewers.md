# Response to Reviewers

**Manuscript:** *Trust-Aware Multi-Agent Reinforcement Learning-Based
Cooperative Interception Strategy for Maneuvering Swarms with Target Assignment*

We thank the Associate Editor and the reviewers for the detailed and
constructive comments. We have substantially revised the manuscript, corrected
the dimensional and experimental inconsistencies, and added explicit scope
statements wherever the evidence supports only a limited conclusion.
The revision preserves the original manuscript structure, algorithms, training
curves, two engagement cases, and Monte Carlo analysis. Corrections are made
locally, and additional evidence is inserted next to the corresponding original
material rather than replacing the paper with a new shortened narrative.

One implementation-scope clarification is important for interpreting this
revision. The compact code supplied during review is a lightweight example for
loading a policy and exercising the deployment interface. It is not the complete
training, distributed-consensus, or HIL experimental source. The complete method
used by the authors includes the attention-residual network, recurrent encoder,
trust-modulated exploration, dual-clip/adaptive-KL training, distributed
consensus assignment, and hardware-connected HIL chain described in the paper.
We have therefore retained and clarified these implemented components rather
than withdrawing them on the basis of the simplified example.

The principal revisions are:

1. The target-assignment formulation now matches the 20-interceptor/8-target,
   capacity-constrained problem, with distributed timestamped winner-set
   consensus and an explicit assignment-to-guidance interface.
2. IDBO now has complete pseudocode, a 30-paired-seed coefficient study,
   complexity bounds, and limited convergence statements. We do not claim
   global optimality or a Nash equilibrium.
3. ART-MAPPO now has explicit attention-residual, GRU, trust, dual-clip, and
   adaptive-KL equations, a clarified workflow figure, and complete training
   and deployment pseudocode.
4. The evaluation now uses audited three-dimensional kinematics, 17-dimensional
   observations, and three load-command channels. The inconsistent 25-step and
   two-dimensional descriptions have been corrected locally while preserving
   the original evaluation structure and results.
5. All 42 files in the final two-case figure package are included: ART-MAPPO and
   PN planar/3-D trajectories, distance, kinematics, TGO, TGO error, and
   synchronization results. The original training and Monte Carlo figures and
   their detailed analyses are retained.
6. Delay, noise, command lag, packet loss, a delayed-coordination failure, and
   the hardware-connected HIL campaign are reported separately with their
   respective criteria and limitations.

## Reviewer 1

### Comment 1.1: Provide a concise notation table

**Response.** We agree. The appendix notation table now covers the assignment
matrix and objective, IDBO schedule, communication graph, bids and winner sets,
attention tensors, GRU state, trust variables, three-dimensional actions,
dual-clip and adaptive-KL parameters, reward terms, and evaluation metrics.

**Changes in the manuscript.** See Appendix A, “Notation Table.”

### Comment 1.2: Include pseudocode for IDBO and ART-MAPPO

**Response.** We retained and completed both algorithms. The IDBO pseudocode
includes four search roles, preference binarization, local bidding, timestamped
neighbor exchange, top-capacity consensus, disagreement testing, and export of
the agreed assignment. The ART-MAPPO algorithm includes attention-residual
encoding, GRU updates, trust-blended action selection, GAE,
dual-clip/adaptive-KL updates, and the deterministic deployment path.

**Changes in the manuscript.** See Algorithms 1 and 2.

### Comment 1.3: Add intuitive explanations around the dense equations

**Response.** We added operational explanations after the probability,
assignment, consensus, attention-residual, trust, and policy-update equations.
In particular, the manuscript explains why the survival-probability product
supports many-to-one assignment, why top-\(L_{\max}\) reconciliation enforces
capacity, how attention couples geometry and coordination timing, and how trust
changes the learned/guided exploration mixture.

**Changes in the manuscript.** See Sections III-A--III-C and IV-A--IV-C.

### Comment 1.4: Discuss communication delay, sensing uncertainty, and 3D dynamics

**Response.** The revised model is fully three-dimensional. We report a
mathematical-simulation delay/noise/command-lag case and a separate
hardware-connected HIL campaign with observation delay, action delay, channel
noise, and packet dropout. The consensus layer also has an explicit bounded-delay
model and repeated-delivery condition.

**Changes in the manuscript.** See Sections II, III-C, V-A, and V-F.

### Comment 1.5: Discuss reward-weight and hyperparameter sensitivity

**Response.** We added a 30-paired-seed IDBO schedule comparison and a
probability-weight sweep. The decay reaches its own 1% band slightly earlier but
does not improve the final objective, so we no longer attribute a final-quality
gain to that schedule. For ART-MAPPO, the revised text explains the directional
effect and tradeoff of every reward group. Because a complete multi-seed
retraining grid over all reward combinations is not available, we do not present
cross-definition return values as controlled sensitivity evidence.

**Changes in the manuscript.** See Sections IV-E and V-B.

### Comment 1.6: Clarify convergence and equilibrium properties

**Response.** The claims are now separated precisely:

- Elitist local IDBO acceptance makes the aggregate accepted fitness
  nondecreasing over the finite feasible assignment space.
- For fixed bids, a connected graph, bounded delay, and repeated delivery, the
  deterministic top-capacity exchange reaches a common winner set in finitely
  many forwarding rounds.
- For fixed return statistics, the trust smoothing law is a contraction with
  factor \(1-\alpha_T\).
- ART-MAPPO remains a nonconvex PPO-style method. Neither dual clipping nor
  adaptive KL proves global or Nash convergence.

**Changes in the manuscript.** See Sections III-C and IV-E.

## Reviewer 2

### Comment 2.1: Reward shaping and dual clipping are overstated as innovations

**Response.** We agree that reward shaping and dual clipping are not individually
new general-purpose algorithms. The revised contribution is stated as their
integration with attention-residual recurrent encoding, trust-modulated tactical
exploration, three-dimensional constrained actions, and distributed assignment.
The load envelope is enforced directly by the action space; dual clipping and
adaptive KL are described as training stabilizers, not as physical constraint
enforcers.

**Changes in the manuscript.** See the contribution list and Sections IV-C--IV-D.

### Comment 2.2: The left side of the original Fig. 5 is unclear

**Response.** We retained the original workflow figure and clarified its three
paths in the surrounding text: the deployed attention-residual/GRU actor path,
the trust-modulated exploration path, and the training-only
critic/GAE/dual-clip/adaptive-KL path. This makes the
training-versus-deployment boundary explicit without changing the original
method layout.

**Changes in the manuscript.** See the original ART-MAPPO workflow figure and
the added deployment-scope paragraph in Section IV-C.

### Comment 2.3: The trust text contradicts its smoothing equation

**Response.** We corrected this point without removing the implemented trust
mechanism. The manuscript now writes both the update and its increment:
\[
\mathcal T_i^{(k+1)}-\mathcal T_i^{(k)}
=\alpha_T[\sigma(\tau_T\widetilde R_i^{(k)})-\mathcal T_i^{(k)}].
\]
Trust can therefore rise or fall toward a performance-dependent target; it is
not assumed to increase monotonically. The fixed-input contraction statement
now follows directly from the equation.

**Changes in the manuscript.** See Section IV-B,
“Trust-Modulated Tactical Exploration.”

### Comment 2.4: The environment is too simple for a “high-fidelity” claim

**Response.** We replaced the two-dimensional description with audited
three-dimensional fixed-wing kinematics, vertical target motion, a
17-dimensional observation, three commanded load channels, sensing uncertainty,
command lag, packet loss, and HIL execution. We still avoid presenting the
laboratory model as a complete flight-certification environment.

**Changes in the manuscript.** See Sections II, IV-D, V-A, V-D, and V-F.

### Comment 2.5: The 25-step horizon and \(D_1,D_2,D_3\) geometry are inconsistent

**Response.** Corrected. The evaluated step is 0.05 s and the maximum horizon is
1500 steps (75 s). The revised setup table reports explicit initial radial,
altitude, and speed intervals; the unrelated 25-step auxiliary benchmark and
ambiguous \(D_1,D_2,D_3\) labels have been replaced by the actual evaluated
intervals.

**Changes in the manuscript.** See Section V-A and Table
“UAV Parameters for Attackers and Defenders.”

### Comment 2.6: End-to-end verification and ablation are missing

**Response.** The revised paper records the exact 20-to-8 consensus assignment
exported by distributed IDBO and confirms that ART-MAPPO consumes it without
remapping in the three-dimensional and HIL cases. We added the paired IDBO
schedule study and a GRU-state reset diagnostic. A complete independently
retrained attention/trust/GRU factorial study remains future work; accordingly,
the present component descriptions establish the implemented architecture but
do not assign isolated causal gains to every component.

**Changes in the manuscript.** See Sections V-B--V-F.

## Reviewer 3

### Comment 3.1: Justify IDBO operator coefficients and convergence stability

**Response.** The adaptive coefficients now have a concrete role: they move the
four IDBO search operations from broad exploration toward local exploitation.
Continuous preferences are binarized, while the saturation penalty and
top-\(L_{\max}\) consensus enforce the target-capacity logic. The 30-paired-seed
study shows that the tested decay does not improve final cost over the fixed
schedule, and the manuscript states this negative result explicitly.

**Changes in the manuscript.** See Sections III-B--III-C and V-B.

### Comment 3.2: Add complexity and scalability analysis under communication delay

**Response.** We retained the per-UAV complexity decomposition
\(O(TPN^2)+O(N|\mathcal N_i|)+O(N)\), with
\(O(N|\mathcal N_i|)\) communication and \(O(PN+N)\) memory. The bounded-delay
agreement condition and retransmission assumption are stated explicitly.
Delayed dissemination rounds at four delay levels and the connected 19-node
result after one node is removed are now reported.

**Changes in the manuscript.** See Sections III-C and V-B.

### Comment 3.3: Add trust, GRU, and attention-residual ablations

**Response.** The revised method retains all three implemented components and
defines each mathematically. The HIL diagnostic resets the GRU state at every
action using matched episodes; one Case-2 episode loses group simultaneity while
preserving all individual hits. This supports the practical relevance of
temporal state in that executed case. We do not mislabel it as a retrained
feed-forward baseline. Independent multi-seed retraining ablations for attention
and trust are not yet reported, so the revised manuscript does not claim an
isolated numerical gain for either module.

**Changes in the manuscript.** See Sections IV-A--IV-B and V-F.

### Comment 3.4: Compare the proposed policy with conventional MAPPO theoretically

**Response.** Conventional MAPPO uses CTDE with a feed-forward or recurrent
actor-critic and the standard PPO surrogate. ART-MAPPO retains CTDE but adds
semantic attention-residual encoding, GRU memory, trust-modulated structured
exploration, the dual-clip negative-advantage treatment, and adaptive KL
regularization. These additions change representation, exploration, and update
regularization, but they do not provide a stronger global convergence theorem.

**Changes in the manuscript.** See Sections IV-A--IV-C and IV-E.

### Comment 3.5: Add probability-weight and reward sensitivity analyses

**Response.** The probability-weight sweep is now reported at
\(w_{\mathrm{ZEM}}\in\{0.3,0.5,0.7\}\). Because changing the weight changes the
objective itself, the absolute objective values are interpreted as a tactical
tradeoff rather than ranked as if they shared one scale. Reward-term effects are
described using invariant mission metrics rather than incompatible return
definitions.

**Changes in the manuscript.** See Sections IV-E and V-B.

### Comment 3.6: Discuss dual-clip/adaptive-KL deployment overhead

**Response.** We now distinguish training and execution explicitly. Dual
clipping, the adaptive KL coefficient, the centralized critic, and trust
statistics are updated during training. Deterministic deployment evaluates the
attention-residual encoder, GRU, and actor mean. Thus, dual-clip and adaptive-KL
do not add an online optimization loop to the flight-control command path.

**Changes in the manuscript.** See Section IV-C and Fig. “ART-MAPPO workflow.”

### Comment 3.7: Test packet loss, sensor noise, and partial interceptor failures

**Response.** We added tests with sensing delay, position/velocity noise,
first-order command lag, observation/action delay, normalized channel noise, and
1% command dropout. Removing a node from the communication graph still permits
agreement on the connected 19-node subgraph. Closed-loop reassignment after
vehicle loss and stuck-actuator cases were not completed, and this limitation is
stated explicitly.

**Changes in the manuscript.** See Sections V-B and V-F.

### Comment 3.8: Include unsuccessful or delayed-engagement cases

**Response.** We now report the Case-1 impaired episode in which all 20
interceptors reach their targets but only seven of eight groups satisfy the
3-s synchronization criterion; the worst group spread is 43.65 s. This is
reported as a delayed-coordination failure, not a successful coordinated
engagement.

**Changes in the manuscript.** See Section V-F and Table
“HIL Temporal-State and Channel Diagnostics.”

### Comment 3.9: Evaluate generalization to unseen maneuver patterns

**Response.** The revised study includes piecewise evasive and sinusoidal
weaving profiles with vertical motion, but a held-out-family generalization
campaign is not yet available. We locally corrected the interpretation so that
these two profiles are not presented as proof of broad out-of-distribution
generalization, while retaining both original maneuver profiles and all of
their results; held-out maneuvers are listed as future work.

**Changes in the manuscript.** See Sections V-A, V-E, and VI.

### Comment 3.10: Add HIL and semi-physical validation

**Response.** The revised manuscript reports the authors' hardware-connected HIL
campaign: ART-MAPPO communicates with flight-control endpoints while target
dynamics and engagement orchestration run in real time on the host. The HIL
channel includes observation and action delay, noise, and packet dropout, and
the ten-episode diagnostics include both nominal and impaired cases.

The compact code supplied during review co-locates compatible endpoint objects
in one Python process so that the deployment interface can be exercised without
the laboratory hardware. We have now stated clearly that this portable example
is not the complete HIL source and must not be used to reclassify the actual HIL
campaign as software-only emulation.

**Changes in the manuscript.** See the Note to Practitioners and Section V-F.

## Reproducibility and Scope Statement

The revision separates five evidence levels: the original multi-algorithm
training comparison, the 30-paired-seed IDBO schedule study, the delayed
consensus-communication study, the complete two-case three-dimensional and
Monte Carlo guidance results, and the hardware-connected HIL diagnostics.
The exported assignment vector, observation/action dimensions, delays, noise
levels, dropout rate, hit and synchronization criteria, episode counts, and
failure outcome are all stated explicitly.

The simplified released example documents a portable deployment interface; it
does not replace the complete ART-MAPPO training implementation, distributed
assignment implementation, or HIL experimental chain. Conversely, the retained
method and HIL claims do not imply global optimality, Nash convergence,
statistical superiority from the representative runs, failure-reassignment
robustness, out-of-distribution generalization, or flight certification.
