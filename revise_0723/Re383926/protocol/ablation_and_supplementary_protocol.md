# Pre-registered reviewer experiment protocol

## Evidence and integrity boundary

- Every ablation uses the same environment, paper five-component reward,
  initialization, seed set, rollout budget, optimizer, learning-rate schedule,
  evaluator, and final-checkpoint rule.
- The only difference between a pair is removal of the named component:
  trust-modulated training exploration, GRU temporal encoder, or the
  attention-residual backbone.
- Hyperparameter pilots use a development seed that is excluded from the
  five formal seeds and the 100-episode Monte Carlo tests.
- No episode, seed, or named checkpoint may be removed because its result is
  unfavorable.  A pre-declared stride plus the final checkpoint is evaluated
  on held-out validation seeds; model selection is never based on the test
  set.
- Failed interceptions remain in reliability denominators.  Conditional
  terminal metrics are computed only where their arrival-time definition is
  valid, and their sample count is always reported.

## Paper alignment

The actor directly supplies the clipped command
`[n_x,n_y,n_z]` in `[-0.1,1] x [-1,1] x [-1,1]`; no deterministic base
guidance is added at deployment.  The command response time constant is
0.30 s.  Observations are clipped to `[-1,1]`.

PPO actor losses and guided-action counts include only samples for which the
defender was alive when the action was selected.  Once a defender reaches its
target, subsequent absorbing/post-terminal steps are masked.  This is applied
identically to all variants.

Externally sampled PN, boundary-probe, and uniform-mixture transitions remain
in the joint trajectory and therefore train the centralized critic.  They are
masked from the Gaussian actor surrogate because they were not sampled from
`pi_theta` and cannot carry a valid `pi_theta/pi_theta_old` likelihood ratio.
Only actions sampled by the learned actor contribute to its PPO loss.  This is
the discrete-mixture limit of the paper's trust-exploration distribution, not
an imitation or auxiliary loss.

The 4096-transition PPO buffer is an optimization chunk, not the physical
engagement horizon.  Unfinished engagements continue across updates and are
terminated/reset only at all-defender termination or the independent
1500-physics-step horizon.  Trust statistics are updated from completed
episode returns only.

The training reward is exactly

`w_dist exp(-alpha_dist d)
 - w_angle alpha_angle e_angle
 + w_hit R_hit I_first_hit
 - w_coord alpha_coord |tgo_i-mean_group_tgo|
 - w_energy alpha_energy ||n||_2^2`.

Nominal values are `w_dist=4`, `w_angle=0.05`, `w_hit=200`,
`w_coord=18`, `w_energy=0.005`, `alpha_dist=2e-4`,
`alpha_angle=0.002`, `alpha_coord=0.16`, `alpha_energy=0.01`,
and `R_hit=2200`.

## Ablation endpoints

Each formal variant is trained with five independent seeds.  After held-out
checkpoint selection within every seed, each selected model receives 20
disjoint final episodes.  Their union is exactly 100 episodes per
variant/case.  The episode allocation and seeds are identical across the four
variants, so both training-seed variability and paired Monte Carlo effects
remain visible.

Primary mission endpoints:

1. target-coverage success: every one of the eight attackers is reached by at
   least one assigned defender;
2. all-defender hit rate: every assigned defender reaches its target;
3. cooperative success: every target group is complete and its
   max-min arrival-time spread is at most 0.5 s;
4. mission success: target coverage and cooperative success.

Component-specific endpoints:

- Trust mechanism: early-return area under the curve, updates to 90% of
  terminal return, inter-seed tail-return variability, and entropy evolution.
- GRU: tail critic loss, Case-2 cooperative success, and frozen-policy
  performance on unseen temporal maneuver waveforms.
- Attention-residual backbone: early-return area under the curve, tail return,
  target coverage, and closest-approach/miss metrics.

The theory is considered supported by consistent component-specific evidence;
the full model is not required to dominate every secondary metric.  Mixed or
null findings are retained and discussed.

## Section 3.8: failure-case analysis

The frozen full policy is evaluated under nominal uncertainty and controlled
stressors: increased sensing delay/noise, increased command lag, and a
compound stress condition.  Every condition uses 100 episodes.

Failure classes are mutually exclusive:

1. unsuccessful interception: at least one attacker is uncovered;
2. incomplete cooperative group: all attackers are covered, but at least one
   assigned defender does not arrive;
3. delayed cooperative engagement: every defender arrives, but a group spread
   exceeds 0.5 s;
4. mission success.

Closest approach, target/group completion, overload saturation, and the four
paper terminal metrics are retained for diagnosis.

## Section 3.9: unseen-maneuver generalization

No retraining is allowed.  The frozen Case-2 full policy is evaluated on
nominal sinusoidal penetration and three unseen waveforms: frequency chirp,
multi-sine, and smoothed piecewise jink.  Each condition uses 100 identical
Monte Carlo seeds.  Success-rate Wilson intervals and paired degradation from
nominal are reported.

## Section 1.6: end-to-end IDBO to ART-MAPPO

Each episode first perturbs the paper initial engagement snapshot within the
declared physical envelope.  Paper-faithful IDBO then computes a capacity
constrained assignment from the actual positions and velocities.  A
deterministic feasibility repair implements the manuscript's stated uncovered-
target repair.  The resulting target indices are passed directly to the frozen
shared ART-MAPPO actors.  Fixed assignment uses the identical perturbed
episode seeds as a paired reference.  IDBO cost, disagreement, repair count,
runtime, and all guidance endpoints are saved for 100 episodes per case.

## Plotting

All figures import the V10 conventions: 3.5/7.16 inch IEEE column widths,
Times-compatible serif text, STIX math, inward ticks, 1.5-point lines,
color-blind-friendly colors, compact legends, PDF/SVG vector output, and
600-dpi PNG output.

## Frozen validation checkpoint selection

Because policy performance can be non-monotonic, the final training update is
not assumed to be the best checkpoint. Named checkpoints are evaluated with a
held-out validation seed block that is disjoint from all final Monte Carlo
test seeds. Selection uses one fixed lexicographic rule for every variant:
maximize mission success, strict target coverage, mean coordinated groups,
mean complete groups, and mean targets covered; then minimize mean closest
approach and maximize mean return. The selected actor and critic are frozen
before the 100-episode test. No test seed participates in checkpoint
selection, and validation performs no gradient, backpropagation, or optimizer
step.

Independent evaluation episodes may be distributed over CPU workers. The
parallel implementation preserves the exact episode-to-seed mapping, merges
raw rows without smoothing or alteration, and recomputes confidence intervals
from the complete merged sample. Serial-versus-parallel verification must
show zero per-episode numerical difference before formal use.
