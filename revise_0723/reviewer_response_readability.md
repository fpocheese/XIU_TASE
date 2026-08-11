### Reviewer Comment

The presentation of the paper is somewhat hindered by the high density of mathematical
expressions, especially in Sections III and IV, where multiple equations are introduced in
rapid succession with limited intuitive explanation. This can make the paper difficult to
follow, even for technically proficient readers. Additionally, the notation is extensive and
sometimes cumbersome, which further affects readability. While the experimental section is
strong and well-supported by visualizations, some figures are overloaded with multiple curves,
reducing clarity. Overall, the paper demonstrates strong technical depth and clear structure,
but its readability could be improved through better balance between formalism and intuition.

### Response

Thank you for this careful and constructive comment. We agree that the previous version placed
too many core derivations, standard formulas, implementation details, and procedural
descriptions at the same narrative level. This made Sections III and IV appear more complicated
than the underlying information flow. We have therefore substantially reorganized these two
sections, reduced the number of displayed equations, added physical and algorithmic
interpretations after the retained equations, and unified the notation. We also re-examined the
multi-curve simulation figures and clarified the reason for retaining individual-agent curves
in the constraint-verification plots. All revised material is highlighted in blue in the marked
manuscript.

#### 1. Reduction and reorganization of the equations in Section III

Section III has been rewritten around the following causal chain:

`relative engagement geometry -> pairwise interception score -> constrained assignment objective
-> local preference search -> discrete candidate selection -> consensus bidding -> consistent
assignment`.

As a result, the number of numbered equations in Section III has been reduced from 22 to 9
(revised Eqs. (8)--(16), pp. 3--5). The reduction was not achieved by deleting information
required for reproducibility. Instead, repetitive equations were merged, standard operations
were moved into the algorithm box or prose, and statements that could not be rigorously
supported were removed or appropriately weakened. The principal changes are as follows.

1. The original ZEM term, angular term, and combined interception-probability expression have
   been merged into the pairwise score in revised Eq. (8). We now explain immediately after the
   equation that a large predicted miss or an unfavorable heading geometry reduces the score,
   whereas a geometrically favorable pair produces a score close to one. The normalization is
   also changed to RMS scales with a numerical floor. This prevents signed ZEM values from
   cancelling and makes the two physical terms dimensionally comparable.

2. The assignment objective and all feasibility constraints are retained together in revised
   Eq. (9). We added the feasibility condition
   $N_A\leq N_D\leq L_{\max}N_A$ and explicitly stated the conditional-independence assumption
   underlying the product-form survival approximation. This lets the reader understand both
   what is being optimized and when the optimization problem has a feasible solution.

3. The local fitness and target-capacity penalty are combined in revised Eq. (10). The latent
   continuous search variable is now denoted by $\xi_{ij}$, its sigmoid-mapped preference by
   $\pi_{ij}$, and the binary assignment only by $X_{ij}$. This removes the previous ambiguity in
   which the same symbol was used for both a continuous preference and a binary decision.

4. The combat-advantage expression is retained as revised Eq. (11), but its physical factors
   are now identified explicitly as intercept-time compatibility, relative-velocity
   compatibility, encounter geometry, and local competition. Positive-part clipping prevents
   an unfavorable speed or aspect term from changing the intended meaning of the multiplicative
   advantage.

5. Four consecutive DBO update equations have been replaced by the compact implementation
   operator in revised Eq. (12). The rolling, dancing, breeding, and stealing operations are
   explained in one paragraph as local exploitation, controlled perturbation, elite
   recombination, and preference exchange. Their procedural order remains explicit in
   Algorithm 1. This preserves the algorithmic information while allowing the main text to
   emphasize the assignment logic.

6. The previous independent thresholding rule could return either no target or multiple targets
   for one defender. It has therefore been replaced by the one-hot argmax rule in revised
   Eq. (13). The target-specific bid and top-$L_{\max}$ consensus update are then given by revised
   Eqs. (14) and (15), respectively. These equations now connect directly to the one-defender--
   one-target and target-capacity constraints in Eq. (9).

7. The stopping condition is rewritten as the pairwise disagreement measure in revised
   Eq. (16), which directly counts inconsistent communication links. The former abstract
   hierarchical/three-phase equations were deleted because they repeated the equations,
   flowchart, and Algorithm 1 without adding new information.

8. The unsupported $\varepsilon$-Nash statement and the unconditional monotonic-fitness claim
   have been removed. The revised convergence subsection now states the precise scope of the
   result: under a fixed engagement snapshot, connected communication, bounded delay, and
   deterministic tie breaking, winner information reaches a locally stable consensus; no global
   optimum or game-theoretic error bound is claimed. Computational complexity is reported in
   prose with separate search, communication, and memory terms rather than as another displayed
   equation.

These changes retain the equations that define the proposed assignment method while removing
equations that merely restated the same procedure.

#### 2. Reduction and intuitive explanation of the equations in Section IV

Section IV has been reorganized according to the actual training and execution flow:

`physical observation -> attention/residual encoding -> GRU history -> actor/critic ->
trust-modulated training exploration -> rollout and GAE -> dual-clip/KL update -> new policy`.

The number of numbered equations in Section IV has been reduced from 41 to 14 (revised
Eqs. (17)--(30), pp. 5--8). Together, Sections III and IV now contain 23 numbered equations
instead of 63. The detailed changes are as follows.

1. The feature encoder is rewritten in revised Eq. (17) so that each token corresponds to one
   physical observation channel. The Q/K/U projection, masked attention, head concatenation,
   and output projection are consolidated in revised Eq. (18), with all matrix dimensions stated
   explicitly. This corrects the previous dimensional ambiguity and makes it clear that
   attention learns interactions among interpretable guidance quantities rather than among
   arbitrarily reshaped mixed features.

2. Three residual-layer equations are merged into revised Eq. (19). We retain the intuitive
   explanation that the identity shortcut preserves low-level kinematic information and
   provides a direct gradient path. The former Jacobian lower-bound expression was removed
   because such a strict nonvanishing-gradient bound does not generally follow in the presence
   of nonlinear branches and inactive ReLU units.

3. The GRU and shared actor are combined in revised Eq. (20), and the centralized critic input
   is given once in revised Eq. (21). The accompanying text now distinguishes centralized
   training from decentralized execution: the critic receives the joint state during training,
   whereas each deployed actor uses only its local observation and recurrent state.

4. The running return statistics and normalization are consolidated in revised Eq. (22), and
   the trust update is retained as revised Eq. (23). We now state explicitly that trust is a
   training-time exploration variable, not a flight-safety certificate. The guided exploration
   mixture is collected in revised Eq. (24), and deployment is defined by $\beta_i=0$.

5. The previous guided-action definition relied on an undefined action-value function
   $Q_\phi$, although the manuscript defined only the state-value critic $V_\phi$. It also used
   an action minimizing the critic as a supposed safe action. The revised formulation instead
   uses a clipped PN command, a feasible boundary probe whose direction reduces the current LOS
   error, and bounded uniform exploration. This aligns the formula with the stated actor--critic
   architecture and gives every exploratory component a clear tactical meaning.

6. The standard GAE relations are combined in revised Eq. (25). The PPO ratio is renamed
   $\varrho_t$ to distinguish it from the physical distance and reward symbols. Standard clipping
   and dual clipping are consolidated in revised Eq. (26), followed by the adaptive KL rule in
   revised Eq. (27) and the total training loss in revised Eq. (28). We removed the former
   equation claiming that dual clipping forces the probability ratio into a hard two-sided
   interval, because dual clipping bounds the surrogate contribution rather than the realized
   ratio itself. We also clarify that adaptive KL regularization controls policy-update speed;
   it is not, by itself, a structural overload constraint.

7. The four-dimensional observation is now presented in one compact expression, revised
   Eq. (29). The effort component uses the previous command rather than the not-yet-generated
   current action, thereby restoring temporal causality. A denominator floor is added to the
   time-coordination term, and fixed normalization/clipping is stated explicitly so quantities
   with different physical scales can enter the network consistently.

8. The reward is retained as one weighted sum in revised Eq. (30), while the five reward
   components and their physical roles are moved to Table I. The terminal hit reward is defined
   as a one-time event, and the effort term includes both load magnitude and command variation.
   We also clarify that the environment computes the reward; the critic estimates its discounted
   return but does not generate the reward.

9. The evaluation criteria have been moved from the algorithmic development to the simulation
   section, where they are now collected in revised Eq. (31) (p. 9). The coordination metric is
   defined from actual arrival times instead of terminal time-to-go values, the terminal load is
   evaluated over an explicit terminal window, and the miss distance and engagement duration
   are defined separately. This placement and definition make the connection between each metric
   and the reported Monte Carlo results more direct.

For every retained core equation, the surrounding text now explains (i) its physical or
algorithmic input, (ii) the transformation being performed, and (iii) how its output is used by
the next stage. This was our main principle for improving the balance between formalism and
intuition.

#### 3. Simplification and unification of the notation

We conducted a notation-wide consistency review and removed the most consequential symbol
collisions. In particular:

- $N_D$ and $N_A$ are used consistently for the numbers of defenders and attackers/targets;
- $\mathbf p_i$ denotes position, $\xi_{ij}$ a latent target preference, and $X_{ij}$ only a
  binary assignment;
- $p_{ij}^{\mathrm{int}}$ denotes the pairwise interception proxy, while $\chi_{ij}$ denotes
  combat advantage;
- $\mathcal A$ is reserved for the feasible action set and $\boldsymbol\alpha_h$ for attention
  weights;
- $\varrho_t$ denotes the PPO importance ratio, avoiding conflict with range and reward;
- $\varepsilon_{\mathrm{num}}$, $\varepsilon_{\mathrm{con}}$, and
  $\varepsilon_{\mathrm{PPO}}$ separately denote numerical stabilization, consensus tolerance,
  and PPO clipping;
- $c_{\mathrm{DC}}$, $\delta_{\mathrm{KL}}$, and $\beta_{\mathrm{KL}}$ now have distinct and
  unambiguous optimization roles; and
- assignment weights use $\omega$-symbols, whereas reward weights use $\lambda$-symbols.

The corresponding entries in the ART-MAPPO parameter table have also been renamed. In addition,
the former sequential notation list has been replaced by the grouped notation summary in
Table V (Appendix A, p. 18), which separates sets/indices and physical quantities, assignment
quantities, learning quantities, and evaluation quantities. Symbols used only locally are now
defined next to their equations instead of being added to the global notation burden.

#### 4. Treatment of the multi-curve simulation figures

We also carefully reconsidered Figs. 12, 14, 18, and 20, which contain the individual load,
heading, speed, and axial-command histories, as well as the time-to-go plots in Figs. 15 and 21.
During revision, we compared three lower-density alternatives: target-group small multiples,
group means with min--max envelopes, and heat maps. Although these alternatives reduced the
number of visible curves, they introduced an important loss of evidence. In particular, a group
mean or representative subset can conceal a single defender that violates the $\pm1g$ load
limit, while a heat map weakens the direct relationship between load command and the resulting
heading or speed response. The small-multiple arrangement was substantially longer and highly
repetitive for the present 20-defender cases.

For this reason, we deliberately retained the individual-agent curves in the constraint-
verification figures rather than replacing them with averages. These figures are intended to
answer the safety question “does any defender violate the envelope?”, not to require the reader
to identify every trajectory at every instant. Their presentation therefore separates the four
physical quantities into distinct subpanels, uses one shared defender legend and a consistent
defender--color mapping, and reports time-to-go coordination in separate figures. The captions
and accompanying discussion direct attention to the safety bounds, oscillation/saturation
patterns, velocity management, and within-group time-to-go convergence. In addition, Fig. 23
uses compact boxplots from 1000 Monte Carlo trials for the statistical comparison, so the main
quantitative conclusions do not rely solely on visually tracing the individual curves.

We believe that these revisions substantially improve readability while preserving the
technical definitions and the individual-agent evidence needed to support the assignment,
guidance, coordination, and safety claims. We sincerely thank the reviewer for prompting this
comprehensive presentation and notation review.

