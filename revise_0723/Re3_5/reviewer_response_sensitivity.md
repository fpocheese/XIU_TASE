# Re3_5 — Sensitivity analysis for the weighting coefficients

**Reviewer comment.** "Meanwhile, it is important to provide sensitivity analysis for
weighting coefficients in the block probability model and reward formulation because
cooperative interception performance varies under different tactical priorities."

## Response

We added two sensitivity studies that quantify how cooperative interception performance
responds to the two coefficient sets the reviewer names: the weights `(w1, w2)` of the
block (interception) probability model in Eq. (12), and the weights `(w1..w5)` of the
reward formulation in Eq. (64). Both studies use the paper's scenario geometry (20
interceptors vs. 8 maneuvering attackers), and both are reported with figures in the
response letter (Fig. R10, Fig. R11) plus two highlighted paragraphs in the manuscript.

### (a) Block-probability weights (w1, w2), w2 = 1 - w1

We sweep `w1` from 0 to 1 over 12 randomized geometries and solve the capacitated
many-to-one assignment (1 ≤ load ≤ L_max = 3) at each setting. The two objectives trade
off monotonically:

- **Miss distance** (mean selected ZEM) drops **56.9%**, from 1152 m to 676 m, as `w1`
  moves toward the ZEM term (reachability priority).
- **Terminal aspect error** rises from **61.5° to 93.5°** over the same sweep; weighting
  the geometry term P_σ instead minimizes aspect error.
- **Aggregate effectiveness is robust.** Expected surviving targets `J` stays below 0.18
  across the whole sweep (> 97.7% expected neutralization), and the weakest-target
  coverage probability varies by under 3%.

Interpretation: `(w1, w2)` is a **tactical dial** (reachability vs. firing-geometry
quality), not an effectiveness switch. The near-balanced `w1 ≈ 0.55` adopted in the
paper is a favorable compromise between the two objectives.

### (b) Reward weights (w1..w5)

We scale each reward weight over [0.1, 2.0]× its nominal value, holding the others
fixed, and track the four evaluation metrics (E_co-time, E_n, E_miss, ISR). Each term
dominates its intended tactical dimension; normalized local sensitivities
|d ln M / d ln w| at the nominal operating point:

- **w1 (distance)** → terminal miss distance E_miss, sensitivity ≈ **0.76** (highest for E_miss).
- **w5 (energy)** → terminal normal overload E_n, sensitivity ≈ **0.71** (highest for E_n).
- **w4 (coordination)** → temporal coordination E_co-time, sensitivity ≈ **0.78** (highest for E_co-time).
- **w3 (hit)** and **w2 (angle)** → reinforce interception success rate and heading alignment.

Interpretation: because the metric–weight coupling is monotone and well separated, the
operating point can be shifted toward precision, safety, or temporal-coordination
priorities without destabilizing the others. This supports the balanced nominal setting
adopted in the paper.

## Files

- `code/sensitivity_block_prob.py` — block-probability sweep + capacitated assignment;
  outputs `sensitivity_block_prob.pdf/.png` and `sensitivity_block_prob_results.json`.
- `code/sensitivity_reward.py` — reward-weight sensitivity; outputs
  `sensitivity_reward.pdf/.png` and `sensitivity_reward_results.json`.

## Manuscript changes (new_highlight/main.tex)

- Section III-A: highlighted paragraph after the assignment formulation, block-probability
  `(w1, w2)` sensitivity.
- Section IV-E: highlighted paragraph after the reward-component definitions, reward-weight
  `(w1..w5)` sensitivity.

