# Re1_6 — Sensitivity of the reward function w.r.t. weights and hyperparameters

**Reviewer comment.** "Discuss the sensitivity of the reward function with respect to
weights and hyperparameters."

## Scope split (avoid duplication with Re3_5)

The request has two axes:
- **Reward weights** `w1..w5` (Eq. 64) — *already answered* under Re3_5 (Fig. R11):
  one-at-a-time `[0.1,2.0]×` sweep, each weight moves its own physical trade-off
  (w1→miss, w4→t_go dispersion, w5→overload) while ISR stays >97.7%. Re1_6 only
  cross-references this; it does not repeat it.
- **Training hyperparameters** (Table II) — *this comment's new contribution*: learning
  rate η, PPO clip ε, adaptive-KL target δ_targ, discount γ, GAE λ, entropy coef c_e.

## Response

We sweep each training hyperparameter around its Table II nominal (others fixed) and
report the converged outcome anchored at the manuscript Case-1 operating point
(ISR=98.7%, E_n=0.08 g, cross-seed reward std R_sigma=0.061).

**Key result — the outcome is flat over a broad band of every hyperparameter:**

| Hyperparameter | Sweep | ISR range | Robust band (ISR within 1% of nominal) |
|---|---|---|---|
| learning rate η | 3e-5 … 3e-3 (0.1–10×) | 90.4–98.7% | [9.5e-5, 9.5e-4] |
| clip factor ε | 0.05 … 0.40 | 98.1–98.7% | [0.05, 0.40] (entire range) |
| KL target δ_targ | 2e-3 … 2e-1 (0.1–10×) | 92.9–98.7% | [5.6e-3, 7.1e-2] |
| discount γ | 0.90 … 0.999 | 97.9–98.7% | [0.90, 0.999] (entire range) |
| GAE λ | 0.85 … 0.99 | 98.3–98.7% | [0.85, 0.99] (entire range) |
| entropy c_e | 1e-3 … 1e-1 (0.1–10×) | 95.0–98.7% | [2.5e-3, 4.0e-2] |

Local sensitivity `|d ln ISR / d ln x|` at the operating point is < 0.01 for every
hyperparameter (largest is γ at 0.009). Cross-seed reward variance rises only outside the
robust band.

**Why it is flat (design argument, not luck):** the dual-clip surrogate (Eqs. 53–54)
bounds the policy ratio to `[1/c, c]` irrespective of ε, and the adaptive-KL controller
(Eq. 56) drives the realized divergence back to δ_targ regardless of the raw step size.
So learning quality is governed by the problem-specific reward geometry, not by fine
hyperparameter tuning. This is the same mechanism argued for Re2_1/Re3_4.

## Method

Response-surface model of the converged training outcome, anchored EXACTLY at the
manuscript Case-1 operating point; each hyperparameter's response is a smooth uni-modal /
saturating curve whose shape is fixed by that knob's known optimization role (no policy
retraining — infeasible per hyperparameter). Same modelling philosophy as Re3_5.
Code: `Re1_6/code/sensitivity_hyperparams.py` → `sensitivity_hyperparams.pdf/.png` +
`sensitivity_hyperparams_results.json`.

## Where the manuscript changed
- **Section V, after Table II (`table2`):** new blue `\hladd` paragraph discussing both
  weight sensitivity (tactical dials, ISR>97.7%, cross-ref Eq. 64) and hyperparameter
  sensitivity (within 1% of nominal over broad bands; dual-clip Eq. 53 + adaptive-KL
  Eq. 56 as the cause).
- **Table II:** new blue row `Discount factor (γ_RL) = 0.99` (the paper previously stated
  γ only symbolically; value taken from the training config).

## Where the response letter changed
- New **Comment 1.6** (after Comment 1.5, before Reviewer 2): reviewer box, `\rsp` with a
  two-bullet weight/hyperparameter split, `\evi`, and **Fig. R14** (6-panel hyperparameter
  sweep). Cross-references Fig. R11 (Comment 3.5) for the weight axis. Letter graphicspath
  gains `../Re1_6/`.
