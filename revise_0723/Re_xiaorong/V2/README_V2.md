# ART-MAPPO component ablation — V2 delivery

V2 aligns the ablation interpretation with the mechanisms claimed in the
original manuscript:

- trust-aware exploration is evaluated primarily through policy-entropy
  evolution, reward stability, and the transition from exploration to
  exploitation;
- the GRU and attention-residual backbone are evaluated from both training
  behavior and post-training Monte Carlo interception performance;
- \(E_n\), \(E_{\mathrm{miss}}\), and \(E_t\) are intentionally outside this
  ablation table, as requested.

## Evaluation protocol

Every variant/case result pools 100 held-out Monte Carlo trials from five
independently trained seeds, with 20 trials per seed. The four variants and
two cases therefore contain 800 evaluation episodes in total. Evaluation uses
deterministic policy inference and disables trust-guided exploration.

The strict trial-level quantities are:

- interception success: all assigned targets are intercepted in a trial;
- coordination success: all assigned targets also satisfy the synchronization
  condition.

Per-target rates are retained as secondary diagnostic metrics when the strict
all-target indicator has a floor effect.

## New training figures

Three standalone figures, each containing Case 1 and Case 2, are provided in
PDF, SVG, and 600-dpi PNG:

1. `figures/ablation_training_reward.*`;
2. `figures/ablation_critic_loss.*`;
3. `figures/ablation_policy_entropy.*`.

A six-panel convenience figure is provided as
`figures/ablation_training_metrics_combined.*`.

## Data adaptation

`data/converted_npy/` follows the `simple_converge_v7` interface, with one
file per variant, seed, metric, and case:

- `_rewards.npy`;
- `_critic_loss.npy`;
- `_entropy.npy`;
- `_steps.npy`.

`data/plot_csv/` stores the plotted curves using:

`environment_steps_k, mean, shadow_lower, shadow_upper`

plus the seed count and smoothing-window metadata. The original raw values are
unchanged. Case 1 uses a 5-update trailing average and Case 2 a 15-update
average, both equal to 5% of the respective training horizon. The shaded
region is the 95% Student-\(t\) confidence interval over five independent
training seeds.

## Research-integrity note

The old `simple_converge_v7/plot_results_ieee_v3.py` contains
method-specific ideal saturation, curve blending, variance annealing, and
synthetic random noise. V2 uses its publication style and file interface only.
No ideal curve, injected noise, hand-adjusted point, or method-specific
post-processing is used.
