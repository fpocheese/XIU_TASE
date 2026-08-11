# ART-MAPPO Component Ablation Protocol

## Scope and provenance

- Source project: `/home/a2rl/xiu_onpolicy_3d_fix/stable_V2`.
- Isolated experiment clone:
  `/home/a2rl/reviewer_xiu_ablation_20260729/code`.
- The source project and its trained Case-1/Case-2 checkpoints are read-only
  references and are not overwritten.
- The ablation clone retains the verified three-dimensional dynamics, target
  trajectories, target assignment, reward, action envelope, deterministic
  guidance chain, and residual-action interface.
- Paper mechanisms added to the clone are channel-wise multi-head attention,
  two residual feature blocks, a defender-specific GRU, and return-dependent
  trust-modulated exploration.

## Controlled variants

| Variant | Trust-aware training exploration | GRU | Attention-residual backbone |
|---|---:|---:|---:|
| Full ART-MAPPO | on | on | on |
| No trust | off | on | on |
| No GRU | on | off | on |
| No attention-residual | on | on | off |

The no-attention-residual control uses a capacity-matched plain MLP, avoiding
the confound that a weaker result could be caused merely by fewer trainable
parameters. Full and no-trust use the same architecture and parameter count.
All non-ablated settings are held fixed.

## Training protocol

- Cases: paper Case 1 and Case 2.
- Training seeds: 8301, 8302, 8303.
- Budget: 81,920 environment transitions per variant, case, and seed.
- Rollout buffer: 1,024 transitions; the physical engagement continues across
  PPO updates and resets only after terminal engagement conditions.
- PPO epochs: 5; minibatches: 4; recurrent chunk length: 32.
- Actor and critic learning rate: \(3\times10^{-4}\), cosine annealed.
- Trust initial value: 0.80; trust EMA coefficient: 0.05; trust update
  coefficient: 0.10; sigmoid temperature: 1.0.
- Guided mixture weights: base guidance 0.70, boundary probe 0.20, uniform
  exploration 0.10. The selected action source is held for five 50-ms steps.
- Guided exploration is enabled only during training. It is disabled in every
  validation and test rollout.
- Logged at each PPO update: episode return, critic loss, policy entropy,
  policy loss, approximate KL divergence, gradient norms, trust statistics,
  guided-action fraction, interception rate, and coordination rate.

## Frozen-policy model selection and test

- Named checkpoints are sampled at a preregistered fixed stride.
- Checkpoint selection uses ten held-out validation episodes and a common
  lexicographic rule for every variant. Test seeds are not used for selection.
- The final test pools exactly 100 episodes per variant and case across the
  three training seeds (34/33/33).
- Test episode seeds are paired across all variants.
- Every test adds the same reproducible physical initial-state perturbation
  model and nominal sensor noise.
- Evaluation is deterministic, uses the learned actor alone, and reports
  `training_performed=false`, `optimizer_steps=0`, and
  `backpropagation_performed=false`.

## Reported test metrics

- Target-coverage success: all eight attacking UAVs are intercepted.
- All-defender hit: all 20 assigned interceptors hit their targets.
- Cooperative success: every assigned group is complete and its maximum
  arrival-time spread is no more than 0.5 s.
- \(E_{\mathrm{co\text{-}time}}\): group-averaged mean absolute deviation of
  interceptor hit times from their group mean.
- \(E_n\): group-averaged terminal normal overload over the final 1.0 s.
- \(E_{\mathrm{miss}}\): group-averaged hit distance.
- \(E_t\): group-averaged latest hit time.
- Success-rate uncertainty uses Wilson 95% intervals. Continuous-metric
  uncertainty and plots use the unmodified episode-level samples.

## Figure protocol

- V10/IEEE-TASE style: Times New Roman-compatible serif font, STIX math,
  color-blind-friendly palette, single-column width 3.5 in, double-column
  width 7.16 in.
- Solid curves are three-seed means; shaded bands are inter-seed standard
  deviations.
- Training figures: episode return, critic loss, and policy entropy.
- Test figures: success rates and boxplots for
  \(E_{\mathrm{co\text{-}time}}, E_n, E_{\mathrm{miss}}, E_t\).
- Each figure is exported as PDF, SVG, and 600-dpi PNG.
