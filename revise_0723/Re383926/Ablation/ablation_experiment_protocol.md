# ART-MAPPO Component Ablation Protocol

## Scope and provenance

- Source project: `/home/a2rl/xiu_onpolicy_3d_fix/stable_V2`.
- Isolated experiment clone:
  `/home/a2rl/reviewer_xiu_ablation_20260729/code`.
- The source project and its trained Case-1/Case-2 checkpoints are read-only
  references and were not overwritten.
- The clone retains the verified three-dimensional dynamics, target
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

The no-attention-residual control uses a capacity-matched plain MLP. Full and
no-trust actor parameter counts are 732,774; the no-attention-residual count
is 733,200. All non-ablated settings are held fixed.

## Training protocol

- Cases: paper Case 1 and Case 2.
- Seeds: 8301, 8302, 8303.
- Budget: 81,920 environment transitions per variant, case, and seed.
- Rollout buffer: 1,024 transitions.
- PPO epochs: 5; minibatches: 4; recurrent chunk length: 32.
- Actor and critic learning rate: \(3\times10^{-4}\), cosine annealed.
- Trust initial value: 0.80; trust EMA coefficient: 0.05; trust update
  coefficient: 0.10; sigmoid temperature: 1.0.
- Guided mixture weights: base guidance 0.70, boundary probe 0.20, uniform
  exploration 0.10. The selected source is held for five 50-ms steps.
- Guided exploration is enabled only during training.
- Logged at every update: episode return, Critic Loss, Policy Entropy, policy
  loss, approximate KL divergence, gradient norms, trust statistics,
  guided-action fraction, interception rate, and coordination rate.

## Frozen-policy selection and test

- Eight named checkpoints per run are evaluated on ten held-out validation
  episodes using a common lexicographic rule.
- Test seeds are never used for checkpoint selection.
- The final test pools exactly 100 episodes per variant and case across the
  three training seeds (34/33/33).
- Test seeds are paired across variants.
- Every test uses the same reproducible initial-state perturbations and sensor
  noise.
- Evaluation is deterministic and reports `training_performed=false`,
  `optimizer_steps=0`, and `backpropagation=false`.

## Metrics

- Target-coverage success: all eight attacking UAVs are intercepted.
- All-defender hit: all 20 assigned interceptors hit.
- Cooperative success: every assigned group is complete and its maximum
  arrival-time spread is at most 0.5 s.
- \(E_{\mathrm{co\text{-}time}}\): group-averaged mean absolute deviation
  from group mean hit time.
- \(E_n\): group-averaged terminal normal overload over the final 1.0 s.
- \(E_{\mathrm{miss}}\): group-averaged hit distance.
- \(E_t\): group-averaged latest hit time.
- Success uncertainty uses Wilson 95% intervals.

## Figure protocol

- V10/IEEE-TASE style with Times-compatible serif font and STIX math.
- Color-blind-friendly palette and white background.
- Solid curves: three-seed mean; shaded band: inter-seed standard deviation.
- Export formats: vector PDF, vector SVG, and 600-dpi PNG.
