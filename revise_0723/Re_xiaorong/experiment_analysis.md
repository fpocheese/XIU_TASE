# Detailed analysis of the ART-MAPPO component ablation

## 1. Controlled design

| Variant | Trust-aware training | GRU | Attention-residual backbone |
|---|---:|---:|---:|
| Full ART-MAPPO | on | on | on |
| w/o trust-aware | off | on | on |
| w/o GRU | on | off | on |
| w/o attention-residual | on | on | off |

The attention-residual replacement is parameter matched. Every other setting
is fixed. Training uses paired seeds 701–705 in both cases. Case 1 has 600,000
environment steps; Case 2 has 1,800,000. Evaluation uses 20 common held-out
episodes per seed/case/variant and deterministic actions. The total evaluation
sample is 800 episodes.

## 2. Statistical protocol

- Independent unit: one training seed within a case.
- Paired unit: the same seed and case for full versus one ablation.
- Interval: 95% Student-\(t\) CI across five seed-level estimates.
- Test: two-sided exact sign-flip test over matched seed–case differences.
- Multiplicity: Holm correction over the three component removals separately
  for each metric.
- Effect size: paired Cohen's \(d_z\).

Episode samples generated from the same trained seed are not treated as 20
independent training replicates. This avoids pseudo-replication.

## 3. Training-return evidence

### Case 1 final return

| Variant | Mean final return (\(\times10^6\)) | 95% CI half-width (\(\times10^6\)) |
|---|---:|---:|
| Full | 164.555 | 0.491 |
| w/o trust-aware | 158.987 | 7.664 |
| w/o GRU | 163.620 | 1.677 |
| w/o attention-residual | 161.590 | 2.047 |

Relative to the ablated models, the full model is higher by 3.50%, 0.57%, and
1.84%, respectively. The no-trust dispersion is much larger: converting the
reported CI widths back to sample standard deviations gives approximately
0.396 million for full versus 6.174 million for no-trust, a factor of about
15.6. This indicates a trust-related stability effect in the easier case,
although the paired pooled trust test is not significant.

### Case 2 final return

| Variant | Mean final return (\(\times10^6\)) | 95% CI half-width (\(\times10^6\)) |
|---|---:|---:|
| Full | 162.107 | 1.021 |
| w/o trust-aware | 166.280 | 2.156 |
| w/o GRU | 161.728 | 0.146 |
| w/o attention-residual | 161.654 | 1.258 |

The no-trust variant has a higher scalar training return in Case 2 despite
lower task-aligned target interception and synchronization rates. This is
important: scalar return and terminal mission metrics do not rank every
variant identically. The response therefore reports both and does not use
return alone as evidence of operational superiority.

### Pooled paired final-return comparisons

| Comparison | Full minus ablation | 95% CI half-width | \(d_z\) | raw \(p\) | Holm \(p\) |
|---|---:|---:|---:|---:|---:|
| Full vs w/o trust-aware | \(0.697\times10^6\) | \(4.898\times10^6\) | 0.102 | 0.7598 | 0.7598 |
| Full vs w/o GRU | \(0.657\times10^6\) | \(0.692\times10^6\) | 0.678 | 0.0645 | 0.1289 |
| Full vs w/o attention-residual | \(1.709\times10^6\) | \(1.383\times10^6\) | 0.884 | 0.0156 | **0.0469** |

Only the attention-residual removal reaches the corrected 0.05 threshold.

## 4. Held-out mission metrics

### Case 1

| Variant | Per-target interception | Per-target synchronization | All-target interception | All-target synchronization |
|---|---:|---:|---:|---:|
| Full | 100.000% | 98.625% | 100.0% | 90.0% |
| w/o trust-aware | 100.000% | 98.750% | 100.0% | 90.0% |
| w/o GRU | 100.000% | 99.000% | 100.0% | 92.0% |
| w/o attention-residual | 99.750% | 97.500% | 98.0% | 81.0% |

Case 1 is close to saturation, so it has limited power to distinguish the
first three models using success-rate metrics. The attention-residual
ablation is the only variant with a visible loss in strict all-target
performance.

### Case 2

| Variant | Per-target interception | Per-target synchronization | All-target interception | All-target synchronization |
|---|---:|---:|---:|---:|
| Full | 44.625% | 30.500% | 0% | 0% |
| w/o trust-aware | 34.000% | 24.000% | 0% | 0% |
| w/o GRU | 50.125% | 33.250% | 0% | 0% |
| w/o attention-residual | 38.625% | 23.625% | 0% | 0% |

The full model gains 10.625 interception points and 6.5 synchronization
points over no-trust, and 6.0 and 6.875 points over the
attention-residual ablation. In contrast, no-GRU exceeds full by 5.5 and
2.75 points. Across both cases, none of the target-level comparisons survives
Holm correction; the difficult-case differences are therefore descriptive,
not confirmatory.

The strict Case 2 metric is zero for all variants. The lower panels of the
figure consequently show per-target rates, which remain informative under
this floor effect. The strict indicators remain present in:

- every episode-level evaluation CSV;
- `ablation_seed_level_metrics.csv`;
- `ablation_aggregate_metrics.csv`;
- `ablation_paired_statistics.csv`.

## 5. Component-by-component conclusion

### Attention-residual backbone

This component has the strongest evidence. Its removal causes a significant
pooled final-return loss after multiplicity correction, a large paired effect,
and lower difficult-case interception and synchronization. Because the
replacement MLP is parameter matched, the result is attributable to topology
rather than capacity.

### Trust-aware mechanism

The trust mechanism improves the difficult-case task-aligned metrics and
strongly reduces Case 1 return dispersion. However, the pooled corrected
tests are not significant and Case 2 scalar return favors no-trust. The
defensible wording is “directional operational and stability evidence,” not
“universally significant improvement.”

### GRU temporal encoder

The full model has a positive pooled final-return difference and
\(d_z=0.678\), but the corrected test is not significant. The no-GRU model is
better on Case 2 target-level metrics. The data therefore do not demonstrate
an independent held-out GRU benefit. One plausible interpretation is that the
17-dimensional observation exposes sufficient instantaneous time-to-go and
geometry; this is an inference, not a measured causal mechanism.

## 6. What should and should not be claimed

Supported:

- a rigorous single-component ablation was performed;
- attention-residual encoding has statistically supported independent value;
- trust-aware training shows difficult-case target-level and easy-case
  stability benefits, with uncertainty explicitly quantified;
- GRU advantage is conditional and not significant in this experiment.

Not supported:

- that all three modules are individually significant;
- that the full method solves strict all-eight-target Case 2 at this budget;
- that 800 episodes equal 800 independent training seeds;
- that scalar training return alone establishes mission superiority.

This restrained interpretation is the most credible response to the reviewer
and prevents overclaiming.
