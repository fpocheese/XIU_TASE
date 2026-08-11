# ART-MAPPO ablation: terminal-metric boxplots

This package evaluates the four trained ablation variants with the terminal
metrics used in the simulation section and reproduces the visual language of
`figures_v9/sin_mc_boxplot_compare`: IEEE double-column width, serif/STIX
mathematics, white background, inward ticks, light dashed grids, black medians,
and color-filled boxes.

## Data basis

- Source: the existing deterministic held-out evaluation logs in
  `Re_xiaorong/raw_results/evaluation`.
- Variants: ART-MAPPO, w/o Trust, w/o GRU, and w/o Attention-Residual.
- Cases: Case 1 and Case 2.
- Evaluation size: 5 seeds × 20 episodes = 100 episodes per variant and case,
  800 episodes in total.
- No model was trained or updated and no rollout was rerun.
- The first-hit event logs contain arrival time, terminal resultant load,
  terminal distance, defender/target identity, and seed/episode identity.
- 4,190 complete target-group observations were reconstructed from the raw
  events.

## Metric definitions and an important limitation

For a target group whose assigned members all reached the lethal radius:

- `E_co_time_s`: mean absolute deviation of member arrival times from the group
  mean, matching the revised manuscript definition.
- `E_miss_m`: mean recorded distance at the first lethal-radius entry.
- `E_t_s`: last member arrival time measured from `t0=0`.
- `E_n_terminal_sample_g`: mean resultant load recorded at arrival.

The latest manuscript defines `E_n` as a terminal-*window* average. The existing
800-episode event logs preserve only the terminal sample, not every load sample
inside that window. Therefore the boxplot explicitly labels this quantity
`E_n (terminal sample, g)`. It is a faithful estimator from the available test
data, but must not be described as the exact terminal-window average.

## Statistical strata

- Case 1: boxplots use only strict episodes in which all eight assigned target
  groups were complete. Sample sizes are 90, 90, 92, and 81 for ART-MAPPO,
  w/o Trust, w/o GRU, and w/o A-R.
- Case 2: no variant produced a strict all-eight-groups-complete episode.
  Its boxplot is therefore a diagnostic conditional distribution over episodes
  containing at least one complete group. Sample sizes are 100, 82, 100, and
  99. The accompanying completion table must be reported with this figure to
  expose the conditioning and avoid survivor-bias interpretation.

## Main files

- `output/ablation_terminal_metrics_boxplot_compare.{pdf,svg,png}`: combined
  two-case figure.
- `output/case1_ablation_metric_boxplot_compare.{pdf,svg,png}`: strict Case 1
  comparison.
- `output/case2_ablation_metric_boxplot_compare.{pdf,svg,png}`: Case 2
  complete-group-conditioned diagnostic.
- `output/ablation_terminal_metrics_group.csv`: one row per complete target
  group, preserving raw provenance.
- `output/ablation_terminal_metrics_episode.csv`: one row per original
  evaluation episode.
- `output/ablation_terminal_metrics_summary.csv`: N, mean, SD, median, IQR,
  and bootstrap 95% confidence intervals.
- `output/ablation_terminal_metrics_paired_effects.csv`: paired differences,
  bootstrap confidence intervals, paired effect size, and Wilcoxon p-values.
- `output/ablation_completion_summary.csv`: unconditional hit, target,
  coordinated-target, and complete-group rates.
- `output/validation_report.json`: coverage and finite-value checks.
- `output/sha256_manifest.csv`: integrity hashes.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/codex_mpl_metric \
python3 plot_ablation_terminal_metrics.py \
  --output-dir output
```

The script automatically discovers all 40 evaluation runs and fails if a run
or episode is missing. No plotted value is typed by hand.
