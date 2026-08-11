# Reproducibility protocol

## Factorial design

The study uses a one-component-at-a-time design in each of the two paper scenarios:

| Variant | Trust-aware | GRU | Attention-residual |
|---|---:|---:|---:|
| Full ART-MAPPO | on | on | on |
| No trust | off | on | on |
| No GRU | on | off | on |
| No attention-residual | on | on | off |

Every cell was trained from scratch for 585 updates (599,040 environment steps) using seed 8303. Case 1 and Case 2 used separate models. This corrects the earlier invalid comparison in which Case-1 training was evaluated in Case 2.

## Why training-domain randomization was required

The original paper preset resets to the same physical initial state in every training episode, whereas Monte Carlo evaluation perturbs initial position, heading, and speed. A frozen Case-2 model trained only on the fixed reset failed under that distribution shift. The formal study therefore randomizes only the training initial physical state using the evaluator's declared perturbation ranges. It does not change reward terms, equations of motion, command limits, hit radius, or success thresholds.

## Validation and test isolation

- Checkpoint validation: 20 episodes per candidate; Case-1 seeds 96001--96020 and Case-2 seeds 96201--96220.
- Formal test: 100 paired episodes per variant and case; exact ranges are recorded in `formal_evaluation_n100/formal_evaluation_manifest.json`.
- Test seeds were not used for checkpoint selection.
- Frozen evaluation records `training_performed=false`, `backpropagation_performed=false`, and `optimizer_steps=0`.
- Paired confidence intervals use 20,000 bootstrap resamples. Binary success differences use the exact McNemar test.

## Metric definitions

- `E_co_time_s`: group-level mean absolute arrival-time deviation, in seconds.
- `E_n_g`: mean resultant normal load in the final one-second window, in g.
- `E_miss_m`: mean first-hit distance, in metres.
- `E_t_s`: time from engagement start to the last group-member arrival, in seconds. It is an engagement-duration measure, not a synchronization spread.
- `target_coverage_success`: all eight attacking UAVs are neutralized.
- `all_defenders_hit`: every assigned interceptor hits.
- `cooperative_success`: every required group satisfies the 0.5-s synchronization criterion.
- `mission_success`: the strict conjunction of target coverage, all-defender hits, and cooperative success.

Terminal metrics are computed only for episodes with finite, eligible terminal events; no failed or missing value is imputed. In this formal test all cells have 100 eligible episodes.

## Training summaries

The numerical training table uses the raw per-update series. The convergence point is the first sustained attainment of 90% of a run's own asymptotic return improvement. Case-1 returns fluctuate around a plateau and never satisfy this sustained criterion, so the corresponding entries are reported as `--`, not fabricated. Return AUC and final-window statistics remain available.

Figures apply a 51-update centered moving average only to training traces, matching the moving-average convention of the V10 plotting family while suppressing the cohort-alternation comb produced by vectorized episode completion. Monte Carlo points, boxplots, tables, confidence intervals, and all conclusions use unsmoothed values.

## Validation

`experiment_control/CORE_VALIDATION.json` verifies 8 training traces, 800 episode rows, 6,400 target rows, 16,000 assignment rows, cell counts, finite numeric fields, nominal figure formats, and 600-dpi PNG metadata. `experiment_control/audit/original_unchanged/baseline_content_result.txt` verifies that the contents of all 7,172 files present in the original source tree at experiment start remain unchanged. The separate full-tree file-set comparison detects later-added cache/result files and is not used as a source-content test.
