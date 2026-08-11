# ART-MAPPO case-specific ablation study (V3, 2026-08-01)

## Latest compact Case-1 manuscript update

Following the final manuscript-format instruction, the highlighted paper now
contains one compact ablation subsection after the two scenario descriptions.
It does not split the ablation table by scenario and includes no ablation
boxplot or training figure.  Only Case 1 is reported in the compact table.

- Four rows: complete ART-MAPPO and the three single-component ablations.
- Two training columns: horizon-normalized Training AUC and the standard
  deviation of return over the final 58 updates.
- Five test columns: interception success rate, $E_n$, $E_{miss}$,
  $E_{co\text{-}time}$, and $E_t$.
- Exactly 1000 paired Case-1 frozen-policy trials per algorithm, assembled from
  three disjoint seed blocks: 98401--98500 (100), 100001--100500 (500), and
  102001--102400 (400).
- The final 4000-row table contains Case 1 only; all required values are finite,
  every algorithm has 1000 unique seeds, and every algorithm records 1000/1000
  complete interceptions.

Authoritative compact-update files are under `analysis/compact_case1_n1000/`,
`formal_evaluation_case1_n500/`, `formal_evaluation_case1_n400/`, `tables/`,
and `manuscript/`.

This directory is the complete local delivery for the reviewer-requested ablation of the trust-aware mechanism, GRU temporal encoder, and attention-residual backbone. All reported numbers are computed from the recorded training traces or frozen-policy Monte Carlo trials. No values were manually altered.

## What was run

- Eight independent training jobs: four variants (`full`, `no_trust`, `no_gru`, and `no_attention_residual`) trained separately in Case 1 and Case 2.
- One fixed training seed per cell (`8303`), 585 updates and 599,040 environment steps per job.
- Training-only domain randomization used the same physical initial-state perturbation ranges as evaluation; rewards, dynamics, hit radius, policy weights, and test definitions were not relaxed.
- Frozen checkpoint selection used 20 validation episodes on seeds disjoint from the formal test seeds.
- Formal evaluation used exactly 100 paired Monte Carlo episodes per variant and case (800 total); the same 100 seeds were used for all variants within a case. No optimizer step or backpropagation occurred in evaluation.

## Principal result

All eight cells achieved 100/100 target coverage, 100/100 all-defender interception, 100/100 cooperative success, and 100/100 strict mission success (Wilson 95% interval 96.3--100%). This nominal success-rate ceiling means that the modules are distinguished primarily by training dynamics and continuous terminal-quality metrics, not by claiming an unsupported success-rate advantage.

- Removing the GRU increased final-window critic loss from 0.0583 to 0.3095 in Case 1 (5.31x) and from 0.0348 to 0.2330 in Case 2 (6.69x), identifying the GRU's clearest contribution as value-learning stability.
- Removing trust reduced the final policy entropy from 1.1590 to 0.5172 in Case 1 and from 1.3013 to 0.7561 in Case 2, while the guided-action fraction changed from about 0.53 to zero. This supports the manuscript's training-side interpretation of trust-aware exploration.
- The capacity-matched no-attention-residual control stayed near the full model in return AUC and nominal success. Its measurable effects are secondary: lower final entropy in Case 2 (1.1918 versus 1.3013) and slightly larger terminal load in Case 1 (mean 0.0340 g versus 0.0316 g). The data therefore support a modest representation/safety-margin contribution, not a universal performance advantage.

## Directory map

- `analysis/`: authoritative machine-generated summary CSVs, paired bootstrap comparisons, and integrity audit.
- `training/`: all eight raw per-update `training_metrics.csv` traces plus the exact command configuration and run manifest for each cell.
- `formal_evaluation_n100/combined/`: authoritative 800-row episode table, 6,400-row target table, and 16,000-row assignment table.
- `formal_evaluation_n100/evaluation/`: per-worker raw evaluation outputs used to build the combined tables.
- `formal_evaluation_n100/selection/`: validation records, selection JSONs, and the eight selected frozen actor/critic models; repeated candidate-checkpoint model copies are intentionally omitted.
- `figures_v10/`: 12 figure stems, each in PDF, SVG, and 600-dpi PNG, plus `plot_manifest.json`.
- `tables/`: publication-ready CSV and LaTeX tables.
- `scripts/`: training, domain-randomization, frozen evaluation, analysis, table-generation, plotting, and validation code.
- `experiment_control/`: formal protocol, launch/supervision logs, completion markers, checksums, core validation, and original-code hash audit.
- `reports/`: English/Chinese reviewer responses, detailed interpretation, reproducibility notes, and completion audit.
- `manuscript/`: LaTeX insertion and backups/copies associated with the highlighted manuscript update.

## Authoritative files to read first

1. `reports/experiment_results_analysis_zh.md`
2. `reports/training_only_metrics_analysis_zh.md`
3. `reports/reviewer_response_en.tex`
4. `analysis/training_effect_summary.csv`
5. `tables/paper_training_core_metrics.csv`
6. `analysis/monte_carlo_n100_summary.csv`
7. `analysis/paired_test_comparisons.csv`
8. `experiment_control/CORE_VALIDATION.json`

## Intermediate checkpoints

The 7.0 GB sequence of intermediate checkpoints is retained on the remote server at:

`/home/a2rl/reviewer_xiu_ablation_domainrand_v3_20260801/training`

It is not duplicated here because the delivery already contains every raw numeric trace, every selected checkpoint, and every formal validation/test output. The original remote source directory was `/home/a2rl/xiu_onpolicy_3d_fix`; the content hashes of all 7,172 files present at experiment start passed the end-of-run verification in `experiment_control/audit/original_unchanged/baseline_content_result.txt`. The full file-set comparison is reported separately because caches/results were added to that tree by other activity after the baseline was recorded.
