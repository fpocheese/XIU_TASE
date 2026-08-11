# Completion audit

Date: 2026-08-01 (Asia/Shanghai)

## Required experiment matrix

- [x] Full, No trust, No GRU, and No attention-residual trained in Case 1.
- [x] Full, No trust, No GRU, and No attention-residual trained independently in Case 2.
- [x] Eight raw training traces; each has 585 updates and ends at 599,040 environment steps.
- [x] One declared training seed per cell (8303); no unsupported multi-seed claim.
- [x] Case-1 and Case-2 models are distinct, not one model reused across cases.

## Evaluation isolation and counts

- [x] Frozen checkpoint selection used 20 validation episodes and seeds disjoint from formal tests.
- [x] Formal evaluation performed no training, backpropagation, or optimizer step.
- [x] Exactly 100 paired Monte Carlo episodes per variant and case (800 total).
- [x] Combined raw tables contain 800 episode, 6,400 target, and 16,000 assignment rows.
- [x] All four terminal metrics are finite in every formal episode; no imputation was used.
- [x] Success-rate and continuous paired comparisons were generated automatically from raw rows.

## Outputs

- [x] Training reward, Critic loss, and Policy entropy curves for both cases.
- [x] Target-coverage/cooperative success plots for both cases.
- [x] Four-metric boxplots for both cases.
- [x] Every figure is available as PDF, SVG, and 600-dpi PNG (12 stems, 36 graphics).
- [x] Publication tables are available in CSV and LaTeX.
- [x] Eight selected actor/critic pairs and checkpoint-selection records are included.
- [x] English and Chinese reviewer responses are included.
- [x] Detailed Chinese analysis and reproducibility protocol are included.
- [x] Highlighted-manuscript insertion replaces the superseded 5-seed/1000-trial claims.

## Integrity and interpretation

- [x] Remote core validator reports `PASS`.
- [x] Local curated-delivery validation reports 8/800/6400/16000 expected counts, finite metrics, complete figures, and complete selected models.
- [x] Content hashes of all 7,172 files present in the original source tree at experiment start pass end-of-run verification.
- [x] Later-added files in the original tree are distinguished from content changes in the audit.
- [x] No data were fabricated or manually modified to force the full model to rank first.
- [x] The 100% nominal success-rate ceiling is reported rather than hidden.
- [x] The Case-1 convergence field is reported as unavailable because the sustained criterion is not met.
- [x] The one-training-seed limitation and the difference between environmental and training-seed uncertainty are explicit.

## Storage note

All raw numeric training/evaluation data, selected models, figures, scripts, reports, and audit records are in this delivery. The complete 7.0 GB intermediate-checkpoint sequence remains available on the remote server at `/home/a2rl/reviewer_xiu_ablation_domainrand_v3_20260801/training`; it is omitted locally only to avoid dozens of redundant historical model copies.
