# Re383926 reviewer-experiment deliverables

All simulation, training, evaluation, and plotting jobs were executed on the
remote server. This directory contains the complete returned artifacts.

## 1. `Ablation/`

Controlled ART-MAPPO component ablation:

- Full ART-MAPPO;
- no trust-aware training mechanism;
- no GRU temporal encoder;
- no attention--residual backbone, replaced by a capacity-matched MLP.

It contains all 24 formal training runs, 800 frozen-policy test episodes,
training and test figures in PDF/SVG/600-dpi PNG, raw CSVs, selected models,
code, logs, integrity audit, Chinese analysis, an English reviewer response,
and a standalone manuscript insertion draft.

## 2. `Case3/`

New end-to-end IDBO target-allocation plus frozen ART-MAPPO guidance condition
with a geometry and hybrid attacker maneuver distinct from Cases 1 and 2.

For paper illustration, five reproducible complete successes are supplied in
`Case3/five_successful_trials/` for seeds 74001, 74002, 74003, 74005, and
74009. Every trial includes raw NPZ/CSV trajectories and 3-D, timing, and
combined figures in PDF/SVG/600-dpi PNG. Seed 74001 is the objective
recommendation because it has the smallest worst-group spread (0.05 s) and
the best fixed ranking criterion.

The original 100-run audit is retained as reliability evidence: target
interception was 100/100 and strict all-group coordination was 72/100. The
remaining 28 were timing-only failures. The five-trial display set does not
replace or alter that audit.

## 3. `FailureAnalysis/`

Audited analysis for unsuccessful interception and delayed cooperative
engagement:

- Case-2 full model: 21 all-target-coverage failures, usually 7/8 targets;
- Case 3: 28 timing-only failures despite 8/8 target interception.

It contains the source CSV subsets, target-frequency tables, summary JSON,
V10 PDF/SVG/600-dpi PNG figure, Chinese explanation, and English reviewer
response.

## Recommended paper figures

- Case-3 representative:
  `Case3/five_successful_trials/seed_74001/seed_74001_combined.pdf`
- Case-1 training ablation:
  `Ablation/figures_v10/ablation_training_case1_v10.pdf`
- Case-2 training ablation:
  `Ablation/figures_v10/ablation_training_case2_v10.pdf`
- Case-1 test ablation:
  `Ablation/figures_v10/ablation_monte_carlo_case1_v10.pdf`
- Failure modes:
  `FailureAnalysis/failure_case_analysis_v10.pdf`

## Scientific interpretation

The data support the Case-1 contributions of trust-aware training and the
attention--residual backbone, and show a strong GRU contribution to critic
fitting. They do not support a universal GRU deployment advantage, because
the no-GRU variant performs better in the current Case-1 frozen-policy test.
Case 2 also remains a limitation under the strict all-group criterion. These
results are reported without editing or suppressing unfavorable observations.
