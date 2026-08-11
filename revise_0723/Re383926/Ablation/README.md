# ART-MAPPO component ablation: audited deliverables

This directory contains the complete, unmodified output of the controlled
ART-MAPPO component-ablation experiment performed on the remote server.

## Experimental design

- Source reference: `/home/a2rl/xiu_onpolicy_3d_fix/stable_V2`.
- Isolated remote clone: `/home/a2rl/reviewer_xiu_ablation_20260729/code`.
- Variants: full ART-MAPPO, no trust-aware exploration, no GRU, and no
  attention--residual backbone.
- Cases: Case 1 and Case 2.
- Training seeds: 8301, 8302, and 8303.
- Training budget: 81,920 environment steps for each variant/case/seed
  combination (24 completed runs).
- Frozen-policy evaluation: 100 paired Monte Carlo episodes for each
  variant/case combination (800 episodes in total).
- Evaluation performs no optimization, back-propagation, or model update.

The no-attention--residual actor is a capacity-matched MLP
(733,200 parameters versus 732,774 for the full actor). The no-GRU actor has
337,510 parameters. Full and no-trust have identical architectures.

## Directory map

- `formal_training/`: per-update training CSV files, configurations,
  manifests, logs, and final actor/critic checkpoints for all 24 runs.
- `formal_evaluation/`: validation records, selected frozen checkpoints,
  per-episode test records, and evaluation logs.
- `evaluation_combined/`: combined 800-episode, assignment, and target CSVs.
- `analysis/`: audited summary tables and integrity checks.
- `figures_v10/`: V10/IEEE-style PDF, SVG, and 600-dpi PNG figures.
- `code/`: exact remote experiment and plotting code snapshot.
- `logs/`: top-level remote completion, evaluation, plotting, and analysis
  logs.
- `reviewer_response_ablation_en.md`: reviewer-response text based only on
  the audited data.
- `manuscript_ablation_insertion.tex`: standalone LaTeX insertion draft; the
  original manuscript was not modified.
- `ablation_results_analysis_zh.md`: detailed Chinese interpretation.
- `paper_ready_ablation_success_table.csv`: compact primary test table.

## Integrity result

The formal audit records 24 training runs and exactly 800 test episodes.
Training CSVs contain no NaN/Inf. Evaluation manifests report
`training_performed=false`, `optimizer_steps=0`, and
`backpropagation=false`. The original stable actors were checked after the
experiment:

- Case 1 actor SHA-256:
  `884802798c6c4b27d046bae274a3d354702d6e8646a73875de940202713393a3`
- Case 2 actor SHA-256:
  `8a644cfe9fea7f70fe0165c25d77193887e9c2b9e391ac83ca5efc85714fd0f2`

## Interpretation boundary

The formal results support an independent Case-1 contribution from the
trust-aware training mechanism and the attention--residual backbone. They
also show that the GRU substantially improves critic fitting, but they do not
show a frozen-policy coordination advantage for the GRU under this training
budget and observation setting. In Case 2 no variant satisfies the strict
all-group 0.5-s coordination criterion. These negative results are retained
and discussed explicitly; no episode or plotted value was edited.
