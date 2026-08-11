# ART-MAPPO failure-case analysis package

This package contains the reproducible Case 1/Case 2 boundary-failure
experiment requested by the reviewer.

## Main deliverables

- `reviewer_response_failure_case_full_v10.tex` and `.pdf`
  - final revised reviewer response using one representative run per case;
  - embeds the complete 13-plot V10 output set for both cases.
- `artifacts_remote/selected_representative_full_v10/`
  - Case 1 seed 76014 and Case 2 seed 77008;
  - 27 PDF/SVG/600-dpi PNG triplets (13 plots per case plus shared legend);
  - V10-format exported data and a reproducibility manifest.
- `artifacts_remote/failure_case_analysis_v10/`
  - six-panel V10 figure in PDF/SVG/600-dpi PNG;
  - two case-wise figures and six per-episode diagnostic figures;
  - episode-, interceptor-, and target-group-level metrics;
  - long-form attacker and interceptor trajectory CSV files.
- `artifacts_remote/selected_six_failure_cases/`
  - authoritative native NPZ trajectories;
  - hit-event CSV files, episode summaries, evaluator-native figures, and logs.
- `artifacts_remote/partial_failure_screen_nominal_n100/`
  - complete 100-episode-per-case screening data, commands, candidates, and logs.
- `artifacts_remote/code/`
  - frozen evaluator copy plus screening, replay, and V10 postprocessing scripts.
- `artifacts_remote/models/`, `presets/`, and `state/`
  - copied frozen model files, verified paper preset, and source hash audit.
- `REVIEWER_RESPONSE_EN.md`
  - proposed closed-loop response to the reviewer.
- `PAPER_INSERTION_EN.tex`
  - manuscript-ready subsection, figure caption, and quantitative table.
- `实验说明与结果分析_CN.md`
  - detailed Chinese protocol, results, interpretation, and usage guidance.

No trajectory was edited or synthesized, and no policy training was performed.
