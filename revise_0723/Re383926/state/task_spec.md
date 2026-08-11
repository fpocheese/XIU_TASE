# Re383926 reviewer-experiment task specification

## Scope

1. Improve the full ART-MAPPO implementation under a fair experimental
   protocol and rerun the full/no-trust/no-GRU/no-attention-residual ablation.
2. Use 100 held-out Monte Carlo episodes per variant/scenario.
3. Add reviewer experiments:
   - 3.8 failure cases: unsuccessful interception and delayed cooperation;
   - 3.9 generalization to unseen manoeuvring attack patterns;
   - 1.6 end-to-end task allocation plus cooperative-guidance verification.
4. Deliver code, raw data, plots, statistical analysis, reviewer responses, and
   manuscript-ready text to local `Re383926`.

## Integrity constraints

- No fabricated or edited experimental observations.
- No deliberate weakening or variant-specific under-training of ablations.
- Same environment, budget, seeds, evaluator, and stopping rule across variants.
- Architectural removal is the only intended between-variant difference.
- Conclusions follow measured evidence; mixed or negative findings are retained.

## Completion evidence

- Remote commands, configs, git/source hashes, checkpoints, and logs preserved.
- Exactly 100 test episodes per required variant/scenario unless a documented
  runtime failure makes an episode invalid and it is rerun.
- Raw episode-level CSV/NPZ data and reproducible plotting scripts.
- Failure taxonomy with representative traces.
- At least two unseen manoeuvre families plus severity/OOD metadata.
- End-to-end comparison includes assignment output and downstream guidance
  success for the same missions.
- Figures inspected, NaN/Inf checks passed, hashes generated.
- All supplementary-experiment figures follow the V10 plotting implementation
  under `XIU_tase_paper_V1/new_sim_fig/fig_and_data`, including typography,
  physical dimensions, colors, line widths, legends, and vector/raster export.
- English reviewer responses and manuscript-ready insertion text.
