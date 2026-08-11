# ART-MAPPO component ablation study

## Goal

Implement and run a convincing independent ablation of the trust-aware exploration
mechanism, GRU temporal encoder, and attention-residual backbone requested by the
reviewer.  The authoritative architecture is the revised paper; the starting training
code is `/home/uav/00gao_xueshu/togsy_2025/0620septimedone/on-policy-main`.

## Required comparisons

1. Full ART-MAPPO.
2. ART-MAPPO without trust-modulated exploration.
3. ART-MAPPO without the GRU temporal encoder.
4. ART-MAPPO without the attention-residual backbone.

All variants must use identical environments, reward, optimizer settings, training
budgets, evaluation seeds, and parameter-independent reporting procedures.

## Success criteria

- The paper-defined trust update and exploration mixture are implemented and tested.
- Attention operates over physical observation-channel tokens rather than a length-one
  pseudo-sequence.
- Each component can be disabled independently without silently changing other settings.
- Multi-seed training and held-out evaluation complete on the remote GPU server.
- Raw per-seed curves, checkpoints, logs, aggregate statistics, confidence intervals,
  effect sizes/significance tests, plots, and a reviewer-ready interpretation are saved.
- No existing remote project or result is overwritten.

## Safety

- Upload to a new timestamped/labelled remote directory.
- Preserve the local source tree unchanged; edits occur in the workspace copy.
- Never invent or manually smooth results.
- Diagnose and rerun failed jobs while retaining failure logs.
