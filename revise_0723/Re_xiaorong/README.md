# ART-MAPPO component-ablation delivery

This folder closes the reviewer request:

> “Then, it would be helpful to add an ablation investigation demonstrating
> the independent contribution of the trust-aware mechanism, GRU temporal
> encoder, and attention-residual backbone within ART-MAPPO architecture.”

## What is included

- `reviewer_response_en.md`: submission-ready English response;
- `审稿意见回复_中文.md`: faithful Chinese explanation of the response;
- `manuscript_insert_en.tex`: LaTeX figure, table, and manuscript paragraph;
- `experiment_analysis.md`: complete protocol, numerical results,
  interpretation, and limitations;
- `reproducibility.md`: source/data map and reproduction commands;
- `figures/`: PDF, SVG, and 600-dpi PNG;
- `analysis_tables/`: aggregate, seed-level, paired-statistical, and derived
  reviewer-facing CSVs;
- `raw_results/`: all transferred training/evaluation artifacts, final
  checkpoints, run manifests, source snapshot, logs, and validation records.

## Experimental scope

- Four controlled variants: full ART-MAPPO, without trust-aware training,
  without GRU, and without the attention-residual backbone.
- Two three-dimensional interception cases.
- Five paired training seeds (701–705) for each variant/case.
- Training budgets: 0.6 million environment steps in Case 1 and 1.8 million
  in Case 2.
- Twenty held-out deterministic episodes for every seed/case/variant,
  yielding 800 evaluation episodes in total.
- Trust-guided exploration is disabled during evaluation, so the trust-aware
  comparison measures its training-time effect rather than injecting a
  privileged controller at test time.

## Integrity

`raw_results/local_validation_report.json` records 40 training metric files,
8,000 training rows, 40 evaluation episode files, and 800 evaluation
episodes, with `passed=true` and no errors. `raw_results/SHA256SUMS.txt`
provides file hashes from the first complete transfer. A delivery-level hash
manifest is stored as `DELIVERY_SHA256SUMS.txt`.

## Important interpretation

The results are deliberately reported without selective omission:

- the attention-residual backbone has the strongest independent evidence;
- the trust-aware mechanism improves difficult-case target interception and
  synchronization relative to its ablation, but the pooled corrected tests
  are not significant at the current five-seed budget;
- the GRU does not show an independent held-out advantage under this budget,
  although the full model has a positive pooled training-return difference.

Accordingly, the supplied manuscript text supports the backbone strongly,
describes the trust effect as directional, and does not claim that the GRU is
indispensable. This is the statistically defensible closure of the review
request.
