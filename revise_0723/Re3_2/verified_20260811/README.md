# Reviewer 3.2 verified deliverables

This directory contains a newly verified response package for the reviewer
comment on IDBO complexity, scalability, and communication delay.

## Files

- `analysis_report_zh.md`: Chinese audit, experiment interpretation, limitations,
  and recommended response strategy.
- `reviewer_response_3_2.tex`: publication-ready English response snippet.
- `manuscript_revision_suggestion.tex`: minimal suggested manuscript text. It has
  **not** been written into `new_highlight/main.tex`.
- `code/benchmark_idbo_complexity.py`: reproducible benchmark.
- `data/local_idbo_runtime.csv`: actual `idbo_paper.py` runtime versus local
  population size.
- `data/consensus_delay.csv`: delayed Top-$L_{\max}$ propagation results.
- `data/consensus_topology.csv`: graph-diameter results.
- `data/consensus_scaling.csv`: bounded-degree swarm-size scaling.
- `benchmark_summary.json`: machine-readable validation summary.
- `validation_report.md`: input hashes, environment, automated checks, and the
  precise scope of the benchmark conclusions.
- `figures/idbo_complexity_delay_verified.pdf` and `.png`: compact supplementary
  figure generated directly from the CSV data.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/mpl_re3_2 python3 code/benchmark_idbo_complexity.py \
  --idbo-dir /home/uav/00gao_xueshu/DT_PAPER/XIU_tase_paper_V1/revise_0723/idbo_paper \
  --out-dir .
```

The local optimizer benchmark imports the user's current IDBO implementation.
The consensus benchmark is intentionally isolated: it reproduces delayed,
neighbor-to-neighbor Top-$L_{\max}$ record propagation for a static assignment
snapshot and verifies the final result against the global Top-$L_{\max}$ fixed
point. It is not presented as a replacement for a complete engagement simulator.
