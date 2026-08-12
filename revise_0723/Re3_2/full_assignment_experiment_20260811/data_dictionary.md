# Data dictionary

Each of `V1/data/` and `V2/data/` contains the following files.

- `runtime_raw.csv`: one measured optimizer runtime per version, scale, and seed.
- `runtime_summary.csv`: cross-seed mean, standard deviation, and 95% CI.
- `static_delay_raw.csv`: one static consensus run for each 20×8 scene and delay.
- `static_delay_summary.csv`: consensus rounds, latency, messages, and record entries versus delay.
- `topology_raw.csv`, `topology_summary.csv`: the same quantities versus graph diameter and degree.
- `scaling_raw.csv`, `scaling_summary.csv`: 10×4 through 160×64 scaling at bounded degree and 100-ms additional delay.
- `dynamic_trace_raw.csv`: every 50-ms exchange of every moving-deflection sequence.
- `dynamic_run_summary.csv`: one row per sequence and delay.
- `dynamic_summary.csv`: cross-sequence mean, standard deviation, and 95% CI.
- `dynamic_change_rate.csv`: rate at which the instantaneous winner set changes between 2-s updates.
- `dynamic_oracle_quality.csv`: IDBO assignment cost and maximum proposed target load for each dynamic snapshot.

Important dynamic metrics:

- `winner_jaccard`: Jaccard agreement between a node's delayed Top-$L_{\max}$ winner set and the current instantaneous winner set.
- `exact_node_fraction`: fraction of nodes whose full winner set exactly equals the current set.
- `edge_disagreement`: fraction of communication edges whose endpoint replicas differ.
- `stale_record_fraction`: fraction of local origin records older than the current allocation epoch.
- `recovery_rate`: fraction of changed allocation epochs that recover at least 95% exact node agreement before the next 2-s update.

`comparison/data/` contains:

- `paired_quality_raw.csv`: V1/V2 results on the same 30 static 20×8 scenes.
- `paired_quality_summary.csv`: static cost, interception probability, adversarial advantage, pair score, target load, coverage, and capacity feasibility.
- `version_comparison_summary.csv`: compact cross-version runtime and dynamic metrics.

The `nan` entries in individual `dynamic_run_summary.csv` recovery-time fields mean that no changed epoch recovered for that seed/delay, so a recovery time is undefined. They are not failed numerical computations; the associated `recovery_rate` is zero.
