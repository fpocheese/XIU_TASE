# Failure-case analysis data

All tables and figures in this directory are derived directly from six native
evaluation NPZ files and their hit-event CSV files.  No trajectory value was
edited or synthesized.

- `failure_case_metrics.csv`: one row per selected episode.
- `defender_metrics.csv`: one row per interceptor and episode.
- `target_group_metrics.csv`: one row per target-assignment group and episode.
- `attacker_trajectories_long.csv`: long-form attacker positions.
- `defender_trajectories_long.csv`: long-form interceptor positions, controls,
  time-to-go estimates, and distance to the assigned target.
- `*_v10.pdf`, `*_v10.svg`, `*_v10.png`: vector and 600-dpi plots.

The native NPZ files remain the authoritative raw data and are copied together
with this analysis package.
