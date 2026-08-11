# Reproducibility and data map

## Data lineage

The experiment was executed in an isolated remote copy:

`/home/a2rl/reviewer_art_mappo_ablation_20260724`

The immutable local mirror is placed under:

`raw_results/`

The canonical source supplied by the author was not overwritten. The
experiment source snapshot used for every run is stored in
`raw_results/source_snapshot/`.

## Directory map

- `raw_results/training/<variant>/<case>/seed<seed>/`
  - `training_metrics.csv`: raw update-level curve;
  - `run_manifest.json`: architecture, hyperparameters, case preset, and
    component switches;
  - `command_config.json`: resolved command configuration;
  - `models/actor.pt`, `critic.pt`, `checkpoint_latest.pt`: final weights and
    restartable state;
  - `logs/`: TensorBoard and summary output.
- `raw_results/evaluation/<variant>/<case>/seed<seed>/`
  - `eval_summary.csv`: 20 held-out episode records;
  - `<case>/<case>_episode_summary.csv`: raw episode outcome records;
  - `<case>/<case>_hit_events.csv`: hit events;
  - `<case>/<case>_selected_episode.npz`: selected trajectory variables;
  - trajectory/control/time-to-go figures and evaluation log.
- `raw_results/analysis/`
  - seed-level, aggregate, and paired-statistics CSVs;
  - final PDF/SVG/PNG figure;
  - analysis manifest and log.
- `raw_results/source_snapshot/`
  - exact ART actor/critic, policy, optimizer, runner, environment, training,
    evaluation, and analysis code.
- `raw_results/experiment_control/`
  - launch, monitoring, fetch, validation, and state records.

## Reproduction commands

From the transferred `source_snapshot` restored into the corresponding
on-policy project:

```bash
python onpolicy/scripts/run_art_mappo_ablation_suite.py \
  --output_root /path/to/training \
  --steps 600000 \
  --seeds 701 702 703 704 705 \
  --cases case1 case2
```

The formal launcher resolves Case 1 to 600,000 steps and Case 2 to 1,800,000
steps. Evaluation:

```bash
python onpolicy/scripts/run_art_mappo_ablation_eval_suite.py \
  --training_root /path/to/training \
  --output_root /path/to/evaluation \
  --seeds 701 702 703 704 705 \
  --episodes_per_seed_case 20
```

Analysis:

```bash
python onpolicy/scripts/analyze_art_mappo_ablation.py \
  --training_root /path/to/training \
  --evaluation_root /path/to/evaluation \
  --output_dir /path/to/analysis
```

Delivery tables can be regenerated without changing the raw data:

```bash
python scripts/generate_delivery_tables.py
```

## Validation criteria

The strict validator requires:

- 40 training runs;
- five seeds for every variant/case;
- 100 updates and 600,000 final steps for Case 1;
- 300 updates and 1,800,000 final steps for Case 2;
- strictly increasing steps and finite core metrics;
- 40 evaluation seed/case files;
- exactly 20 episodes per evaluation file;
- 800 total episodes;
- no NaN/Inf in required fields.

The transferred `local_validation_report.json` records `passed: true`.
