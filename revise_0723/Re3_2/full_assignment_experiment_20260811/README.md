# Reviewer 3.2 — Full IDBO assignment experiment

This package contains a target-assignment-only study of IDBO complexity,
scalability, communication delay, graph diameter, and time-varying cooperative
deflection. The manuscript and both original code bases were treated as read-only.

## Directory structure

- `V2/`: complete experiment using `revise_0723/idbo_paper`.
- `V1/`: the same protocol using the optimizer in `DT_PAPER/idbo_code`.
- `comparison/`: paired V1/V2 data, figure, table, and Chinese interpretation.
- `code/`: all reproducible wrappers, communication models, experiment runners,
  plotting scripts, and the V1 adapter.
- `validation_report.md`: hashes and validation results.
- `data_dictionary.md`: definitions of every output group and metric.

The nominal assignment scale is **20 interceptors against 8 targets**. Larger
and smaller problems preserve the nominal ratio $M/N=2.5$ and are used only to
measure scaling.

## Reproduction

```bash
python3 code/run_full_assignment_experiment.py \
  --idbo-dir /home/uav/00gao_xueshu/DT_PAPER/XIU_tase_paper_V1/revise_0723/idbo_paper \
  --out-dir V2

python3 code/run_v1_full_assignment_experiment.py \
  --v1-dir /home/uav/00gao_xueshu/DT_PAPER/idbo_code/python_project \
  --v2-scenario-dir /home/uav/00gao_xueshu/DT_PAPER/XIU_tase_paper_V1/revise_0723/idbo_paper \
  --out-dir V1

python3 code/run_version_comparison.py \
  --v1-dir /home/uav/00gao_xueshu/DT_PAPER/idbo_code/python_project \
  --v2-dir /home/uav/00gao_xueshu/DT_PAPER/XIU_tase_paper_V1/revise_0723/idbo_paper \
  --experiment-root . --out-dir comparison/data

MPLCONFIGDIR=/tmp/mpl_reviewer32 python3 code/plot_full_assignment_experiment.py --out-dir V2
MPLCONFIGDIR=/tmp/mpl_reviewer32 python3 code/plot_full_assignment_experiment.py --out-dir V1
MPLCONFIGDIR=/tmp/mpl_reviewer32 python3 code/plot_version_comparison.py --root .
```

The V1 adapter intentionally discards the V1 legacy fixed scenario and applies
only its optimizer to the current V2 three-dimensional snapshots and objective.
This is necessary for a scientifically meaningful paired comparison.
