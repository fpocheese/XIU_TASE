#!/usr/bin/env bash
set -u

ROOT=/home/a2rl/reviewer_art_mappo_ablation_20260724
TRAIN=${ROOT}/formal_paper_ablation_5seed
EVAL=${ROOT}/formal_paper_ablation_5seed_eval
ANALYSIS=${ROOT}/formal_paper_ablation_5seed_analysis
PY=/home/a2rl/miniconda3/bin/python

date --iso-8601=seconds
if test -f "${TRAIN}/suite_status.json"; then
  "${PY}" -c 'import collections,json,sys; d=json.load(open(sys.argv[1])); c=collections.Counter(v.get("status","unknown") for v in d.get("jobs",{}).values()); print("train_status",d.get("status","running"),"jobs",dict(c))' "${TRAIN}/suite_status.json"
fi
if test -f "${EVAL}/evaluation_status.json"; then
  "${PY}" -c 'import collections,json,sys; d=json.load(open(sys.argv[1])); c=collections.Counter(v.get("status","unknown") for v in d.get("jobs",{}).values()); print("eval_status",d.get("status","running"),"jobs",dict(c))' "${EVAL}/evaluation_status.json"
fi
printf 'metric_files %s\n' "$(find "${TRAIN}" -name training_metrics.csv 2>/dev/null | wc -l)"
"${PY}" - "${TRAIN}" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
status_path = root / "suite_status.json"
if status_path.exists():
    state = json.loads(status_path.read_text())
    for key, value in sorted(state.get("jobs", {}).items()):
        if value.get("status") != "running":
            continue
        path = root / key / "training_metrics.csv"
        if not path.exists():
            print("running", key, "initializing")
            continue
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            print("running", key, "initializing")
            continue
        row = rows[-1]
        print(
            "running",
            key,
            "update=" + row["update"],
            "steps=" + row["environment_steps"],
            "all_hit=" + row["all_hit_rate"],
            "all_sync=" + row["all_sync_rate"],
            "trust=" + row["trust_mean"],
            "guided=" + row["guided_action_fraction"],
        )
PY
nvidia-smi \
  --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
  --format=csv,noheader,nounits
if test -f "${ANALYSIS}/analysis_manifest.json"; then
  echo analysis_complete
fi
