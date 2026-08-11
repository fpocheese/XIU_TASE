#!/usr/bin/env bash
set -euo pipefail

PY=/home/a2rl/miniconda3/envs/rlgpu/bin/python
ROOT=/home/a2rl/reviewer_xiu_ablation_20260729
RUNNER="$ROOT/code/onpolicy/scripts/run_reviewer_eval_parallel.py"
OUT="$ROOT/gru_delay_screen_n12"
mkdir -p "$OUT"

pids=()
for case_name in case1 case2; do
  for variant_name in full no_gru; do
    model_dir="$ROOT/formal_evaluation_n100/selection/$variant_name/$case_name/seed8301/selected_model"
    for delay_steps in 1 3 5 10; do
      run_dir="$OUT/${case_name}_${variant_name}_delay${delay_steps}"
      mkdir -p "$run_dir"
      "$PY" "$RUNNER" \
        --episodes 12 \
        --workers 3 \
        --seed 99001 \
        --outdir "$run_dir" \
        --condition "gru_delay_screen_${variant_name}_${case_name}_d${delay_steps}" \
        --case "$case_name" \
        --variant "$variant_name" \
        --model_dir "$model_dir" \
        --max_steps 1500 \
        --assignment_mode fixed \
        --initial_perturbation_scale 1.0 \
        --sensor_delay_steps "$delay_steps" \
        --cpu_eval \
        > "$run_dir/runner.log" 2>&1 &
      pids+=("$!")
    done
  done
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "SCREEN_FAILED"
  exit 1
fi

"$PY" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/summary.json")):
    summary = json.loads(path.read_text())
    condition = path.parent.name.split("_")
    rows.append(
        {
            "case": condition[0],
            "variant": condition[1] if condition[1] == "full" else "no_gru",
            "delay_steps": int(condition[-1].replace("delay", "")),
            "target_coverage": summary["target_coverage_success_rate"],
            "all_defenders_hit": summary["all_defenders_hit_rate"],
            "cooperative_success": summary["cooperative_success_rate"],
        }
    )
rows.sort(key=lambda r: (r["case"], r["variant"], r["delay_steps"]))
print(json.dumps(rows, indent=2))
PY
