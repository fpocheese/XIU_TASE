#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/../.." && pwd)
experiment_root=/home/a2rl/reviewer_xiu_ablation_20260729
training_root="$experiment_root/formal_81920"
evaluation_root="$experiment_root/formal_evaluation_n100"
status_file="$experiment_root/watch_status.txt"

while true; do
    done_count=$(
        { grep -Rl '\[DONE\]' "$training_root/logs" \
            --include='*.log' 2>/dev/null || true; } | wc -l
    )
    error_count=$(
        { grep -RliE 'Traceback|RuntimeError|ValueError' \
            "$training_root/logs" --include='*.log' 2>/dev/null || true; } \
            | wc -l
    )
    printf '%s training_done=%s/24 errors=%s\n' \
        "$(date --iso-8601=seconds)" "$done_count" "$error_count" \
        >"$status_file"
    if (( error_count > 0 )); then
        exit 2
    fi
    if (( done_count == 24 )); then
        break
    fi
    sleep 60
done

printf '%s evaluation_started\n' "$(date --iso-8601=seconds)" \
    >"$status_file"
PYTHONPATH="$project_root" \
/home/a2rl/miniconda3/envs/rlgpu/bin/python -u \
    "$project_root/onpolicy/scripts/run_formal_ablation_evaluation.py" \
    --training_root "$training_root/training" \
    --outroot "$evaluation_root" \
    --seeds 8301,8302,8303 \
    --validation_episodes 10 \
    --validation_workers 5 \
    --checkpoint_stride 1 \
    --test_episodes 100 \
    --test_workers 5 \
    --max_parallel 2 \
    >"$evaluation_root.log" 2>&1

printf '%s plotting_started\n' "$(date --iso-8601=seconds)" \
    >"$status_file"
PYTHONPATH="$project_root" \
/home/a2rl/miniconda3/envs/rlgpu/bin/python -u \
    "$project_root/plot_reviewer_experiments_v10.py" \
    --training_root "$training_root/training" \
    --evaluation_root "$evaluation_root/combined" \
    --outdir "$experiment_root/figures_v10" \
    >"$experiment_root/figures_v10.log" 2>&1

printf '%s audit_started\n' "$(date --iso-8601=seconds)" \
    >"$status_file"
PYTHONPATH="$project_root" \
/home/a2rl/miniconda3/envs/rlgpu/bin/python -u \
    "$project_root/summarize_xiu_ablation.py" \
    --training_root "$training_root/training" \
    --episodes_csv "$evaluation_root/combined/episodes.csv" \
    --outdir "$experiment_root/analysis" \
    >"$experiment_root/analysis.log" 2>&1

printf '%s complete\n' "$(date --iso-8601=seconds)" >"$status_file"
