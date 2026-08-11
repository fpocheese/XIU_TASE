#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/a2rl/reviewer_xiu_ablation_domainrand_v3_20260801
PYTHON=/home/a2rl/miniconda3/envs/rlgpu/bin/python
EVALUATOR="$ROOT/code/onpolicy/scripts/run_paper_nominal_eval_parallel.py"
OUTROOT="$ROOT/formal_evaluation_pooled_n1000"
SELECTION="$ROOT/formal_evaluation_n100/selection"
LOGROOT="$OUTROOT/logs"

mkdir -p "$LOGROOT"

run_one() {
    local variant="$1"
    local case_name="$2"
    local seed="$3"
    local outdir="$OUTROOT/evaluation/$variant/$case_name/seed8303"
    local model_dir="$SELECTION/$variant/$case_name/seed8303/selected_model"
    local logfile="$LOGROOT/${variant}_${case_name}.log"

    if [[ -s "$outdir/episodes.csv" ]]; then
        local count
        count=$(($(wc -l < "$outdir/episodes.csv") - 1))
        if [[ "$count" -eq 500 ]]; then
            printf 'SKIP completed %s %s (%s episodes)\n' "$variant" "$case_name" "$count"
            return 0
        fi
    fi

    mkdir -p "$outdir"
    "$PYTHON" "$EVALUATOR" \
        --episodes 500 \
        --workers 5 \
        --seed "$seed" \
        --outdir "$outdir" \
        --condition "pooled_ablation_n1000_${variant}_${case_name}" \
        --case "$case_name" \
        --variant "$variant" \
        --model_dir "$model_dir" \
        --max_steps 1500 \
        --assignment_mode fixed \
        --initial_perturbation_scale 1.0 \
        --cpu_eval >"$logfile" 2>&1
    printf 'DONE %s %s\n' "$variant" "$case_name"
}

# Four five-worker evaluators are run at a time on the 20-core host.  The same
# disjoint seed block is reused across variants within each scenario to retain a
# paired design; no training, backpropagation, or optimizer update is performed.
for variant in full no_trust no_gru no_attention_residual; do
    run_one "$variant" case1 100001 &
done
wait

for variant in full no_trust no_gru no_attention_residual; do
    run_one "$variant" case2 101001 &
done
wait

touch "$OUTROOT/COMPLETE"
printf 'All eight 500-episode frozen-policy evaluations completed.\n'
