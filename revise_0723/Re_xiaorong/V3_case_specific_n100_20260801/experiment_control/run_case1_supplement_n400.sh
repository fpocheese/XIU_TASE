#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/a2rl/reviewer_xiu_ablation_domainrand_v3_20260801
PYTHON=/home/a2rl/miniconda3/envs/rlgpu/bin/python
EVALUATOR="$ROOT/code/onpolicy/scripts/run_paper_nominal_eval_parallel.py"
OUTROOT="$ROOT/formal_evaluation_case1_supplement_n400"
SELECTION="$ROOT/formal_evaluation_n100/selection"
mkdir -p "$OUTROOT/logs"

run_one() {
    local variant="$1"
    local outdir="$OUTROOT/evaluation/$variant/case1/seed8303"
    "$PYTHON" "$EVALUATOR" \
        --episodes 400 \
        --workers 5 \
        --seed 102001 \
        --outdir "$outdir" \
        --condition "case1_supplement_n400_${variant}" \
        --case case1 \
        --variant "$variant" \
        --model_dir "$SELECTION/$variant/case1/seed8303/selected_model" \
        --max_steps 1500 \
        --assignment_mode fixed \
        --initial_perturbation_scale 1.0 \
        --cpu_eval >"$OUTROOT/logs/${variant}.log" 2>&1
}

# Together with the disjoint original n=100 test block (98401--98500) and the
# completed new n=500 block (100001--100500), this block (102001--102400)
# yields exactly 1000 Case-1 trials per algorithm.
for variant in full no_trust no_gru no_attention_residual; do
    run_one "$variant" &
done
wait

touch "$OUTROOT/COMPLETE"
printf 'Case-1 n=400 supplement completed for all four frozen policies.\n'
