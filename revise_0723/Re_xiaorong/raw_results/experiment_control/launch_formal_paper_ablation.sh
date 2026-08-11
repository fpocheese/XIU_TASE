#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/a2rl/reviewer_art_mappo_ablation_20260724
CODE=${ROOT}/on-policy-main
TRAIN_OUT=${ROOT}/formal_paper_ablation_5seed
EVAL_OUT=${ROOT}/formal_paper_ablation_5seed_eval
ANALYSIS_OUT=${ROOT}/formal_paper_ablation_5seed_analysis
CONDA=/home/a2rl/miniconda3/bin/conda
SEEDS=(701 702 703 704 705)

mkdir -p "${TRAIN_OUT}" "${EVAL_OUT}" "${ANALYSIS_OUT}"
cd "${CODE}"

env CUDA_VISIBLE_DEVICES=0 "${CONDA}" run -n rlgpu \
  python onpolicy/scripts/run_art_mappo_ablation_suite.py \
  --output_root "${TRAIN_OUT}" \
  --steps 600000 \
  --case1_steps 600000 \
  --case2_steps 1800000 \
  --episode_length 1500 \
  --rollout_threads 4 \
  --seeds "${SEEDS[@]}" \
  --cases case1 case2 \
  --max_parallel 3 \
  --max_retries 2 \
  --stall_minutes 20 \
  > "${TRAIN_OUT}/suite.log" 2>&1

env CUDA_VISIBLE_DEVICES=0 "${CONDA}" run -n rlgpu \
  python onpolicy/scripts/run_art_mappo_ablation_eval_suite.py \
  --training_root "${TRAIN_OUT}" \
  --output_root "${EVAL_OUT}" \
  --seeds "${SEEDS[@]}" \
  --cases case1 case2 \
  --episodes_per_seed_case 20 \
  --max_steps 1500 \
  --max_parallel 3 \
  > "${EVAL_OUT}/suite.log" 2>&1

env CUDA_VISIBLE_DEVICES=0 "${CONDA}" run -n rlgpu \
  python onpolicy/scripts/analyze_art_mappo_ablation.py \
  --training_root "${TRAIN_OUT}" \
  --evaluation_root "${EVAL_OUT}" \
  --output_dir "${ANALYSIS_OUT}" \
  > "${ANALYSIS_OUT}/analysis.log" 2>&1
