#!/usr/bin/env bash
set -euo pipefail

project_root=/home/a2rl/reviewer_art_mappo_paper_aligned_20260728
output_root=/home/a2rl/reviewer_art_mappo_paper_aligned_20260728_trust_tuning
python_bin=/home/a2rl/miniconda3/envs/rlgpu/bin/python
mkdir -p "${output_root}/logs"
cd "${project_root}"

nohup "${python_bin}" onpolicy/scripts/train_art_mappo_ablation_3d.py \
    --variant full \
    --case_3d case2 \
    --seed 8803 \
    --save_dir "${output_root}/training" \
    --compare_steps 120000 \
    --episode_length 1500 \
    --n_rollout_threads 4 \
    --trust_initial 0.20 \
    --trust_omega_pn 0.80 \
    --trust_omega_probe 0.15 \
    --trust_omega_random 0.05 \
    >"${output_root}/logs/full_case2_seed8803.log" 2>&1 &
echo "$!" >"${output_root}/logs/full_case2_seed8803.pid"
printf 'tuned_full_pid=%s\n' "$!"
