#!/usr/bin/env bash
set -euo pipefail

project_root=/home/a2rl/reviewer_art_mappo_paper_aligned_20260728
output_root=/home/a2rl/reviewer_art_mappo_paper_aligned_20260728_pilot
python_bin=/home/a2rl/miniconda3/envs/rlgpu/bin/python

mkdir -p "${output_root}/logs"
cd "${project_root}"

for variant in no_gru no_attention_residual; do
    log_path="${output_root}/logs/${variant}_case2_seed8802.log"
    CUDA_VISIBLE_DEVICES=0 nohup "${python_bin}" \
        onpolicy/scripts/train_art_mappo_ablation_3d.py \
        --variant "${variant}" \
        --case_3d case2 \
        --seed 8802 \
        --save_dir "${output_root}/training" \
        --compare_steps 120000 \
        --episode_length 1500 \
        --n_rollout_threads 4 \
        --cuda \
        >"${log_path}" 2>&1 &
    echo "$!" >"${output_root}/logs/${variant}_case2_seed8802.pid"
    printf '%s_pid=%s\n' "${variant}" "$!"
done
