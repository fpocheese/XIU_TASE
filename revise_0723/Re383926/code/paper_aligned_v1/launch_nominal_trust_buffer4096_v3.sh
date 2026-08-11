#!/usr/bin/env bash
set -euo pipefail

project_root=/home/a2rl/reviewer_art_mappo_paper_aligned_20260728
output_root=/home/a2rl/reviewer_art_mappo_paper_aligned_20260728_nominal_trust_buffer4096_v3
python_bin=/home/a2rl/miniconda3/envs/rlgpu/bin/python

mkdir -p "${output_root}/logs"
cd "${project_root}"

# Table II reports a replay buffer of 4096 transitions.  Four rollout
# environments and a 1024-step horizon reproduce that number exactly.
for case_name in case1 case2; do
    log_file="${output_root}/logs/full_${case_name}_seed9001.log"
    pid_file="${output_root}/logs/full_${case_name}_seed9001.pid"
    nohup "${python_bin}" onpolicy/scripts/train_art_mappo_ablation_3d.py \
        --variant full \
        --case_3d "${case_name}" \
        --seed 9001 \
        --save_dir "${output_root}/training" \
        --compare_steps 614400 \
        --episode_length 1024 \
        --n_rollout_threads 4 \
        --trust_initial 0.20 \
        --trust_alpha 0.10 \
        --trust_omega_pn 0.04 \
        --trust_omega_probe 0.95 \
        --trust_omega_random 0.01 \
        --ppo_epoch 5 \
        --save_interval 5 \
        --checkpoint_interval 5 \
        >"${log_file}" 2>&1 &
    printf '%s\n' "$!" >"${pid_file}"
    printf '%s_pid=%s\n' "${case_name}" "$!"
done
