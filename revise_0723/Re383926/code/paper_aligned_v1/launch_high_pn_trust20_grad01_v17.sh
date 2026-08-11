#!/usr/bin/env bash
set -euo pipefail

project_root=/home/a2rl/reviewer_art_mappo_paper_aligned_20260728
output_root=/home/a2rl/reviewer_art_mappo_paper_aligned_20260728_high_pn_trust20_grad01_v17
python_bin=/home/a2rl/miniconda3/envs/rlgpu/bin/python

mkdir -p "${output_root}/logs"
cd "${project_root}"

for case_name in case1 case2; do
    log_file="${output_root}/logs/full_${case_name}_seed9541.log"
    pid_file="${output_root}/logs/full_${case_name}_seed9541.pid"
    nohup "${python_bin}" onpolicy/scripts/train_art_mappo_ablation_3d.py \
        --variant full \
        --case_3d "${case_name}" \
        --seed 9541 \
        --save_dir "${output_root}/training" \
        --compare_steps 614400 \
        --episode_length 1024 \
        --n_rollout_threads 4 \
        --physical_episode_horizon_steps 1500 \
        --trust_initial 0.20 \
        --trust_rho 0.05 \
        --trust_alpha 0.005 \
        --trust_tau 1.0 \
        --trust_omega_pn 0.92 \
        --trust_omega_probe 0.07 \
        --trust_omega_random 0.01 \
        --trust_pn_navigation_constant 10.0 \
        --trust_action_hold_steps 5 \
        --include_guided_actor_samples \
        --ppo_epoch 1 \
        --max_grad_norm 0.10 \
        --save_interval 5 \
        --checkpoint_interval 5 \
        >"${log_file}" 2>&1 &
    printf '%s\n' "$!" >"${pid_file}"
    printf '%s_pid=%s\n' "${case_name}" "$!"
done
