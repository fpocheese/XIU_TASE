#!/usr/bin/env bash
set -euo pipefail

project_root=/home/a2rl/reviewer_art_mappo_paper_aligned_20260728
output_root=/home/a2rl/reviewer_art_mappo_paper_aligned_20260728_pilot
python_bin=/home/a2rl/miniconda3/envs/rlgpu/bin/python

mkdir -p "${output_root}/logs"
cd "${project_root}"

launch_variant() {
    local variant="$1"
    local seed="$2"
    local log_path="${output_root}/logs/${variant}_case2_seed${seed}.log"
    CUDA_VISIBLE_DEVICES=0 nohup "${python_bin}" \
        onpolicy/scripts/train_art_mappo_ablation_3d.py \
        --variant "${variant}" \
        --case_3d case2 \
        --seed "${seed}" \
        --save_dir "${output_root}/training" \
        --compare_steps 120000 \
        --episode_length 1500 \
        --n_rollout_threads 4 \
        --cuda \
        >"${log_path}" 2>&1 &
    echo "$!" >"${output_root}/logs/${variant}_case2_seed${seed}.pid"
}

launch_variant full 8802
launch_variant no_trust 8802

printf 'pilot_root=%s\n' "${output_root}"
printf 'full_pid=%s\n' "$(cat "${output_root}/logs/full_case2_seed8802.pid")"
printf 'no_trust_pid=%s\n' "$(cat "${output_root}/logs/no_trust_case2_seed8802.pid")"
