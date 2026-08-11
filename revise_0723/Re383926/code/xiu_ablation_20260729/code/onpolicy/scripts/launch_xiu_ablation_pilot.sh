#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/../.." && pwd)
output_root=${1:-/home/a2rl/reviewer_xiu_ablation_20260729/pilot}
seed=${2:-8291}
steps=${3:-81920}
max_parallel=${4:-2}

mkdir -p "$output_root/logs"
active_jobs=0

for variant in full no_trust no_gru no_attention_residual; do
    for case_name in case1 case2; do
        if [[ "$case_name" == "case1" ]]; then
            sync_gain=0.14
            speed_gain=0.016
        else
            sync_gain=1.40
            speed_gain=0.008
        fi
        log_path="$output_root/logs/${variant}_${case_name}_seed${seed}.log"
        PYTHONPATH="$project_root" \
        /home/a2rl/miniconda3/envs/rlgpu/bin/python -u \
            "$project_root/onpolicy/scripts/train_xiu_art_ablation.py" \
            --variant "$variant" \
            --case_3d "$case_name" \
            --seed "$seed" \
            --save_dir "$output_root/training" \
            --compare_steps "$steps" \
            --episode_length 1024 \
            --physical_episode_horizon_steps 1500 \
            --n_rollout_threads 1 \
            --paper_preset_path \
              /home/a2rl/xiu_onpolicy_3d_fix/stable_V2/presets/paper_case_presets_original_assignment_verified.npz \
            --paper_attacker_replay 1 \
            --paper_altitude 120 \
            --paper_altitude_step 0 \
            --paper_defender_climb_to_target 0 \
            --defender_residual_scale 0.20 \
            --defender_sync_speed_gain "$sync_gain" \
            --defender_sync_tgo_ref min \
            --defender_speed_target 40 \
            --defender_speed_gain "$speed_gain" \
            --attacker_load_limit 1.75 \
            --attacker_yaw_scale 1.55 \
            --attacker_pitch_scale 1.55 \
            --trust_initial 0.80 \
            --trust_rho 0.05 \
            --trust_alpha 0.10 \
            --trust_tau 1.0 \
            --trust_omega_pn 0.70 \
            --trust_omega_probe 0.20 \
            --trust_omega_random 0.10 \
            --trust_action_hold_steps 5 \
            --ppo_epoch 5 \
            --num_mini_batch 4 \
            --data_chunk_length 32 \
            --save_interval 10 \
            --checkpoint_interval 2 \
            >"$log_path" 2>&1 &
        echo "$!" >"${log_path%.log}.pid"
        active_jobs=$((active_jobs + 1))
        if (( active_jobs >= max_parallel )); then
            wait -n
            active_jobs=$((active_jobs - 1))
        fi
    done
done

wait
