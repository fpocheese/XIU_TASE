#!/usr/bin/env bash
set -euo pipefail

experiment_root=${1:-/home/a2rl/reviewer_xiu_ablation_domainrand_v3_20260801}
seed=${2:-8303}
max_parallel=${3:-4}
project_root="${experiment_root}/code"
training_root="${experiment_root}/training"
log_root="${experiment_root}/logs"
mkdir -p "${training_root}" "${log_root}"

case_args() {
    case "$1" in
        case1)
            printf '%s\n' \
                '--defender_residual_scale 0.20' \
                '--defender_sync_speed_gain 1.40' \
                '--defender_sync_tgo_ref mean' \
                '--defender_speed_target 40' \
                '--defender_speed_gain 0.016' \
                '--train_initial_perturbation_seed 881003'
            ;;
        case2)
            printf '%s\n' \
                '--defender_residual_scale 0.05' \
                '--defender_sync_speed_gain 2.00' \
                '--defender_sync_tgo_ref mean' \
                '--defender_speed_target 40' \
                '--defender_speed_gain 0.008' \
                '--train_initial_perturbation_seed 883003'
            ;;
        *) return 2 ;;
    esac
}

run_one() {
    local variant=$1
    local engagement_case=$2
    local run_dir="${training_root}/${variant}/${engagement_case}/seed${seed}"
    local stem="${variant}_${engagement_case}_seed${seed}"
    local log_path="${log_root}/${stem}.log"
    local done_path="${log_root}/${stem}.done"
    local fail_path="${log_root}/${stem}.failed"
    if [[ -s "${done_path}" ]]; then return 0; fi
    if [[ -e "${run_dir}/models/checkpoint_latest.pt" ]]; then
        printf 'refusing_unregistered_resume=%s\n' "${run_dir}" > "${fail_path}"
        return 3
    fi
    rm -f "${fail_path}"
    mapfile -t profile < <(case_args "${engagement_case}")
    local profile_args=()
    local item
    for item in "${profile[@]}"; do
        read -r -a words <<< "${item}"
        profile_args+=("${words[@]}")
    done
    {
        printf 'start=%s variant=%s case=%s seed=%s target_steps=600000 protocol=domainrand_v3\n' \
            "$(date --iso-8601=seconds)" "${variant}" "${engagement_case}" "${seed}"
        PYTHONPATH="${project_root}" \
        /home/a2rl/miniconda3/envs/rlgpu/bin/python -u \
            "${project_root}/onpolicy/scripts/train_xiu_art_ablation_paper_nominal.py" \
            --variant "${variant}" --case_3d "${engagement_case}" --seed "${seed}" \
            --save_dir "${training_root}" --compare_steps 600000 \
            --episode_length 1024 --physical_episode_horizon_steps 1500 \
            --n_rollout_threads 1 \
            --paper_preset_path /home/a2rl/xiu_onpolicy_3d_fix/stable_V2/presets/paper_case_presets_original_assignment_verified.npz \
            --paper_attacker_replay 1 --paper_altitude 120 \
            --paper_altitude_step 0 --paper_defender_climb_to_target 0 \
            --train_initial_perturbation_scale 1.0 \
            "${profile_args[@]}" \
            --defender_sensor_delay_steps 0 \
            --defender_obs_pos_noise_std 0.0 \
            --defender_obs_vel_noise_std 0.0 \
            --defender_obs_filter_alpha 1.0 \
            --attacker_load_limit 1.75 --attacker_yaw_scale 1.55 \
            --attacker_pitch_scale 1.55 \
            --trust_initial 0.80 --trust_rho 0.05 --trust_alpha 0.10 \
            --trust_tau 1.0 --trust_omega_pn 0.70 \
            --trust_omega_probe 0.20 --trust_omega_random 0.10 \
            --trust_action_hold_steps 5 --ppo_epoch 5 \
            --num_mini_batch 4 --data_chunk_length 32 \
            --save_interval 10 --checkpoint_interval 2
        printf 'complete=%s\n' "$(date --iso-8601=seconds)" > "${done_path}"
    } > "${log_path}" 2>&1 || {
        status=$?
        printf 'failed=%s exit_code=%s\n' \
            "$(date --iso-8601=seconds)" "${status}" > "${fail_path}"
        return "${status}"
    }
}

active=0
failed=0
for engagement_case in case1 case2; do
    for variant in full no_trust no_gru no_attention_residual; do
        run_one "${variant}" "${engagement_case}" &
        active=$((active + 1))
        if (( active >= max_parallel )); then
            if ! wait -n; then failed=$((failed + 1)); fi
            active=$((active - 1))
        fi
    done
done
while (( active > 0 )); do
    if ! wait -n; then failed=$((failed + 1)); fi
    active=$((active - 1))
done

done_count=$(find "${log_root}" -name "*_seed${seed}.done" | wc -l)
printf 'finished=%s failures=%s done=%s expected=8\n' \
    "$(date --iso-8601=seconds)" "${failed}" "${done_count}" \
    > "${experiment_root}/training_suite_complete.txt"
if (( failed > 0 || done_count != 8 )); then exit 1; fi
