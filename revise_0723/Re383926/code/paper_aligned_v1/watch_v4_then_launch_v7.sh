#!/usr/bin/env bash
set -euo pipefail

v4_root=/home/a2rl/reviewer_art_mappo_paper_aligned_20260728_curriculum_trust_v4
project_root=/home/a2rl/reviewer_art_mappo_paper_aligned_20260728
watch_log="${v4_root}/watch_v4_then_launch_v7.log"

case1_pid="$(tr -d '[:space:]' <"${v4_root}/logs/full_case1_seed9021.pid")"
case2_pid="$(tr -d '[:space:]' <"${v4_root}/logs/full_case2_seed9021.pid")"
printf '[%s] watching v4 pids %s %s\n' \
    "$(date --iso-8601=seconds)" "${case1_pid}" "${case2_pid}" >>"${watch_log}"
case1_log="${v4_root}/logs/full_case1_seed9021.log"
case2_log="${v4_root}/logs/full_case2_seed9021.log"
cd "${project_root}"

launch_case() {
    case_name="$1"
    output_root=/home/a2rl/reviewer_art_mappo_paper_aligned_20260728_onpolicy_mixture_v7
    python_bin=/home/a2rl/miniconda3/envs/rlgpu/bin/python
    mkdir -p "${output_root}/logs"
    if pgrep -af "train_art_mappo_ablation_3d.py.*case_3d ${case_name}.*seed 9101" >/dev/null; then
        printf '[%s] v7 %s already running; duplicate suppressed\n' \
            "$(date --iso-8601=seconds)" "${case_name}" >>"${watch_log}"
        return
    fi
    log_file="${output_root}/logs/full_${case_name}_seed9101.log"
    pid_file="${output_root}/logs/full_${case_name}_seed9101.pid"
    nohup /home/a2rl/miniconda3/envs/rlgpu/bin/python \
        onpolicy/scripts/train_art_mappo_ablation_3d.py \
        --variant full --case_3d "${case_name}" --seed 9101 \
        --save_dir "${output_root}/training" \
        --compare_steps 614400 --episode_length 1024 \
        --n_rollout_threads 4 --physical_episode_horizon_steps 1500 \
        --trust_initial 0.20 --trust_alpha 0.01 \
        --trust_omega_pn 0.04 --trust_omega_probe 0.95 \
        --trust_omega_random 0.01 --ppo_epoch 5 \
        --save_interval 5 --checkpoint_interval 5 \
        >"${log_file}" 2>&1 &
    printf '%s\n' "$!" >"${pid_file}"
    printf '[%s] v4 %s DONE verified; launched v7 %s pid=%s\n' \
        "$(date --iso-8601=seconds)" "${case_name}" \
        "${case_name}" "$!" >>"${watch_log}"
}

while kill -0 "${case1_pid}" 2>/dev/null; do
    sleep 20
done
if ! grep -q '^\[DONE\]' "${case1_log}"; then
    printf '[%s] v4 case1 exited without DONE; v7 case1 not launched\n' \
        "$(date --iso-8601=seconds)" >>"${watch_log}"
    exit 1
fi
launch_case case1

while kill -0 "${case2_pid}" 2>/dev/null; do
    sleep 20
done
if ! grep -q '^\[DONE\]' "${case2_log}"; then
    printf '[%s] v4 case2 exited without DONE; v7 case2 not launched\n' \
        "$(date --iso-8601=seconds)" >>"${watch_log}"
    exit 1
fi
launch_case case2
