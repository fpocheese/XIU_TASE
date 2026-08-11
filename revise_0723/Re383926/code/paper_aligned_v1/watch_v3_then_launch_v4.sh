#!/usr/bin/env bash
set -euo pipefail

v3_root=/home/a2rl/reviewer_art_mappo_paper_aligned_20260728_nominal_trust_buffer4096_v3
project_root=/home/a2rl/reviewer_art_mappo_paper_aligned_20260728
watch_log="${v3_root}/watch_v3_then_launch_v4.log"

case1_pid="$(tr -d '[:space:]' <"${v3_root}/logs/full_case1_seed9001.pid")"
case2_pid="$(tr -d '[:space:]' <"${v3_root}/logs/full_case2_seed9001.pid")"
printf '[%s] watching v3 pids %s %s\n' \
    "$(date --iso-8601=seconds)" "${case1_pid}" "${case2_pid}" >>"${watch_log}"

while kill -0 "${case1_pid}" 2>/dev/null || kill -0 "${case2_pid}" 2>/dev/null; do
    sleep 20
done

case1_log="${v3_root}/logs/full_case1_seed9001.log"
case2_log="${v3_root}/logs/full_case2_seed9001.log"
if ! grep -q '^\[DONE\]' "${case1_log}" || ! grep -q '^\[DONE\]' "${case2_log}"; then
    printf '[%s] v3 exited without two DONE markers; v4 not launched\n' \
        "$(date --iso-8601=seconds)" >>"${watch_log}"
    exit 1
fi

if pgrep -af 'train_art_mappo_ablation_3d.py.*seed 9021' >/dev/null; then
    printf '[%s] seed9021 already running; duplicate launch suppressed\n' \
        "$(date --iso-8601=seconds)" >>"${watch_log}"
    exit 0
fi

cd "${project_root}"
chmod 755 launch_curriculum_trust_v4.sh
printf '[%s] two DONE markers verified; launching v4\n' \
    "$(date --iso-8601=seconds)" >>"${watch_log}"
./launch_curriculum_trust_v4.sh >>"${watch_log}" 2>&1
