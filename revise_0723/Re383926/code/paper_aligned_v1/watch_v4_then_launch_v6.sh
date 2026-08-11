#!/usr/bin/env bash
set -euo pipefail

v4_root=/home/a2rl/reviewer_art_mappo_paper_aligned_20260728_curriculum_trust_v4
project_root=/home/a2rl/reviewer_art_mappo_paper_aligned_20260728
watch_log="${v4_root}/watch_v4_then_launch_v6.log"

case1_pid="$(tr -d '[:space:]' <"${v4_root}/logs/full_case1_seed9021.pid")"
case2_pid="$(tr -d '[:space:]' <"${v4_root}/logs/full_case2_seed9021.pid")"
printf '[%s] watching v4 pids %s %s\n' \
    "$(date --iso-8601=seconds)" "${case1_pid}" "${case2_pid}" >>"${watch_log}"

while kill -0 "${case1_pid}" 2>/dev/null || kill -0 "${case2_pid}" 2>/dev/null; do
    sleep 20
done

case1_log="${v4_root}/logs/full_case1_seed9021.log"
case2_log="${v4_root}/logs/full_case2_seed9021.log"
if ! grep -q '^\[DONE\]' "${case1_log}" || ! grep -q '^\[DONE\]' "${case2_log}"; then
    printf '[%s] v4 exited without two DONE markers; v6 not launched\n' \
        "$(date --iso-8601=seconds)" >>"${watch_log}"
    exit 1
fi
if pgrep -af 'train_art_mappo_ablation_3d.py.*seed 9081' >/dev/null; then
    printf '[%s] seed9081 already running; duplicate launch suppressed\n' \
        "$(date --iso-8601=seconds)" >>"${watch_log}"
    exit 0
fi

cd "${project_root}"
chmod 755 launch_continuous_ppo1_v6.sh
printf '[%s] two v4 DONE markers verified; launching v6\n' \
    "$(date --iso-8601=seconds)" >>"${watch_log}"
./launch_continuous_ppo1_v6.sh >>"${watch_log}" 2>&1
