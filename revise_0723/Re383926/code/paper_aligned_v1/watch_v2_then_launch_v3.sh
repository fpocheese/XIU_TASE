#!/usr/bin/env bash
set -euo pipefail

v2_root=/home/a2rl/reviewer_art_mappo_paper_aligned_20260728_active_mask_slow_trust_v2
project_root=/home/a2rl/reviewer_art_mappo_paper_aligned_20260728
watch_log="${v2_root}/watch_v2_then_launch_v3.log"

case1_pid="$(tr -d '[:space:]' <"${v2_root}/logs/full_case1_seed8981.pid")"
case2_pid="$(tr -d '[:space:]' <"${v2_root}/logs/full_case2_seed8981.pid")"
printf '[%s] watching v2 pids %s %s\n' \
    "$(date --iso-8601=seconds)" "${case1_pid}" "${case2_pid}" >>"${watch_log}"

while kill -0 "${case1_pid}" 2>/dev/null || kill -0 "${case2_pid}" 2>/dev/null; do
    sleep 20
done

case1_log="${v2_root}/logs/full_case1_seed8981.log"
case2_log="${v2_root}/logs/full_case2_seed8981.log"
if ! grep -q '^\[DONE\]' "${case1_log}" || ! grep -q '^\[DONE\]' "${case2_log}"; then
    printf '[%s] v2 exited without two DONE markers; v3 not launched\n' \
        "$(date --iso-8601=seconds)" >>"${watch_log}"
    exit 1
fi

if pgrep -af 'seed 9001' >/dev/null; then
    printf '[%s] seed9001 already running; duplicate launch suppressed\n' \
        "$(date --iso-8601=seconds)" >>"${watch_log}"
    exit 0
fi

cd "${project_root}"
chmod 755 launch_nominal_trust_buffer4096_v3.sh
printf '[%s] two DONE markers verified; launching v3\n' \
    "$(date --iso-8601=seconds)" >>"${watch_log}"
./launch_nominal_trust_buffer4096_v3.sh >>"${watch_log}" 2>&1
