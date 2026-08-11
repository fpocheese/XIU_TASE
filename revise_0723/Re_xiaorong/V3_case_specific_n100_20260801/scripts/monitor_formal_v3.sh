#!/usr/bin/env bash
set -u

root=${1:-/home/a2rl/reviewer_xiu_ablation_domainrand_v3_20260801}
heartbeat="${root}/logs/remote_heartbeat.csv"
if [[ ! -s "${heartbeat}" ]]; then
    printf 'time,train_processes,training_done,training_failed,latest_update_sum,post_pipeline_complete,gpu_memory_mib,gpu_util_percent\n' \
        > "${heartbeat}"
fi

while :; do
    now=$(date --iso-8601=seconds)
    train_processes=$(pgrep -fc "${root}/code/onpolicy/scripts/train_xiu_art_ablation_paper_nominal.py" || true)
    training_done=$(find "${root}/logs" -name '*_seed8303.done' -type f -size +0c 2>/dev/null | wc -l)
    training_failed=$(find "${root}/logs" -name '*.failed' -type f -size +0c 2>/dev/null | wc -l)
    latest_update_sum=$(awk -F, 'FNR>1 {u[FILENAME]=$1} END {s=0; for (f in u) s+=u[f]; print s+0}' \
        "${root}"/training/*/*/seed8303/training_metrics.csv 2>/dev/null || printf '0')
    post_complete=0
    [[ -s "${root}/post_training_pipeline_complete.txt" ]] && post_complete=1
    gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    gpu_memory=${gpu%%,*}; gpu_memory=${gpu_memory// /}
    gpu_util=${gpu##*,}; gpu_util=${gpu_util// /}
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${now}" "${train_processes}" "${training_done}" \
        "${training_failed}" "${latest_update_sum}" "${post_complete}" \
        "${gpu_memory:-NA}" "${gpu_util:-NA}" >> "${heartbeat}"
    if (( post_complete == 1 || training_failed > 0 )); then break; fi
    sleep 60
done
