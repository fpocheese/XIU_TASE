#!/usr/bin/env bash
set -euo pipefail

project_root=/home/a2rl/reviewer_art_mappo_paper_aligned_20260728
run_root=/home/a2rl/reviewer_art_mappo_paper_aligned_20260728_sensor_masked_chunk64_v21
python_bin=/home/a2rl/miniconda3/envs/rlgpu/bin/python
monitor_log="${run_root}/checkpoint_monitor.log"
while [[ ! -f "${run_root}/logs/full_case1_seed9701.pid" ||
         ! -f "${run_root}/logs/full_case2_seed9701.pid" ]]; do sleep 20; done
case1_pid="$(tr -d '[:space:]' <"${run_root}/logs/full_case1_seed9701.pid")"
case2_pid="$(tr -d '[:space:]' <"${run_root}/logs/full_case2_seed9701.pid")"
printf '[%s] monitoring v21 pids %s %s\n' "$(date --iso-8601=seconds)" \
    "${case1_pid}" "${case2_pid}" >>"${monitor_log}"
cd "${project_root}"
for update in 5 10 15 20 30 45 60 100 150; do
    tag="$(printf '%04d' "${update}")"
    checkpoint1="${run_root}/training/full/case1/seed9701/models/checkpoint_update_${tag}.pt"
    checkpoint2="${run_root}/training/full/case2/seed9701/models/checkpoint_update_${tag}.pt"
    while [[ ! -f "${checkpoint1}" || ! -f "${checkpoint2}" ]]; do
        if ! kill -0 "${case1_pid}" 2>/dev/null &&
           ! kill -0 "${case2_pid}" 2>/dev/null; then exit 1; fi
        sleep 20
    done
    out="${run_root}/heldout_checkpoint_validation/update_${tag}"
    mkdir -p "${out}"
    validation_pids=()
    for case_name in case1 case2; do
        seed=100081; [[ "${case_name}" == case2 ]] && seed=100091
        one="${out}/${case_name}_checkpoint_only"; mkdir -p "${one}"
        cp "${run_root}/training/full/${case_name}/seed9701/models/checkpoint_update_${tag}.pt" "${one}/"
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
            "${python_bin}" onpolicy/scripts/select_checkpoint_frozen_validation.py \
            --case "${case_name}" --variant full --model_dir "${one}" \
            --outdir "${out}/${case_name}" --episodes 10 --workers 4 \
            --seed "${seed}" --max_steps 1500 >"${out}/${case_name}.log" 2>&1 &
        validation_pids+=("$!")
    done
    failed=0
    for pid in "${validation_pids[@]}"; do wait "${pid}" || failed=1; done
    [[ "${failed}" -eq 0 ]] || exit 1
    printf '[%s] validation complete at checkpoint %s\n' \
        "$(date --iso-8601=seconds)" "${tag}" >>"${monitor_log}"
done
