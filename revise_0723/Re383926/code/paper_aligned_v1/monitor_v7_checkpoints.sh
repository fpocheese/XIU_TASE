#!/usr/bin/env bash
set -euo pipefail

project_root=/home/a2rl/reviewer_art_mappo_paper_aligned_20260728
run_root=/home/a2rl/reviewer_art_mappo_paper_aligned_20260728_onpolicy_mixture_v7
python_bin=/home/a2rl/miniconda3/envs/rlgpu/bin/python
monitor_log="${run_root}/checkpoint_monitor.log"

while [[ ! -f "${run_root}/logs/full_case1_seed9101.pid" ||
         ! -f "${run_root}/logs/full_case2_seed9101.pid" ]]; do
    sleep 20
done
case1_pid="$(tr -d '[:space:]' <"${run_root}/logs/full_case1_seed9101.pid")"
case2_pid="$(tr -d '[:space:]' <"${run_root}/logs/full_case2_seed9101.pid")"
printf '[%s] monitoring v7 pids %s %s\n' \
    "$(date --iso-8601=seconds)" "${case1_pid}" "${case2_pid}" >>"${monitor_log}"

cd "${project_root}"
for update in 5 15 30 60 100 150; do
    tag="$(printf '%04d' "${update}")"
    checkpoint1="${run_root}/training/full/case1/seed9101/models/checkpoint_update_${tag}.pt"
    checkpoint2="${run_root}/training/full/case2/seed9101/models/checkpoint_update_${tag}.pt"
    while [[ ! -f "${checkpoint1}" || ! -f "${checkpoint2}" ]]; do
        if ! kill -0 "${case1_pid}" 2>/dev/null &&
           ! kill -0 "${case2_pid}" 2>/dev/null; then
            printf '[%s] v7 exited before checkpoint %s\n' \
                "$(date --iso-8601=seconds)" "${tag}" >>"${monitor_log}"
            exit 1
        fi
        sleep 20
    done

    out="${run_root}/heldout_checkpoint_validation/update_${tag}"
    mkdir -p "${out}"
    validation_pids=()
    for case_name in case1 case2; do
        seed=99881
        [[ "${case_name}" == case2 ]] && seed=99891
        one="${out}/${case_name}_checkpoint_only"
        mkdir -p "${one}"
        cp "${run_root}/training/full/${case_name}/seed9101/models/checkpoint_update_${tag}.pt" "${one}/"
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
            "${python_bin}" onpolicy/scripts/select_checkpoint_frozen_validation.py \
                --case "${case_name}" --variant full \
                --model_dir "${one}" --outdir "${out}/${case_name}" \
                --episodes 10 --workers 4 --seed "${seed}" --max_steps 1500 \
                >"${out}/${case_name}.log" 2>&1 &
        validation_pids+=("$!")
    done
    validation_failed=0
    for validation_pid in "${validation_pids[@]}"; do
        if ! wait "${validation_pid}"; then
            validation_failed=1
        fi
    done
    if [[ "${validation_failed}" -ne 0 ]]; then
        printf '[%s] validation failed at checkpoint %s\n' \
            "$(date --iso-8601=seconds)" "${tag}" >>"${monitor_log}"
        exit 1
    fi
    printf '[%s] validation complete at checkpoint %s\n' \
        "$(date --iso-8601=seconds)" "${tag}" >>"${monitor_log}"
done
