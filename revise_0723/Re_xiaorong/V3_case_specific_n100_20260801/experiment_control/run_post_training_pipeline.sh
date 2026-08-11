#!/usr/bin/env bash
set -euo pipefail

experiment_root=${1:-/home/a2rl/reviewer_xiu_ablation_case_specific_20260731}
seed=${2:-8303}
code_root="${experiment_root}/code"
training_root="${experiment_root}/training"
evaluation_root="${experiment_root}/formal_evaluation_n100"
analysis_root="${experiment_root}/analysis"
figure_root="${experiment_root}/figures_v10"
pipeline_log="${experiment_root}/post_training_pipeline.log"

exec >>"${pipeline_log}" 2>&1
printf 'pipeline_start=%s\n' "$(date --iso-8601=seconds)"

while [[ ! -s "${experiment_root}/training_suite_complete.txt" ]]; do
    printf 'waiting_for_training=%s\n' "$(date --iso-8601=seconds)"
    sleep 30
done

if ! grep -q 'failures=0' "${experiment_root}/training_suite_complete.txt"; then
    printf 'pipeline_abort_training_failure=%s\n' "$(date --iso-8601=seconds)"
    exit 2
fi

printf 'evaluation_start=%s\n' "$(date --iso-8601=seconds)"
PYTHONPATH="${code_root}" /home/a2rl/miniconda3/envs/rlgpu/bin/python -u \
    "${code_root}/onpolicy/scripts/run_formal_ablation_evaluation.py" \
    --training_root "${training_root}" \
    --outroot "${evaluation_root}" \
    --seeds "${seed}" \
    --validation_episodes 20 \
    --validation_workers 5 \
    --checkpoint_stride 5 \
    --test_episodes 100 \
    --test_workers 5 \
    --max_parallel 2

printf 'analysis_start=%s\n' "$(date --iso-8601=seconds)"
/home/a2rl/miniconda3/envs/rlgpu/bin/python -u \
    "${experiment_root}/analyze_case_specific_ablation.py" \
    --training_root "${training_root}" \
    --episodes_csv "${evaluation_root}/combined/episodes.csv" \
    --evaluation_manifest "${evaluation_root}/formal_evaluation_manifest.json" \
    --outdir "${analysis_root}"

printf 'table_generation_start=%s\n' "$(date --iso-8601=seconds)"
/home/a2rl/miniconda3/envs/rlgpu/bin/python -u \
    "${experiment_root}/generate_paper_tables.py" \
    --analysis_dir "${analysis_root}" \
    --outdir "${experiment_root}/tables"

printf 'plot_start=%s\n' "$(date --iso-8601=seconds)"
/home/a2rl/miniconda3/envs/rlgpu/bin/python -u \
    "${experiment_root}/plot_case_specific_ablation_v10.py" \
    --training_root "${training_root}" \
    --episodes_csv "${evaluation_root}/combined/episodes.csv" \
    --outdir "${figure_root}"

printf 'core_validation_start=%s\n' "$(date --iso-8601=seconds)"
/home/a2rl/miniconda3/envs/rlgpu/bin/python -u \
    "${experiment_root}/validate_case_specific_delivery.py" \
    --experiment_root "${experiment_root}" \
    --out "${experiment_root}/CORE_VALIDATION.json"

find "${experiment_root}" -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum > "${experiment_root}/SHA256SUMS.txt"
printf 'pipeline_complete=%s\n' "$(date --iso-8601=seconds)" \
    > "${experiment_root}/post_training_pipeline_complete.txt"
printf 'pipeline_complete=%s\n' "$(date --iso-8601=seconds)"
