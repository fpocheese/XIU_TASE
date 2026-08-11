#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/../.." && pwd)
output_root=${1:-/home/a2rl/reviewer_xiu_ablation_20260729/formal_81920}
steps=${2:-81920}
max_parallel=${3:-2}
shift $(( $# >= 3 ? 3 : $# ))

if (( $# > 0 )); then
    seeds=("$@")
else
    seeds=(8301 8302 8303)
fi

mkdir -p "$output_root"
for seed in "${seeds[@]}"; do
    seed_log="$output_root/launcher_seed${seed}.log"
    bash "$project_root/onpolicy/scripts/launch_xiu_ablation_pilot.sh" \
        "$output_root" "$seed" "$steps" "$max_parallel" \
        >"$seed_log" 2>&1
done
