#!/usr/bin/env bash
set -uo pipefail

REMOTE=a2rl@192.168.1.219
KNOWN_HOSTS=/tmp/codex_known_hosts_192_168_1_219
REMOTE_ROOT=/home/a2rl/reviewer_art_mappo_ablation_20260724
LOCAL_ROOT=/home/uav/00gao_xueshu/DT_PAPER/XIU_code/reviewer_ablation_artifacts/results/formal_paper_ablation_5seed
SCRIPT_ROOT=/home/uav/00gao_xueshu/DT_PAPER/XIU_code/reviewer_ablation_artifacts
SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=${KNOWN_HOSTS} -o ConnectTimeout=12 -o ServerAliveInterval=15"

if test -z "${SSHPASS:-}"; then
  echo "SSHPASS must be set"
  exit 2
fi

mkdir -p "${LOCAL_ROOT}"

while true; do
  ready="$(
    sshpass -e ssh \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile="${KNOWN_HOSTS}" \
      -o ConnectTimeout=12 \
      -o ServerAliveInterval=15 \
      "${REMOTE}" \
      "test -s '${REMOTE_ROOT}/formal_paper_ablation_5seed_analysis/analysis_manifest.json' && echo ready" \
      2>/dev/null
  )"
  if test "${ready}" = "ready"; then
    break
  fi
  echo "$(date --iso-8601=seconds) waiting_for_analysis"
  sleep 120
done

sync_tree() {
  local remote_path="$1"
  local local_path="$2"
  shift 2
  while true; do
    mkdir -p "${local_path}"
    if sshpass -e rsync \
      -a --partial --human-readable --info=stats1 \
      -e "${SSH_COMMAND}" \
      "$@" \
      "${REMOTE}:${remote_path}/" "${local_path}/"; then
      return 0
    fi
    echo "$(date --iso-8601=seconds) rsync_retry ${remote_path}"
    sleep 60
  done
}

sync_tree \
  "${REMOTE_ROOT}/formal_paper_ablation_5seed" \
  "${LOCAL_ROOT}/training" \
  --exclude='checkpoint_update_*.pt' \
  --exclude='recovery_archive_*' \
  --exclude='__pycache__'
sync_tree \
  "${REMOTE_ROOT}/formal_paper_ablation_5seed_eval" \
  "${LOCAL_ROOT}/evaluation" \
  --exclude='__pycache__'
sync_tree \
  "${REMOTE_ROOT}/formal_paper_ablation_5seed_analysis" \
  "${LOCAL_ROOT}/analysis" \
  --exclude='__pycache__'
sync_tree \
  "${REMOTE_ROOT}/on-policy-main" \
  "${LOCAL_ROOT}/source_snapshot" \
  --include='/ART_MAPPO_ABLATION_README.md' \
  --include='/paper_case_presets_original_assignment_verified.npz' \
  --include='/onpolicy/' \
  --include='/onpolicy/config.py' \
  --include='/onpolicy/algorithms/' \
  --include='/onpolicy/algorithms/r_mappo/' \
  --include='/onpolicy/algorithms/r_mappo/algorithm/' \
  --include='/onpolicy/algorithms/r_mappo/algorithm/r_actor_critic_art.py' \
  --include='/onpolicy/algorithms/r_mappo/algorithm/rMAPPOPolicy_art.py' \
  --include='/onpolicy/algorithms/r_mappo/r_mappo_art.py' \
  --include='/onpolicy/runner/' \
  --include='/onpolicy/runner/shared/' \
  --include='/onpolicy/runner/shared/art_ablation_runner.py' \
  --include='/onpolicy/envs/' \
  --include='/onpolicy/envs/env_wrappers.py' \
  --include='/onpolicy/envs/mpe/' \
  --include='/onpolicy/envs/mpe/MPE_env.py' \
  --include='/onpolicy/envs/mpe/core.py' \
  --include='/onpolicy/envs/mpe/environment.py' \
  --include='/onpolicy/envs/mpe/scenarios/' \
  --include='/onpolicy/envs/mpe/scenarios/simple_world_comm_3d.py' \
  --include='/onpolicy/scripts/' \
  --include='/onpolicy/scripts/train_art_mappo_ablation_3d.py' \
  --include='/onpolicy/scripts/eval_art_mappo_ablation_3d.py' \
  --include='/onpolicy/scripts/run_art_mappo_ablation_suite.py' \
  --include='/onpolicy/scripts/run_art_mappo_ablation_eval_suite.py' \
  --include='/onpolicy/scripts/analyze_art_mappo_ablation.py' \
  --include='/onpolicy/utils/' \
  --include='/onpolicy/utils/valuenorm.py' \
  --exclude='*'

mkdir -p "${LOCAL_ROOT}/experiment_control"
cp \
  "${SCRIPT_ROOT}/launch_formal_paper_ablation.sh" \
  "${SCRIPT_ROOT}/formal_ablation_status.sh" \
  "${SCRIPT_ROOT}/monitor_formal_remote.sh" \
  "${SCRIPT_ROOT}/watch_and_fetch_formal.sh" \
  "${SCRIPT_ROOT}/validate_formal_results.py" \
  "${LOCAL_ROOT}/experiment_control/"
mkdir -p \
  "${LOCAL_ROOT}/experiment_control/state" \
  "${LOCAL_ROOT}/experiment_control/logs"
cp -a \
  "${SCRIPT_ROOT}/state/." \
  "${LOCAL_ROOT}/experiment_control/state/"
cp -a \
  "${SCRIPT_ROOT}/logs/." \
  "${LOCAL_ROOT}/experiment_control/logs/"

python "${SCRIPT_ROOT}/validate_formal_results.py" "${LOCAL_ROOT}"
(
  cd "${LOCAL_ROOT}" || exit 1
  find . -type f ! -name SHA256SUMS.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS.txt
)
date --iso-8601=seconds > "${LOCAL_ROOT}/FETCH_COMPLETE"
echo "$(date --iso-8601=seconds) fetch_and_validation_complete ${LOCAL_ROOT}"
