#!/usr/bin/env bash
set -u

REMOTE=a2rl@192.168.1.219
KNOWN_HOSTS=/tmp/codex_known_hosts_192_168_1_219
ROOT=/home/a2rl/reviewer_art_mappo_ablation_20260724

while true; do
  sshpass -e ssh \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile="${KNOWN_HOSTS}" \
    -o ConnectTimeout=10 \
    "${REMOTE}" \
    "${ROOT}/on-policy-main/onpolicy/scripts/formal_ablation_status.sh" \
    || echo "$(date --iso-8601=seconds) remote_status_unavailable"
  sleep 60
done
