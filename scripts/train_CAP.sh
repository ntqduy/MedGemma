#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SAMPLE="${1:-100}"
MODE="${2:-lora}"
SPLIT="${3:-train}"

EXTRA_ARGS=()
if [ "$#" -gt 3 ]; then
  EXTRA_ARGS=("${@:4}")
fi

python "${PROJECT_ROOT}/train.py" \
  --config "${PROJECT_ROOT}/config/CAP_task.yaml" \
  --task cap \
  --sample "${SAMPLE}" \
  --train-mode "${MODE}" \
  --split "${SPLIT}" \
  "${EXTRA_ARGS[@]}"
