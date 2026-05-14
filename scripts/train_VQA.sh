#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SAMPLE="${1:-100}"
MODE="${2:-lora}"

EXTRA_ARGS=()
if [ "$#" -gt 2 ]; then
  EXTRA_ARGS=("${@:3}")
fi

python "${PROJECT_ROOT}/train.py" \
  --config "${PROJECT_ROOT}/config/VQA_task.yaml" \
  --task vqa \
  --sample "${SAMPLE}" \
  --train-mode "${MODE}" \
  "${EXTRA_ARGS[@]}"
