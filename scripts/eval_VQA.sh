#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SAMPLE="${1:-100}"

EXTRA_ARGS=()
if [ "$#" -gt 1 ]; then
  EXTRA_ARGS=("${@:2}")
fi

python "${PROJECT_ROOT}/evaluate.py" \
  --config "${PROJECT_ROOT}/config/VQA_task.yaml" \
  --task vqa \
  --sample "${SAMPLE}" \
  "${EXTRA_ARGS[@]}"
