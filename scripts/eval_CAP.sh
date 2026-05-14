#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SAMPLE="${1:-100}"
SPLIT="${2:-test1k}"

EXTRA_ARGS=()
if [ "$#" -gt 2 ]; then
  EXTRA_ARGS=("${@:3}")
fi

python "${PROJECT_ROOT}/evaluate.py" \
  --config "${PROJECT_ROOT}/config/CAP_task.yaml" \
  --task cap \
  --sample "${SAMPLE}" \
  --split "${SPLIT}" \
  "${EXTRA_ARGS[@]}"
