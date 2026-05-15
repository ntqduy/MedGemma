#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SAMPLE="${1:-100}"
SPLIT="${2:-test1k}"
GPU_ID="${CUDA_DEVICE_ID:-1}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

EXTRA_ARGS=()
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'USAGE'
Usage:
  bash scripts/eval_CAP.sh [sample] [split] [num_slices] [view] [inference_mode] [slice_strategy] [extra args...]

Examples:
  bash scripts/eval_CAP.sh 1 test1k
  bash scripts/eval_CAP.sh 1 test1k 9 axial montage center_uniform
  bash scripts/eval_CAP.sh 1 test1k --num_slices 9 --view axial --inference_mode montage --slice_strategy center_uniform

GPU:
  Default physical GPU: 1
  Override: CUDA_DEVICE_ID=0 bash scripts/eval_CAP.sh 1 test1k
USAGE
  exit 0
fi

if [ "$#" -gt 2 ] && [[ "${3}" != -* ]]; then
  NUM_SLICES="${3}"
  VIEW="${4:-axial}"
  INFERENCE_MODE="${5:-montage}"
  SLICE_STRATEGY="${6:-center_uniform}"
  EXTRA_ARGS=(
    --num_slices "${NUM_SLICES}"
    --slice_strategy "${SLICE_STRATEGY}"
    --view "${VIEW}"
    --inference_mode "${INFERENCE_MODE}"
  )
  if [ "$#" -gt 6 ]; then
    EXTRA_ARGS+=("${@:7}")
  fi
elif [ "$#" -gt 2 ]; then
  EXTRA_ARGS=("${@:3}")
fi

echo "[eval_CAP] physical_gpu=${GPU_ID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[eval_CAP] sample=${SAMPLE} split=${SPLIT} extra_args=${EXTRA_ARGS[*]:-<from config>}"

python "${PROJECT_ROOT}/evaluate.py" \
  --config "${PROJECT_ROOT}/config/CAP_task.yaml" \
  --task cap \
  --sample "${SAMPLE}" \
  --split "${SPLIT}" \
  "${EXTRA_ARGS[@]}"
