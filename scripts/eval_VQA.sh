#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SAMPLE="${1:-100}"
GPU_ID="${CUDA_DEVICE_ID:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

EXTRA_ARGS=()
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'USAGE'
Usage:
  bash scripts/eval_VQA.sh [sample] [num_slices] [view] [inference_mode] [slice_strategy] [extra args...]

Examples:
  bash scripts/eval_VQA.sh 1
  bash scripts/eval_VQA.sh 1 9 axial montage center_uniform
  bash scripts/eval_VQA.sh 1 --num_slices 9 --view axial --inference_mode montage --slice_strategy center_uniform

GPU:
  Default physical GPU: 0
  Override: CUDA_DEVICE_ID=1 bash scripts/eval_VQA.sh 1
USAGE
  exit 0
fi

if [ "$#" -gt 1 ] && [[ "${2}" != -* ]]; then
  NUM_SLICES="${2}"
  VIEW="${3:-axial}"
  INFERENCE_MODE="${4:-montage}"
  SLICE_STRATEGY="${5:-center_uniform}"
  EXTRA_ARGS=(
    --num_slices "${NUM_SLICES}"
    --slice_strategy "${SLICE_STRATEGY}"
    --view "${VIEW}"
    --inference_mode "${INFERENCE_MODE}"
  )
  if [ "$#" -gt 5 ]; then
    EXTRA_ARGS+=("${@:6}")
  fi
elif [ "$#" -gt 1 ]; then
  EXTRA_ARGS=("${@:2}")
fi

echo "[eval_VQA] physical_gpu=${GPU_ID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[eval_VQA] sample=${SAMPLE} extra_args=${EXTRA_ARGS[*]:-<from config>}"

python "${PROJECT_ROOT}/evaluate.py" \
  --config "${PROJECT_ROOT}/config/VQA_task.yaml" \
  --task vqa \
  --sample "${SAMPLE}" \
  "${EXTRA_ARGS[@]}"
