#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SAMPLE="${1:-100}"
GPU_IDS="${CUDA_DEVICE_IDS:-${CUDA_DEVICE_ID:-}}"

if [ -n "${GPU_IDS}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi

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
  Default: use cuda_visible_devices from config/VQA_task.yaml
  Multi-GPU override: CUDA_DEVICE_IDS=0,1 bash scripts/eval_VQA.sh 1
  Single-GPU override: CUDA_DEVICE_IDS=0 bash scripts/eval_VQA.sh 1
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

echo "[eval_VQA] env_physical_gpus=${GPU_IDS:-<from config>} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[eval_VQA] sample=${SAMPLE} extra_args=${EXTRA_ARGS[*]:-<from config>}"

python "${PROJECT_ROOT}/evaluate.py" \
  --config "${PROJECT_ROOT}/config/VQA_task.yaml" \
  --task vqa \
  --sample "${SAMPLE}" \
  "${EXTRA_ARGS[@]}"
