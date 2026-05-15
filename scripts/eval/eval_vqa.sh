#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SAMPLE="${1:-100}"
GPU_IDS="${CUDA_DEVICE_IDS:-${CUDA_DEVICE_ID:-}}"

if [ -n "${GPU_IDS}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi

EXTRA_ARGS=()
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'USAGE'
Usage:
  bash scripts/eval/eval_vqa.sh [sample] [open|closed] [num_slices] [view] [inference_mode] [slice_strategy] [extra args...]

Examples:
  bash scripts/eval/eval_vqa.sh 100
  bash scripts/eval/eval_vqa.sh 100 open
  bash scripts/eval/eval_vqa.sh 100 closed
  bash scripts/eval/eval_vqa.sh full
  bash scripts/eval/eval_vqa.sh 100 closed 16 axial montage center_uniform
  bash scripts/eval/eval_vqa.sh 100 --num_slices 16 --view axial --inference_mode montage --slice_strategy center_uniform
  bash scripts/eval/eval_vqa.sh 100 --vqa-mode closed --num_slices 16

GPU:
  Default: use cuda_visible_devices from config/VQA_task.yaml.
  Override: CUDA_DEVICE_IDS=0,1 bash scripts/eval/eval_vqa.sh 100

Mode:
  If open/closed is omitted, this script uses config/VQA_task.yaml vqa_eval_mode.
USAGE
  exit 0
fi

REMAINING=()
if [ "$#" -gt 1 ]; then
  REMAINING=("${@:2}")
fi

if [ "${#REMAINING[@]}" -gt 0 ] && [[ "${REMAINING[0]}" =~ ^(open|closed)$ ]]; then
  EXTRA_ARGS+=(--vqa-mode "${REMAINING[0]}")
  REMAINING=("${REMAINING[@]:1}")
fi

if [ "${#REMAINING[@]}" -gt 0 ] && [[ "${REMAINING[0]}" != -* ]]; then
  NUM_SLICES="${REMAINING[0]}"
  VIEW="${REMAINING[1]:-axial}"
  INFERENCE_MODE="${REMAINING[2]:-montage}"
  SLICE_STRATEGY="${REMAINING[3]:-center_uniform}"
  EXTRA_ARGS=(
    "${EXTRA_ARGS[@]}"
    --num_slices "${NUM_SLICES}"
    --slice_strategy "${SLICE_STRATEGY}"
    --view "${VIEW}"
    --inference_mode "${INFERENCE_MODE}"
  )
  if [ "${#REMAINING[@]}" -gt 4 ]; then
    EXTRA_ARGS+=("${REMAINING[@]:4}")
  fi
elif [ "${#REMAINING[@]}" -gt 0 ]; then
  EXTRA_ARGS+=("${REMAINING[@]}")
fi

echo "[eval_vqa] env_physical_gpus=${GPU_IDS:-<from config>} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[eval_vqa] sample=${SAMPLE} extra_args=${EXTRA_ARGS[*]:-<from config>}"

python "${PROJECT_ROOT}/src/eval/eval_vqa.py" \
  --sample "${SAMPLE}" \
  "${EXTRA_ARGS[@]}"
