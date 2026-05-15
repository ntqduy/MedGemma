#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'USAGE'
Usage:
  bash scripts/eval/eval_vqa_open.sh [sample] [num_slices] [view] [inference_mode] [slice_strategy] [extra args...]

Examples:
  bash scripts/eval/eval_vqa_open.sh 100
  bash scripts/eval/eval_vqa_open.sh full
  bash scripts/eval/eval_vqa_open.sh 100 16 axial montage center_uniform
  bash scripts/eval/eval_vqa_open.sh 100 --output-dir results/EVAL_VQA_OPEN_100

Open-ended VQA hides Choice A-D and compares generated text with Answer.
USAGE
  exit 0
fi

bash "${SCRIPT_DIR}/eval_vqa.sh" "${1:-100}" open "${@:2}"

