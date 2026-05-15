#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'USAGE'
Usage:
  bash scripts/eval/eval_vqa_closed.sh [sample] [num_slices] [view] [inference_mode] [slice_strategy] [extra args...]

Examples:
  bash scripts/eval/eval_vqa_closed.sh 100
  bash scripts/eval/eval_vqa_closed.sh full
  bash scripts/eval/eval_vqa_closed.sh 100 16 axial montage center_uniform
  bash scripts/eval/eval_vqa_closed.sh 100 --output-dir results/EVAL_VQA_CLOSED_100

Closed-ended VQA shows Choice A-D and scores mapped option/text accuracy.
USAGE
  exit 0
fi

bash "${SCRIPT_DIR}/eval_vqa.sh" "${1:-100}" closed "${@:2}"

