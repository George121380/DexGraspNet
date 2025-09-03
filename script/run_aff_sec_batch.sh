#!/usr/bin/env bash
set -euo pipefail

# Batch generator for: left hand keypoints, whole point cloud, right-hand affordance scores
# Usage:
#   script/run_aff_sec_batch.sh [IN_ROOT] [OUT_ROOT] [SIGMA] [extra-args...]
# Defaults:
#   IN_ROOT = /home/peiqi621/projects/2026-CVPR-BiDexHand/preprocess/results
#   OUT_ROOT = /home/peiqi621/projects/2026-CVPR-BiDexHand/preprocess/aff_sec_result
#   SIGMA = 0.02
#
# Examples:
#   script/run_aff_sec_batch.sh
#   script/run_aff_sec_batch.sh \
#     /home/peiqi621/projects/2026-CVPR-BiDexHand/preprocess/results \
#     /home/peiqi621/projects/2026-CVPR-BiDexHand/preprocess/aff_sec_result \
#     0.02 --right_kps_use_center_only

PROJ="/home/peiqi621/projects/2026-CVPR-BiDexHand"
PY="/home/peiqi621/anaconda3/envs/affordance/bin/python"

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  echo "Usage: $0 [IN_ROOT] [OUT_ROOT] [SIGMA] [extra-args...]"
  exit 0
fi

IN_ROOT="${1:-$PROJ/preprocess/results}"
OUT_ROOT="${2:-$PROJ/preprocess/aff_sec_result}"
SIGMA="${3:-0.02}"

mkdir -p "$OUT_ROOT"

"$PY" "$PROJ/preprocess/aff_sec_batch.py" \
  --in_root "$IN_ROOT" \
  --out_root "$OUT_ROOT" \
  --sigma "$SIGMA" \
  --right_kps_use_center_only \
  "${@:4}"

echo "Results saved to: $OUT_ROOT"


