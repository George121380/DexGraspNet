#!/usr/bin/env bash
set -euo pipefail

# FUNCTION: Single object preprocessing script
# Extracts point cloud surface and bimanual grasp pose pairs for one specific object
# Converts BimanGrasp dataset format to training-ready point cloud + grasp data

# Run preprocessing: export object point cloud and left/right grasp (qpos + center point) pairs.
# Usage:
#   bash script/run_preprocess.sh [OBJECT_NAME] [NUM]
# Defaults:
#   OBJECT_NAME=3D_Dollhouse_Sofa
#   NUM=0

OBJ_NAME=${1:-3D_Dollhouse_Sofa}
NUM=${2:-0}

PROJ_DIR="/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex"
PY_BIN="/home/george/anaconda3/envs/bimangrasp/bin/python"

SCRIPT_PY="$PROJ_DIR/preprocess/preprocess.py"
RESULT_PATH="$PROJ_DIR/third_party/BimanGrasp-Dataset/BimanGrasp-Dataset-Release-v1"
OUT_DIR="$PROJ_DIR/preprocess/results"
mkdir -p "$OUT_DIR"

# Sampling settings
K=8192
BASE_N=30000
PRE_RAND_N=12000
X_SHIFT=0.0
USE_FPS=1  # 1 to enable FPS; 0 for random downsampling

CMD=("$PY_BIN" "$SCRIPT_PY" \
  --object_name "$OBJ_NAME" \
  --result_path "$RESULT_PATH" \
  --num "$NUM" \
  --k "$K" \
  --base_n "$BASE_N" \
  --pre_rand_n "$PRE_RAND_N" \
  --x_shift "$X_SHIFT" \
  --outer_only \
  --out_dir "$OUT_DIR")

if [[ "$USE_FPS" == "1" ]]; then
  CMD+=(--use_fps)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Done. Outputs (per-object folder):"
echo "  $OUT_DIR/${OBJ_NAME}/obj_points.npy"
echo "  $OUT_DIR/${OBJ_NAME}/grasp_pairs_${NUM}.npy"


