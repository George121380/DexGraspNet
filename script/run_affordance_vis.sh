#!/usr/bin/env bash
set -euo pipefail

# Run affordance visualization and save HTML to preprocess/affordance_vis directory.
# Usage:
#   bash script/run_affordance.sh [OBJECT_NAME] [NUM]
# Defaults:
#   OBJECT_NAME=Asus_M5A99FX_PRO_R20_Motherboard_ATX_Socket_AM3
#   NUM=0

OBJ_NAME=${1:-3D_Dollhouse_Sofa}
NUM=${2:-0}

PROJ_DIR="/home/peiqi621/projects/2026-CVPR-BiDexHand"
PY_BIN="/home/peiqi621/anaconda3/envs/affordance/bin/python"

SCRIPT_PY="$PROJ_DIR/preprocess/visualize_affordance.py"
RESULT_PATH="$PROJ_DIR/third_party/BimanGrasp-Dataset/BimanGrasp-Dataset-Release-v1"
OUT_DIR="$PROJ_DIR/preprocess"
AFF_DIR="$OUT_DIR/affordance_vis"
mkdir -p "$AFF_DIR"

# Sampling settings (edit as needed)
K=8192
DMAX=0.05
BASE_N=30000
PRE_RAND_N=12000
X_SHIFT=0.0
USE_FPS=1  # 1 to enable pure PyTorch FPS; 0 to use random downsampling

OUT_HTML="$AFF_DIR/affordance_${OBJ_NAME}_${NUM}.html"

CMD=("$PY_BIN" "$SCRIPT_PY" \
  --object_name "$OBJ_NAME" \
  --num "$NUM" \
  --k "$K" \
  --dmax "$DMAX" \
  --base_n "$BASE_N" \
  --pre_rand_n "$PRE_RAND_N" \
  --x_shift "$X_SHIFT" \
  --result_path "$RESULT_PATH" \
  --outer_only \
  --save_html "$OUT_HTML")

if [[ "$USE_FPS" == "1" ]]; then
  CMD+=(--use_fps)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Saved HTML: $OUT_HTML"


