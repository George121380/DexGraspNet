#!/usr/bin/env bash
set -euo pipefail

# FUNCTION: Batch preprocessing script for BimanGrasp dataset
# Processes first N objects from BimanGrasp dataset, extracting point clouds and grasp pairs
# Prepares data for affordance training by sampling object surfaces and hand poses

# Batch preprocess ALL objects, all poses per object. Supports per-object concurrency.
# Usage:
#   bash script/run_preprocess_batch.sh [MAX_PROCS]
#   - MAX_PROCS: number of objects to process in parallel (default 4)

MAX_PROCS=${1:-8}

PROJ_DIR="/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex"
PY_BIN="/home/george/anaconda3/envs/bimangrasp/bin/python"

RESULT_PATH="$PROJ_DIR/third_party/BimanGrasp-Dataset/BimanGrasp-Dataset-Release-v1"
PREPROCESS_PY="$PROJ_DIR/preprocess/preprocess_object_all_poses.py"
OUT_DIR="$PROJ_DIR/data/preprocess_data/results"
mkdir -p "$OUT_DIR"

# Sampling settings (consistent with single-run script)
K=8192
BASE_N=30000
PRE_RAND_N=12000
X_SHIFT=0.0
USE_FPS=1

# Auto-select device (use CUDA if available)
DEVICE=$($PY_BIN -c 'import torch;print("cuda" if torch.cuda.is_available() else "cpu")')
echo "Using device: $DEVICE"

if [[ ! -d "$RESULT_PATH" ]]; then
  echo "Dataset path not found: $RESULT_PATH" >&2
  exit 1
fi

mapfile -t OBJ_FILES < <(ls -1 "$RESULT_PATH"/*.npy 2>/dev/null | sort)

if [[ ${#OBJ_FILES[@]} -eq 0 ]]; then
  echo "No .npy object files found under $RESULT_PATH" >&2
  exit 1
fi

echo "Processing ALL objects under: $RESULT_PATH (parallel objects: $MAX_PROCS)"

process_one_obj() {
  local obj_file="$1"
  local obj_name=$(basename "$obj_file")
  obj_name="${obj_name%.npy}"
  echo "Object: $obj_name (all poses)"
  CMD=(
    "$PY_BIN" "$PREPROCESS_PY"
    --object_name "$obj_name"
    --result_path "$RESULT_PATH"
    --k "$K" --base_n "$BASE_N" --pre_rand_n "$PRE_RAND_N" --x_shift "$X_SHIFT" --outer_only
    --out_dir "$OUT_DIR"
    --device "$DEVICE"
  )
  if [[ "$USE_FPS" == "1" ]]; then
    CMD+=(--use_fps)
  fi
  "${CMD[@]}"
}

pids=()
for obj_file in "${OBJ_FILES[@]}"; do
  process_one_obj "$obj_file" &
  pids+=($!)
  while (( ${#pids[@]} >= MAX_PROCS )); do
    wait -n
    pids=($(jobs -pr))
  done
done
wait

echo "Batch preprocessing done. Results under: $OUT_DIR/<obj_name>/"



