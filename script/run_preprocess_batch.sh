#!/usr/bin/env bash
set -euo pipefail

# Batch preprocess the first N objects (default 5), all poses per object.
# Usage:
#   bash script/run_preprocess_batch.sh [TOP_K]
# Env: assumes run_preprocess.sh is configured and works per single (object, num)

TOP_K=${1:-5}

PROJ_DIR="/home/peiqi621/projects/2026-CVPR-BiDexHand"
PY_BIN="/home/peiqi621/anaconda3/envs/affordance/bin/python"

RESULT_PATH="$PROJ_DIR/third_party/BimanGrasp-Dataset/BimanGrasp-Dataset-Release-v1"
RUN_ONE="$PROJ_DIR/script/run_preprocess.sh"

if [[ ! -d "$RESULT_PATH" ]]; then
  echo "Dataset path not found: $RESULT_PATH" >&2
  exit 1
fi

mapfile -t OBJ_FILES < <(ls -1 "$RESULT_PATH"/*.npy 2>/dev/null | sort | head -n "$TOP_K")

if [[ ${#OBJ_FILES[@]} -eq 0 ]]; then
  echo "No .npy object files found under $RESULT_PATH" >&2
  exit 1
fi

echo "Processing first $TOP_K objects under: $RESULT_PATH"

for obj_file in "${OBJ_FILES[@]}"; do
  obj_name=$(basename "$obj_file")
  obj_name="${obj_name%.npy}"

  # Determine number of poses for this object
  pose_cnt=$($PY_BIN -c "import numpy as np,sys; d=np.load(sys.argv[1], allow_pickle=True); print(len(d))" "$obj_file")
  echo "Object: $obj_name (poses: $pose_cnt)"

  if [[ "$pose_cnt" -lt 1 ]]; then
    echo "  Skip: no poses"; continue
  fi

  for ((i=0; i<pose_cnt; i++)); do
    echo "  >> Preprocess $obj_name pose $i/$((pose_cnt-1))"
    bash "$RUN_ONE" "$obj_name" "$i"
  done
done

echo "Batch preprocessing done. Results under: $PROJ_DIR/preprocess/results/<obj_name>/"



