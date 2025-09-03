#!/usr/bin/env bash
set -euo pipefail

# Example runner to train the actor model using the 'affordance' conda env

PROJ_DIR="/home/peiqi621/projects/2026-CVPR-BiDexHand"
PY_BIN="/home/peiqi621/anaconda3/envs/affordance/bin/python"

RESULTS_ROOT="$PROJ_DIR/preprocess/results"
SAVE_DIR="$PROJ_DIR/outputs/actor"

DEVICE=${1:-cuda}
BATCH=${2:-2}
EPOCHS=${3:-20}

mkdir -p "$SAVE_DIR"

CMD=("$PY_BIN" "$PROJ_DIR/train_actor.py" \
  --results_root "$RESULTS_ROOT" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --num_points 8192 \
  --cond_k 64 \
  --cond_mode center \
  --target_mode right_distance \
  --dmax 0.05 \
  --gaussian_sigma 0.02 \
  --prefer_dual_afford \
  --freeze_backbones \
  --save_dir "$SAVE_DIR")

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Training finished. Checkpoint saved under: $SAVE_DIR"



