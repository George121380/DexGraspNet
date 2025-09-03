#!/usr/bin/env bash
set -euo pipefail

# Train actor using aff_sec_batch precomputed dataset

PROJ_DIR="/home/peiqi621/projects/2026-CVPR-BiDexHand"
PY_BIN="/home/peiqi621/anaconda3/envs/affordance/bin/python"

AFF_SEC_ROOT="$PROJ_DIR/preprocess/aff_sec_result"
SAVE_DIR="$PROJ_DIR/outputs/actor_aff_sec"

DEVICE=${1:-cuda}
BATCH=${2:-2}
EPOCHS=${3:-20}

mkdir -p "$SAVE_DIR"

CMD=("$PY_BIN" "$PROJ_DIR/train_actor.py" \
  --data_mode aff_sec \
  --aff_sec_root "$AFF_SEC_ROOT" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --num_points 8192 \
  --cond_k 32 \
  --prefer_dual_afford \
  --freeze_backbones \
  --save_dir "$SAVE_DIR")

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Training finished. Checkpoint saved under: $SAVE_DIR"



