#!/usr/bin/env bash
set -euo pipefail

# Evaluate actor checkpoint on aff_sec data

PROJ_DIR="/home/peiqi621/projects/2026-CVPR-BiDexHand"
PY_BIN="/home/peiqi621/anaconda3/envs/affordance/bin/python"

AFF_SEC_ROOT="$PROJ_DIR/preprocess/aff_sec_result"
CKPT_PATH=${1:-"$PROJ_DIR/outputs/actor_smoke_cpu/actor.pt"}
DEVICE=${2:-cpu}

CMD=("$PY_BIN" "$PROJ_DIR/eval_actor.py" \
  --data_mode aff_sec \
  --aff_sec_root "$AFF_SEC_ROOT" \
  --checkpoint "$CKPT_PATH" \
  --device "$DEVICE")

echo "Running: ${CMD[*]}"
"${CMD[@]}"



