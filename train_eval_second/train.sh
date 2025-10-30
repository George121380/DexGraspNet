python /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/train_eval_second/train.py \
  --data_dir /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/data/preprocess_data/pot_data/train \
  --output_dir /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/train_eval_second/checkpoints_sapien \
  --batch_size 16 --device auto --condition_mode kp "$@"