python /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/train_eval_first/train.py \
  --data_dir /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/data/preprocess_data/results \
  --output_dir /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/train_eval_first/checkpoints_all \
  --batch_size 64 --max_epochs 1000 --device auto "$@"


