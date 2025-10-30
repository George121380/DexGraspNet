python train_eval_second/evaluate.py \
  --model_path /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/train_eval_second/checkpoints_sapien/best_model.pth \
  --data_path /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/data/preprocess_data/pot_data/eval/100047/aff_sec_pairs.npy \
  --output_dir train_eval_second/evaluation_results \
  --device auto --batch_size 2 --visualize --max_vis_samples 6