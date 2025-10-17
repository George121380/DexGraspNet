conda run -n pn python train_eval/evaluate.py \
  --model_path /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/train_eval/checkpoints_all/best_model.pth \
  --data_path /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/data/preprocess_data/aff_sec_result/Dino_3/aff_sec_pairs.npy \
  --output_dir train_eval/evaluation_results \
  --device auto --batch_size 2 --visualize --max_vis_samples 6