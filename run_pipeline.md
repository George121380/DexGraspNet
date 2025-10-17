## pipeline.py 快速示例（中文注释）

示例 1：在 11pro 物体上运行（两阶段推理 + 可视化 + 短迭代）
```bash
# 点云（N×3 .npy）+ 两阶段权重 + 50 次优化迭代
python /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/pipeline.py \
  --points_path /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/data/preprocess_data/results/5_HTP/obj_points.npy \
  --first_ckpt /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/train_eval_first/checkpoints_agg/best_model.pth \
  --second_ckpt /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/train_eval_second/checkpoints/best_model.pth \
  --out_dir /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/outputs/5_HTP \
  --pn_env pn --bim_env bimangrasp --conda_exe conda \
  --top_k 512 --temperature 0.3 --min_pair_dist 0.05 --mask_sigma 0.03 \
  --condition_mode rd --use_condition \
  --gpu 0 --num_iterations 50 --metrics --vis --vis_frame_stride 50 \
  --exp_name pipeline_11pro_second --object_code 5_HTP
```

示例 2：在 PartNet 100017 物体上运行（两阶段推理 + 可视化）
```bash
# 使用 PartNet 点云与 meshdata 中的对象编码 merged_collision_300k_wt
python /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/pipeline.py \
  --points_path /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/data/sapien_data/partnet-mobility-dataset/100017/watertight/merged_collision_300k_wt_pc_voxel_8192.npy \
  --first_ckpt /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/train_eval_first/checkpoints_agg/best_model.pth \
  --second_ckpt /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/train_eval_second/checkpoints/best_model.pth \
  --out_dir /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/outputs/run_merged_wt \
  --pn_env pn --bim_env bimangrasp --conda_exe conda \
  --top_k 512 --temperature 0.3 --min_pair_dist 0.05 --mask_sigma 0.03 \
  --condition_mode rd --use_condition \
  --gpu 0 --num_iterations 50 --metrics --vis --vis_frame_stride 50 \
  --exp_name pipeline_merged_wt --object_code merged_collision_300k_wt
```

示例 3：最小运行（仅一阶段 + 关闭可视化，快速验证连通性）
```bash
# 第二阶段留空，右手分数回退到第一阶段；不生成视频，速度更快
python /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/pipeline.py \
  --points_path /ABS/TO/points.npy \
  --first_ckpt /ABS/TO/first_ckpt.pth \
  --second_ckpt "" \
  --out_dir /ABS/TO/outputs/demo_min \
  --pn_env pn --bim_env bimangrasp --conda_exe conda \
  --top_k 256 --temperature 0.5 \
  --gpu 0 --num_iterations 20 \
  --exp_name demo_min --object_code merged_collision_300k_wt
```



