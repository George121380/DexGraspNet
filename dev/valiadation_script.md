### Run all bimanual bundles under experiments (chunked)

```bash
/home/george/anaconda3/envs/dexgraspnet/bin/python grasp_generation/scripts/valiation_grasps_all.py \
  --experiments_root /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/third_party/DexGraspNet/data/experiments \
  --mesh_path /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/benchmark/meshdata \
  --result_root /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/third_party/DexGraspNet/data/dataset \
  --gpu 0 \
  --chunk_size 5000
```

- Optional: append `--max_files N` to limit files; use `--pass_args --gui` to visualize.

```bash
cd /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/third_party/DexGraspNet/grasp_generation && \
CUDA_VISIBLE_DEVICES=0 python scripts/validate_grasps.py \
  --gpu 0 \
  --bimanual \
  --object_code 100015 \
  --mesh_path /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/third_party/DexGraspNet/data/meshdata \
  --grasp_file_bimanual /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/third_party/DexGraspNet/data/BimanGrasp-Dataset/100015.npy \
  --result_path /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/third_party/DexGraspNet/data/dataset \
  --gui
‘’‘

Compute success rate
```bash
python3 /media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/third_party/DexGraspNet/grasp_generation/scripts/compute_success_rates.py \
  --bimanual-dir "/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/third_party/DexGraspNet/data/dataset/segments/bimanual" \
  --categories-file "/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/third_party/DexGraspNet/data/dataset/segments/object_by_category.json" \
  --output-json "/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/third_party/DexGraspNet/data/graspdata_bimanual_success_rates.json"
'''