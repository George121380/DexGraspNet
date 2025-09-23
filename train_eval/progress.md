# Training & Evaluation Progress

## What is implemented
- Model: `models/affordance_pointnet2.py`
  - `AffordancePointNet2SSG` built from PointNet2 SSG blocks:
    - Encoder: `PointnetSAModule` x4 (npoint: 1024→256→64→16)
    - Decoder: `PointnetFPModule` x4 (FP4→FP1) back to original N points
    - Head: `Conv1d(128→1)` outputs per-point logits `(B,1,N)`
  - Conditioning (toggleable):
    - `use_condition`: bool (default true)
    - `condition_mode`: `none` | `r` | `rd` (default `rd`).
      - `r = p − c` (3 ch)
      - `d = ||p − c||` (1 ch)
      - Total extra channels `C ∈ {0,3,4}`; input shaped `(B,N,3+C)` with xyz first.
  - Library alignment: respects PointNet2 `_break_up_pc` pattern and `use_xyz=True` channel conventions.

- Dataset: `data/affordance_dataset.py`
  - `AffordancePairsDataset` consumes one `aff_sec_pairs.npy` and expands each `(left_kps, aff_scores_right)` pair into an item.
  - Returns `(points(N,3), center(3,), target(N,))` with optional Z-rotation, jitter, and point shuffling.
  - Deterministic split (`train`/`val`) from the same file with `val_ratio`.

- Training: `train_eval/train.py`
  - CLI args compatible with `script/train_eval/train.sh` (plus conditioning toggles):
    - `--data_path`, `--batch_size`, `--learning_rate`, `--max_epochs`, `--output_dir`, `--save_interval`, `--device`
    - `--use_condition`, `--condition_mode {none,r,rd}`
  - Optimizer: AdamW; gradient clip: 1.0.
  - Loss: L1/MAE on probabilities in [0,1] (apply `sigmoid` to logits inside loss).
  - Saves `best_model.pth` (by val_loss) and periodic checkpoints.

- Evaluation: `train_eval/evaluate.py`
  - Loads `best_model.pth`, builds dataset (full set), computes predictions and metrics.
  - Outputs: `predictions.npy`, `targets.npy`, `metrics.json` in `--output_dir`.
  - Metrics: MSE, MAE, RMSE, Pearson correlation, prediction range.

## Design rationale (recap)
- SA/FP mirrors PointNet++ semantic segmentation: downsample for robust local features, upsample with interpolation + skip connections for dense per-point prediction.
- Conditioning features provide relative geometry to the single conditioning point `c` without changing PointNet2 interfaces.
- Targets are already normalized to [0,1]; L1/MAE directly matches the regression nature of affordance.

## Dataset details (current file)
- Confirmed mapping: each pair has one `left_kps (1,3)` and one `aff_scores_right (N,)`; current file has 36 pairs with `N=8192`.

## Quick smoke test result
- Env: `pn` conda; `pointnet2_ops` installed via `third_party/Pointnet2_PyTorch/pointnet2_ops_lib`.
- Command (2 epochs, sanity check):
  - `python train_eval/train.py --data_path <aff_sec_pairs.npy> --batch_size 2 --learning_rate 1e-3 --max_epochs 2 --output_dir ./checkpoints_dev --device auto --save_interval 1 --use_condition --condition_mode rd`
- Output sample:
  - `Epoch 001: train_loss=0.382558 val_loss=0.446414`
  - `Epoch 002: train_loss=0.220128 val_loss=0.959628`
  - Best saved to `checkpoints_dev/best_model.pth` (val_loss=0.446414)

## How to run
- Training (via script):
  - `script/train_eval/train.sh standard --data_path /abs/path/to/aff_sec_pairs.npy`
  - or directly: see command above; ensure `PYTHONPATH` includes repo root or run from repo root.
- Evaluation:
  - `python train_eval/evaluate.py --model_path ./checkpoints_dev/best_model.pth --data_path /abs/path/to/aff_sec_pairs.npy --output_dir ./evaluation_results --device auto`

## Next steps
- Add richer augmentations/scheduler as needed.
- Implement visualization overlay for predictions (optional).
- Hyperparameter sweeps: `condition_mode` ablations (`none`, `r`, `rd`), batch size, LR.








