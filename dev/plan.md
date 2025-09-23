# Affordance Model Implementation Plan

## Goal
- Input: object point cloud `P ∈ R^{N×3}` and a feature point `c ∈ R^3` (or a small set of keypoints).
- Output: per-point affordance map `A ∈ R^{N}` (values in [0,1]).
- Training data: `/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/aff_sec_result/11pro_SL_TRX_FG/aff_sec_pairs.npy` (and similar under `aff_sec_result/*/aff_sec_pairs.npy`).
- Backend: PointNet++ (PointNet2) backbone from `third_party/Pointnet2_PyTorch` with set abstraction (SSG) + feature propagation for dense prediction.

## Data Understanding and I/O
- Each `aff_sec_pairs.npy` stores a dict with keys:
  - `object_name`: string
  - `points`: `(N,3)` float32 array — object point cloud
  - `pairs`: dict[int -> sample]
    - For each pose index `i`:
      - `left_kps`: `(1,3)` float32 array — exactly one conditioning point per pair (current file)
      - `aff_scores_right`: `(N,)` float32 array — per-point affordance aligned with `points` (normalized)
  - `meta`: configuration (sigma, etc.)
- Note: In `/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/aff_sec_result/11pro_SL_TRX_FG/aff_sec_pairs.npy`, there are 36 pairs; `N = 8192` for all pairs. Mapping is 1 left_kps → 1 affordance map per pair (not many-to-many).
- For our model: condition on one feature point `c = left_kps[0]`. If future datasets provide multiple keypoints per pair, we can extend via pooling (e.g., mean/attention) or set-conditioning without changing the training API.

## Model Design (`models/affordance_pointnet2.py`)
- Reuse classes from `third_party/Pointnet2_PyTorch` directly:
  - Set abstraction and grouping: `pointnet2_ops.pointnet2_modules.PointnetSAModule` (SSG)
  - Feature propagation: `pointnet2_ops.pointnet2_modules.PointnetFPModule`
  - Input formatting helper: reuse `_break_up_pc` pattern from `pointnet2/models/pointnet2_ssg_cls.py`/`pointnet2_ssg_sem.py` so inputs are `(B, N, 3 + C)` with xyz first.
- Conditioning with minimal change to interfaces:
  - For each point `p`, build per-point features `f = concat(r = p - c, d = ||p - c||)`; default `C = 4`.
  - Expose toggles for experiments:
    - `use_condition` (bool, default: true): if false, do not append conditioning features ⇒ `C = 0`.
    - `condition_mode` (str, default: 'rd'): one of `{'none','r','rd'}` mapping to `C ∈ {0,3,4}`.
    - When `use_xyz=True`, the first SA MLP input channels should be `C + 3` (e.g., 7 for `rd`, 6 for `r`, 3 for `none`).
  - Pack model input as `(B, N, 3 + C)` where the first 3 are xyz and the next `C` are features; set `use_xyz=True` in SA modules so xyz are concatenated inside ops as the library expects.
- SA/FP architecture mirroring `pointnet2_ssg_sem.py`, with correct first-MLP dims:
  - Because `use_xyz=True`, the first MLP input channel should be `C + 3` (= 7 when `C=4`).
  - Suggested SSG stack:
    - SA1: `PointnetSAModule(npoint=1024, radius=0.1, nsample=32, mlp=[C+3, 32, 32, 64], use_xyz=True)`
    - SA2: `PointnetSAModule(npoint=256,  radius=0.2, nsample=32, mlp=[64, 64, 64, 128], use_xyz=True)`
    - SA3: `PointnetSAModule(npoint=64,   radius=0.4, nsample=32, mlp=[128, 128, 128, 256], use_xyz=True)`
    - SA4: `PointnetSAModule(npoint=16,   radius=0.8, nsample=32, mlp=[256, 256, 256, 512], use_xyz=True)`
  - FP stack (mirrors semseg model):
    - `FP4: PointnetFPModule(mlp=[512 + 256, 256, 256])`
    - `FP3: PointnetFPModule(mlp=[256 + 128, 256, 256])`
    - `FP2: PointnetFPModule(mlp=[256 + 64, 256, 128])`
    - `FP1: PointnetFPModule(mlp=[128 + C+3, 128, 128, 128])`  ← input skip has channels `(C+3)` like semseg uses `128 + 6`
- Head and outputs (stay compatible with library forward style):
  - Final head: `Conv1d(128, 1, kernel_size=1)` to produce per-point logits `(B, 1, N)`.
  - Train with L1 (MAE) on probabilities: apply `sigmoid` to logits inside the loss; keep returning logits from the model, and use `sigmoid` only for evaluation/visualization to obtain `(B, N)`.
- Drop-in alternatives without interface changes:
  - MSG variant: if multi-scale features are desired, use `PointnetSAModuleMSG` to construct the SA layers with the same interfaces (only adjust `mlp` lists).
  - Minimal variant using only `xyz + r` (`C=3`): change the first SA1 MLP input from `C+3=7` to `6`; other layers unchanged.
- Input convention (aligned with the library):
  - `pointcloud` tensor has shape `(B, N, 3 + C)` with xyz first and features after; `_break_up_pc` splits `xyz` and `features` before feeding `PointnetSAModule`/FP.
- Minimal customization scope:
  - Create a thin wrapper class (e.g., `AffordancePointNet2SSG`) to assemble the modules and head; do not modify CUDA/ops or module interfaces, to benefit directly from the library.

## Dataset and Dataloader (`data/affordance_dataset.py`)
- Load from one or multiple directories under `aff_sec_result/*/aff_sec_pairs.npy`.
- For each sample (object, pose index):
  - `points`: `(N,3)`
  - `center`: `left_kps[0]` as `(3,)`
  - `target`: `(N,)` affordance
- Collate to tensors:
  - Inputs: `(B, N, C_in)` with channels as designed; centers `(B, 3)` kept separately if needed.

## Loss, Metrics, and Training
- Loss: L1 (MAE) on probabilities in [0,1]; compute `pred = sigmoid(logits)` and apply `L1Loss(pred, target)`. Optionally compare with `BCEWithLogitsLoss` as a baseline.
- Metrics: MSE, MAE, RMSE, Pearson correlation, range of predictions (as in eval.sh expects).
- Optimizer: AdamW; LR from `train.sh` args; weight decay 1e-5; gradient clipping.
- Scheduler: optional cosine or step decay.

## Training/Eval Entry Points
- Implement `train_eval/train.py` and `train_eval/evaluate.py` expected by `script/train_eval/*.sh` if not already present:
  - `train.py`:
    - Args: `--data_path`, `--batch_size`, `--learning_rate`, `--max_epochs`, `--focal_alpha`, `--focal_gamma`, `--output_dir`, `--save_interval`, `--device`, `--max_grad_norm`, `--weight_decay`.
    - Load dataset(s), build model, train loop (PyTorch), save `best_model.pth` and periodic checkpoints.
  - `evaluate.py`:
    - Args: `--model_path`, `--data_path`, `--output_dir`, `--batch_size`, `--device`, `--visualize`, `--max_vis_samples`.
    - Produce `predictions.npy`, `targets.npy`, `metrics.json`, and automatically do HTML visualizations similar to preprocess visualizers.

## PointNet2 Integration
- Use installed ops from `third_party/Pointnet2_PyTorch/pointnet2_ops_lib` via `from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetFPModule`.
- Follow the data layout expected in `pointnet2_ssg_sem.py`:
  - Input `pointcloud`: `(B, N, 3 + in_channels)` where xyz comes first.
  - Use `_break_up_pc` to split xyz and features when reusing helper functions.

## File/Code Additions
- `models/affordance_pointnet2.py`: model class `AffordancePointNet2`.
- `data/affordance_dataset.py`: dataset class `AffordancePairsDataset`.
- `train_eval/train.py`: training script integrating with `script/train_eval/train.sh`.
- `train_eval/evaluate.py`: evaluation and visualization script integrating with `eval.sh`.
- `utils/metrics.py`: optional metrics helpers.

## Minimal APIs
- Model forward:
  - Input: `points (B,N,3)`, `centers (B,3)` or pre-concatenated `(B,N,C_in)`.
  - Output: `(B,N)` affordance in [0,1] (apply sigmoid inside).
- Dataset item:
  - Returns: `points (N,3)`, `center (3,)`, `target (N,)`.

## Training Details
- Start with N=8192; batch size from script; defaultly use `cuda` if available.
- Normalize inputs (zero-mean per cloud) or leave raw to keep absolute scale (preferred to match grasp geometry). Keep center subtraction only in relative features.

## Visualization
- Reuse plotting approach from `preprocess/visualize_affordance.py` to color point cloud by predicted affordance and optionally overlay center.

## Milestones
1) Implement dataset and model.
2) Implement training loop; overfit small subset to sanity-check.
3) Implement evaluation and visualization; verify outputs saved per `eval.sh`.
4) Tune input channels, loss, and augmentations.

