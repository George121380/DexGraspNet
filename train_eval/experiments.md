# Experiments Summary

- Dataset: `/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/aff_sec_result/11pro_SL_TRX_FG/aff_sec_pairs.npy` (36 pairs, N=8192)
- Model: `AffordancePointNet2SSG` (SSG SA/FP), default conditioning `rd` unless noted
- Loss (default): L1 on sigmoid probabilities

## All-pairs training

- 10 epochs (rd + L1)
  - Checkpoint: `/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/checkpoints_vis/best_model.pth`
  - Visualizations: `/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/evaluation_results_vis/visualizations/index.html`
  - Metrics: MSE 0.01779, MAE 0.04648, RMSE 0.13339, Corr -0.06307, Pred range [3.5e-05, 0.04905]

- 10 epochs (rd + L1, second quick run)
  - Checkpoint: `/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/checkpoints_run/best_model.pth`
  - Visualizations: `/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/evaluation_results_run/visualizations/index.html`
  - Metrics: MSE 0.01700, MAE 0.04884, RMSE 0.13038, Corr 0.12347, Pred range [0.000823, 0.10101]

- 50 epochs (rd + L1)
  - Checkpoint: `/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/checkpoints_run50/best_model.pth`
  - Visualizations: `/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/evaluation_results_run50/visualizations/index.html`
  - Metrics: MSE 0.01684, MAE 0.04046, RMSE 0.12979, Corr 0.24211, Pred range [1.88e-05, 0.22985]

- 100 epochs (rd + L1)
  - Checkpoint: `/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/checkpoints_run100/best_model.pth`
  - Visualizations: `/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/evaluation_results_run100/visualizations/index.html`
  - Metrics: MSE 0.01662, MAE 0.04014, RMSE 0.12893, Corr 0.34654, Pred range [7.44e-07, 0.06292]

## Ablations (20 epochs)

- r + L1
  - Checkpoint: `/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/checkpoints_ablate_r_l1/best_model.pth`
  - Visualizations: `/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/evaluation_results_ablate_r_l1/visualizations/index.html`
  - Metrics: MSE 0.01642, MAE 0.04048, RMSE 0.12816, Corr 0.33161, Pred range [0.00105, 0.10202]

- rd + Weighted L1 (alpha=9)
  - Checkpoint: `/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/checkpoints_ablate_rd_wl1/best_model.pth`
  - Note: validation loss ~0.1736 (poor convergence; skipped full eval)

- rd + BCEWithLogits (pos_weight=5)
  - Checkpoint: `/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/checkpoints_ablate_rd_bce/best_model.pth`
  - Note: validation loss ~0.4139 (unstable; skipped full eval)

## Overfit on single pair (pair_idx=0)

- 400 epochs (rd + L1), no augmentation
  - Checkpoint: `/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/checkpoints_overfit2/best_model.pth`
  - Visualization (pair 0 only): `/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/evaluation_results_overfit_pair0/visualizations/index.html`
  - Metrics (pair 0): MSE 0.000121, MAE 0.002843, RMSE 0.01101, Corr 0.99529, Pred range [5.57e-05, 0.95451]

## Notes / Next steps

- L1 is most stable on this small dataset. `r`-only conditioning is comparable to `rd`, sometimes with clearer dynamic range.
- Weighted L1 / BCE need better tuning or more data to avoid boosting background errors.
- Consider: MSG multi-scale context, larger radii, more data or cosine schedule, and adding |pred−target| error panels in visualization to localize failures.
