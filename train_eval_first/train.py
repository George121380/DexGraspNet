import os
import argparse
import json
import sys
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Disable cuDNN by default for stability on older PyTorch builds with new GPUs
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Ensure repo root on sys.path
CUR_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CUR_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:
    def tqdm(x, **kwargs):  # type: ignore
        return x

# Robust imports
try:
    from models.affordance_first import AffordanceFirstPointNet2SSG  # type: ignore
except Exception as _e:
    import importlib.util as _ilu
    _m_path = os.path.abspath(os.path.join(REPO_ROOT, 'models', 'affordance_first.py'))
    _spec = _ilu.spec_from_file_location('affordance_first', _m_path)
    assert _spec and _spec.loader
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)  # type: ignore
    AffordanceFirstPointNet2SSG = getattr(_mod, 'AffordanceFirstPointNet2SSG')

try:
    from data.affordance_first_dataset import AffordanceFirstPairsDataset, AffordanceFirstPairsMultiDataset, AffordanceFirstAggregatedDataset, AffordanceFirstAggregatedMultiDataset  # type: ignore
except Exception:
    import importlib.util as _ilu2
    _d_path = os.path.abspath(os.path.join(REPO_ROOT, 'data', 'affordance_first_dataset.py'))
    _spec2 = _ilu2.spec_from_file_location('affordance_first_dataset', _d_path)
    assert _spec2 and _spec2.loader
    _mod2 = _ilu2.module_from_spec(_spec2)
    _spec2.loader.exec_module(_mod2)  # type: ignore
    AffordanceFirstPairsDataset = getattr(_mod2, 'AffordanceFirstPairsDataset')
    AffordanceFirstPairsMultiDataset = getattr(_mod2, 'AffordanceFirstPairsMultiDataset')
    AffordanceFirstAggregatedDataset = getattr(_mod2, 'AffordanceFirstAggregatedDataset')
    AffordanceFirstAggregatedMultiDataset = getattr(_mod2, 'AffordanceFirstAggregatedMultiDataset')


def set_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def compute_loss(logits: torch.Tensor, target: torch.Tensor, *, loss_type: str = 'l1', w_l1_alpha: float = 9.0, bce_pos_weight: float = 5.0) -> torch.Tensor:
    # logits: (B,1,N); target: (B,N)
    if loss_type == 'l1':
        pred = torch.sigmoid(logits.squeeze(1))
        return torch.nn.functional.l1_loss(pred, target)
    elif loss_type == 'w_l1':
        pred = torch.sigmoid(logits.squeeze(1))
        w = 1.0 + float(w_l1_alpha) * target
        return torch.mean(w * torch.abs(pred - target))
    elif loss_type == 'bce':
        pos_w = torch.tensor(float(bce_pos_weight), dtype=logits.dtype, device=logits.device)
        return torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(1), target, pos_weight=pos_w)
    else:
        raise ValueError("loss_type must be one of {'l1','w_l1','bce'}")


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    loss_type: str,
    w_l1_alpha: float,
    bce_pos_weight: float,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    count = 0
    pbar = tqdm(loader, desc="Train", unit="batch")
    for pts, centers, target in pbar:
        pts = pts.to(device)
        centers = centers.to(device)
        target = target.to(device)

        optimizer.zero_grad(set_to_none=True)
        # AffordanceFirstPointNet2SSG expects (B,N,3) input only, no centers
        logits = model(pts)  # (B,1,N)
        pred = torch.sigmoid(logits.squeeze(1))
        if pred.shape != target.shape:
            raise RuntimeError(f"Shape mismatch before loss: pred={tuple(pred.shape)} target={tuple(target.shape)}")
        loss = compute_loss(logits, target, loss_type=loss_type, w_l1_alpha=w_l1_alpha, bce_pos_weight=bce_pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item()) * pts.shape[0]
        count += pts.shape[0]
        pbar.set_postfix(avg_loss=(total_loss / max(1, count)))
    return total_loss / max(1, count), count


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    loss_type: str,
    w_l1_alpha: float,
    bce_pos_weight: float,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    count = 0
    pbar = tqdm(loader, desc="Val", unit="batch")
    for pts, centers, target in pbar:
        pts = pts.to(device)
        centers = centers.to(device)
        target = target.to(device)
        logits = model(pts)
        pred = torch.sigmoid(logits.squeeze(1))
        if pred.shape != target.shape:
            raise RuntimeError(f"[eval] Shape mismatch before loss: pred={tuple(pred.shape)} target={tuple(target.shape)}")
        loss = compute_loss(logits, target, loss_type=loss_type, w_l1_alpha=w_l1_alpha, bce_pos_weight=bce_pos_weight)
        total_loss += float(loss.item()) * pts.shape[0]
        count += pts.shape[0]
        pbar.set_postfix(avg_loss=(total_loss / max(1, count)))
    return total_loss / max(1, count), count


def main():
    # Parse config path only to load defaults
    base_parser = argparse.ArgumentParser(add_help=False)
    default_cfg_path = os.path.join(os.path.dirname(__file__), 'config.json')
    base_parser.add_argument('--config', type=str, default=default_cfg_path)
    cfg_args, _ = base_parser.parse_known_args()

    # Load defaults from JSON config if present
    cfg = {}
    if cfg_args.config and os.path.isfile(cfg_args.config):
        try:
            with open(cfg_args.config, 'r') as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}

    parser = argparse.ArgumentParser(parents=[base_parser])

    # Data args
    parser.add_argument('--data_path', type=str, required=False, help='Single aff_first_pairs_left.npy')
    parser.add_argument('--data_dir', type=str, required=False, help='Directory with subfolders containing aff_first_pairs_left.npy')

    # Training args
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='./checkpoints_first')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # Loss args
    parser.add_argument('--loss_type', type=str, default='l1', choices=['l1', 'w_l1', 'bce'])
    parser.add_argument('--w_l1_alpha', type=float, default=9.0)
    parser.add_argument('--bce_pos_weight', type=float, default=5.0)

    # Dataset split/augmentation controls
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--augment_rotate_z', action='store_true', default=True)
    parser.add_argument('--augment_jitter_std', type=float, default=0.001)
    parser.add_argument('--shuffle_points', action='store_true', default=True)

    # Overfit mode for debugging
    parser.add_argument('--overfit_single', action='store_true')
    parser.add_argument('--pair_idx', type=int, default=0)

    # Dataset schema overrides
    parser.add_argument('--center_key', type=str, default=None, help='Override center key in pair dict')
    parser.add_argument('--target_key', type=str, default=None, help='Override target key matching number of points')
    # Aggregation options
    parser.add_argument('--aggregate_targets', action='store_true', help='Aggregate all pair targets per object into a single multi-modal target')
    parser.add_argument('--aggregate_mode', type=str, default='sum', choices=['sum', 'max'])
    parser.add_argument('--aggregate_norm', type=str, default='max', choices=['max', 'l1', 'none'])

    # Apply config defaults AFTER defining all args
    if isinstance(cfg, dict) and len(cfg) > 0:
        parser.set_defaults(**cfg)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = set_device(args.device)

    # Build datasets (first-hand datasets include centers but model ignores centers)
    tr_aug_rot = False if args.overfit_single else bool(args.augment_rotate_z)
    tr_aug_jit = 0.0 if args.overfit_single else float(args.augment_jitter_std)
    tr_shuffle = False if args.overfit_single else bool(args.shuffle_points)

    # Prefer data_path if provided; otherwise use data_dir
    if args.data_path and not args.aggregate_targets:
        dset_train = AffordanceFirstPairsDataset(
            npy_path=args.data_path,
            split='train',
            val_ratio=float(args.val_ratio),
            augment_rotate_z=tr_aug_rot,
            augment_jitter_std=tr_aug_jit,
            shuffle_points=tr_shuffle,
            only_pair_idx=(args.pair_idx if args.overfit_single else None),
            center_key_override=args.center_key,
            target_key_override=args.target_key,
        )
        dset_val = AffordanceFirstPairsDataset(
            npy_path=args.data_path,
            split='val',
            val_ratio=float(args.val_ratio),
            augment_rotate_z=False,
            augment_jitter_std=0.0,
            shuffle_points=False,
            only_pair_idx=(args.pair_idx if args.overfit_single else None),
            center_key_override=args.center_key,
            target_key_override=args.target_key,
        )
    elif args.data_path and args.aggregate_targets:
        # Single object aggregated into one sample for train/val (use the same aggregated sample but different splits not meaningful).
        # We mirror the standard interface by returning a dataset of length 1.
        dset_train = AffordanceFirstAggregatedDataset(
            npy_path=args.data_path,
            center_key=args.center_key,
            target_key=args.target_key,
            aggregate_mode=args.aggregate_mode,
            normalize=args.aggregate_norm,
            augment_rotate_z=tr_aug_rot,
            augment_jitter_std=tr_aug_jit,
            shuffle_points=tr_shuffle,
        )
        dset_val = AffordanceFirstAggregatedDataset(
            npy_path=args.data_path,
            center_key=args.center_key,
            target_key=args.target_key,
            aggregate_mode=args.aggregate_mode,
            normalize=args.aggregate_norm,
            augment_rotate_z=False,
            augment_jitter_std=0.0,
            shuffle_points=False,
        )
    elif args.data_dir and not args.aggregate_targets:
        dset_train = AffordanceFirstPairsMultiDataset(
            data_dir=args.data_dir,
            split='train',
            val_ratio=float(args.val_ratio),
            augment_rotate_z=tr_aug_rot,
            augment_jitter_std=tr_aug_jit,
            shuffle_points=tr_shuffle,
            only_pair_idx=(args.pair_idx if args.overfit_single else None),
            center_key_override=args.center_key,
            target_key_override=args.target_key,
        )
        dset_val = AffordanceFirstPairsMultiDataset(
            data_dir=args.data_dir,
            split='val',
            val_ratio=float(args.val_ratio),
            augment_rotate_z=False,
            augment_jitter_std=0.0,
            shuffle_points=False,
            only_pair_idx=(args.pair_idx if args.overfit_single else None),
            center_key_override=args.center_key,
            target_key_override=args.target_key,
        )
    elif args.data_dir and args.aggregate_targets:
        dset_train = AffordanceFirstAggregatedMultiDataset(
            data_dir=args.data_dir,
            split='train',
            center_key=args.center_key,
            target_key=args.target_key,
            aggregate_mode=args.aggregate_mode,
            normalize=args.aggregate_norm,
            augment_rotate_z=tr_aug_rot,
            augment_jitter_std=tr_aug_jit,
            shuffle_points=tr_shuffle,
        )
        dset_val = AffordanceFirstAggregatedMultiDataset(
            data_dir=args.data_dir,
            split='val',
            center_key=args.center_key,
            target_key=args.target_key,
            aggregate_mode=args.aggregate_mode,
            normalize=args.aggregate_norm,
            augment_rotate_z=False,
            augment_jitter_std=0.0,
            shuffle_points=False,
        )
    else:
        raise ValueError("Please provide either --data_path or --data_dir.")

    loader_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    loader_val = DataLoader(dset_val, batch_size=max(1, args.batch_size // 2), shuffle=False, num_workers=0, pin_memory=True)

    print(f"Dataset ready: train={len(dset_train)} samples, val={len(dset_val)} samples", flush=True)

    # Build model
    model = AffordanceFirstPointNet2SSG(use_xyz=True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if int(args.max_epochs) <= 0:
        print("Max epochs <= 0; exiting after dataset/model initialization (dry-run mode).", flush=True)
        return

    best_val = float('inf')
    best_path = os.path.join(args.output_dir, 'best_model.pth')
    print("Starting training...", flush=True)
    for epoch in range(1, args.max_epochs + 1):
        print(f"Epoch {epoch:03d} started", flush=True)
        train_loss, _ = train_one_epoch(
            model, loader_train, optimizer, device,
            loss_type=args.loss_type, w_l1_alpha=args.w_l1_alpha, bce_pos_weight=args.bce_pos_weight,
        )
        val_loss, _ = evaluate(
            model, loader_val, device,
            loss_type=args.loss_type, w_l1_alpha=args.w_l1_alpha, bce_pos_weight=args.bce_pos_weight,
        )

        print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_loss': best_val, 'cfg': vars(args)}, best_path)

        if (epoch % max(1, args.save_interval)) == 0:
            ckpt_path = os.path.join(args.output_dir, f'model_epoch_{epoch}.pth')
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss, 'cfg': vars(args)}, ckpt_path)

    print(f"Best model saved to: {best_path} (val_loss={best_val:.6f})")


if __name__ == '__main__':
    main()


