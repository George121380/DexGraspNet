import os
import argparse
import json
import sys
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Disable cuDNN to avoid CUDNN_STATUS_EXECUTION_FAILED on older PyTorch builds
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:
    def tqdm(x, **kwargs):  # type: ignore
        return x

# Ensure repo root is on sys.path when running from train_eval/
CUR_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CUR_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Robust imports with fallback to direct file loading
try:
    from models.affordance_second import AffordancePointNet2SSG  # type: ignore
except Exception:
    import importlib.util as _ilu
    _m_path = os.path.abspath(os.path.join(REPO_ROOT, 'models', 'affordance_second.py'))
    _spec = _ilu.spec_from_file_location('affordance_second', _m_path)
    assert _spec and _spec.loader
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)  # type: ignore
    AffordancePointNet2SSG = getattr(_mod, 'AffordancePointNet2SSG')

try:
    from data.affordance_dataset import AffordancePairsDataset, AffordancePairsMultiDataset  # type: ignore
except Exception:
    import importlib.util as _ilu2
    _d_path = os.path.abspath(os.path.join(REPO_ROOT, 'data', 'affordance_dataset.py'))
    _spec2 = _ilu2.spec_from_file_location('affordance_dataset', _d_path)
    assert _spec2 and _spec2.loader
    _mod2 = _ilu2.module_from_spec(_spec2)
    _spec2.loader.exec_module(_mod2)  # type: ignore
    AffordancePairsDataset = getattr(_mod2, 'AffordancePairsDataset')
    AffordancePairsMultiDataset = getattr(_mod2, 'AffordancePairsMultiDataset')


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
        # BCEWithLogits with optional positive class weighting
        pos_w = torch.tensor(float(bce_pos_weight), dtype=logits.dtype, device=logits.device)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(1), target, pos_weight=pos_w)
        return bce
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
        logits = model(pts, centers)  # (B,1,N)
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
        logits = model(pts, centers)
        loss = compute_loss(logits, target, loss_type=loss_type, w_l1_alpha=w_l1_alpha, bce_pos_weight=bce_pos_weight)
        total_loss += float(loss.item()) * pts.shape[0]
        count += pts.shape[0]
        pbar.set_postfix(avg_loss=(total_loss / max(1, count)))
    return total_loss / max(1, count), count


def main():
    # First, parse config path only to load defaults
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

    parser.add_argument('--data_path', type=str, required=False, help='Single aff_sec_pairs.npy')
    parser.add_argument('--data_dir', type=str, required=False, help='Directory containing subfolders with aff_sec_pairs.npy')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--focal_alpha', type=float, default=25.0)
    parser.add_argument('--focal_gamma', type=float, default=3.0)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # Conditioning toggles
    parser.add_argument('--use_condition', action='store_true', default=True)
    parser.add_argument('--condition_mode', type=str, default='rd', choices=['none', 'r', 'rd', 'kp'])
    parser.add_argument('--loss_type', type=str, default='l1', choices=['l1', 'w_l1', 'bce'])
    parser.add_argument('--w_l1_alpha', type=float, default=9.0)
    parser.add_argument('--bce_pos_weight', type=float, default=5.0)

    # Overfit-on-one-sample mode
    parser.add_argument('--overfit_single', action='store_true', help='overfit on a single pair index')
    parser.add_argument('--pair_idx', type=int, default=0, help='pair index to overfit when --overfit_single is set')

    # Dataset split/augmentation controls
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--augment_rotate_z', action='store_true', default=True)
    parser.add_argument('--augment_jitter_std', type=float, default=0.001)
    parser.add_argument('--shuffle_points', action='store_true', default=True)

    # Apply config defaults AFTER defining all args so that
    # precedence becomes: CLI > config.json > code defaults
    if isinstance(cfg, dict) and len(cfg) > 0:
        parser.set_defaults(**cfg)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = set_device(args.device)

    # Build datasets
    # Dataset configs; disable augmentation for overfit-on-one-sample
    tr_aug_rot = False if args.overfit_single else bool(args.augment_rotate_z)
    tr_aug_jit = 0.0 if args.overfit_single else float(args.augment_jitter_std)
    tr_shuffle = False if args.overfit_single else bool(args.shuffle_points)

    if args.data_dir:
        dset_train = AffordancePairsMultiDataset(
            data_dir=args.data_dir,
            split='train',
            val_ratio=float(args.val_ratio),
            augment_rotate_z=tr_aug_rot,
            augment_jitter_std=tr_aug_jit,
            shuffle_points=tr_shuffle,
            only_pair_idx=(args.pair_idx if args.overfit_single else None),
        )
        dset_val = AffordancePairsMultiDataset(
            data_dir=args.data_dir,
            split='val',
            val_ratio=float(args.val_ratio),
            augment_rotate_z=False,
            augment_jitter_std=0.0,
            shuffle_points=False,
            only_pair_idx=(args.pair_idx if args.overfit_single else None),
        )
    else:
        dset_train = AffordancePairsDataset(
            npy_path=args.data_path,
            split='train',
            val_ratio=float(args.val_ratio),
            augment_rotate_z=tr_aug_rot,
            augment_jitter_std=tr_aug_jit,
            shuffle_points=tr_shuffle,
            only_pair_idx=(args.pair_idx if args.overfit_single else None),
        )
        dset_val = AffordancePairsDataset(
            npy_path=args.data_path,
            split='val',
            val_ratio=float(args.val_ratio),
            augment_rotate_z=False,
            augment_jitter_std=0.0,
            shuffle_points=False,
            only_pair_idx=(args.pair_idx if args.overfit_single else None),
        )

    loader_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    loader_val = DataLoader(dset_val, batch_size=max(1, args.batch_size // 2), shuffle=False, num_workers=0, pin_memory=True)

    print(f"Dataset ready: train={len(dset_train)} samples, val={len(dset_val)} samples", flush=True)

    # Build model
    model = AffordancePointNet2SSG(use_condition=args.use_condition, condition_mode=args.condition_mode, use_xyz=True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

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






