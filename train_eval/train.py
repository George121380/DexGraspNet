#!/usr/bin/env python3
"""
SecAff Affordance Model Training Script
Final optimized version with all bug fixes applied

Usage:
    conda activate pn
    python train.py --data_path aff_sec_result/2_of_Jenga_Classic_Game/aff_sec_pairs.npy
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from models.affordance import SecAffModel, FocalMSELoss, AffordanceDataset


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train SecAff Affordance Model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to affordance training data (.npy file)')
    parser.add_argument('--max_points', type=int, default=1024,
                        help='Maximum number of points per point cloud (default: 1024)')
    
    # Model arguments
    parser.add_argument('--extractor_type', type=str, default='ssg', choices=['ssg', 'msg'],
                        help='Point cloud extractor type (default: ssg)')
    parser.add_argument('--pointcloud_feature_dim', type=int, default=512,
                        help='Point cloud feature dimension (default: 512)')
    parser.add_argument('--keypoint_encoder_dim', type=int, default=128,
                        help='Keypoint encoder dimension (default: 128)')
    parser.add_argument('--fusion_hidden_dim', type=int, default=256,
                        help='Fusion network hidden dimension (default: 256)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs (default: 100)')
    parser.add_argument('--focal_alpha', type=float, default=25.0,
                        help='Focal loss alpha parameter (default: 25.0) - increased for imbalanced data')
    parser.add_argument('--focal_gamma', type=float, default=3.0,
                        help='Focal loss gamma parameter (default: 3.0) - increased for imbalanced data')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping (default: 1.0)')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable data augmentation (default: False)')
    parser.add_argument('--overfit_n', type=int, default=0,
                        help='Overfit to N samples for sanity check (0 to disable)')
    parser.add_argument('--use_mse', action='store_true',
                        help='Use plain MSE loss instead of FocalMSE (default: False)')
    parser.add_argument('--aux_cls', action='store_true',
                        help='Enable auxiliary classification loss on high-value points (default: False)')
    parser.add_argument('--cls_threshold', type=float, default=0.5,
                        help='Threshold for classification target (default: 0.5)')
    parser.add_argument('--cls_pos_weight', type=float, default=10.0,
                        help='Positive class weight for BCEWithLogitsLoss (default: 10.0)')
    parser.add_argument('--lambda_cls', type=float, default=1.0,
                        help='Weight for auxiliary classification loss (default: 1.0)')
    parser.add_argument('--dynq_cls', action='store_true',
                        help='Use dynamic quantile threshold for auxiliary classification (per-batch)')
    parser.add_argument('--dynq', type=float, default=0.9,
                        help='Quantile for dynamic threshold when --dynq_cls is enabled (default: 0.9)')
    parser.add_argument('--topk_reg', type=int, default=0,
                        help='If >0, upweight regression loss on top-k GT points per sample')
    parser.add_argument('--topk_reg_weight', type=float, default=3.0,
                        help='Weight multiplier for top-k regression loss')
    parser.add_argument('--margin_high_thr', type=float, default=0.5,
                        help='Target threshold considered high for margin loss (default: 0.5)')
    parser.add_argument('--margin_value', type=float, default=0.5,
                        help='Desired minimum prediction for high targets (default: 0.5)')
    parser.add_argument('--lambda_margin', type=float, default=0.3,
                        help='Weight for margin loss term (default: 0.3)')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints (default: ./checkpoints)')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save model every N epochs (default: 10)')
    parser.add_argument('--no_early_stop', action='store_true',
                        help='Disable early stopping (default: False)')
    parser.add_argument('--pred_minmax_norm', action='store_true',
                        help='Per-sample min-max normalize predictions during train/val (default: False)')
    
    return parser.parse_args()


def create_dataloaders(args):
    """Create training and validation dataloaders"""
    from torch.utils.data import Subset
    
    train_dataset = AffordanceDataset(
        data_path=args.data_path,
        max_points=args.max_points,
        augment=(not args.no_augment)
    )
    
    val_dataset = AffordanceDataset(
        data_path=args.data_path,
        max_points=args.max_points,
        augment=False
    )
    
    if args.overfit_n and args.overfit_n > 0:
        n = min(args.overfit_n, len(train_dataset))
        indices = list(range(n))
        train_dataset = Subset(train_dataset, indices)
        val_dataset = Subset(val_dataset, indices)
    
    effective_batch_size = max(2, args.batch_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def train_epoch(model, dataloader, optimizer, criterion_reg, device, max_grad_norm=1.0, aux_cfg=None, pred_minmax_norm: bool = False):
    """Train for one epoch with gradient clipping"""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_cls = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        pointcloud = batch['pointcloud'].to(device)
        keypoints = batch['keypoints'].to(device)
        target_scores = batch['affordance_scores'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred_scores = model(pointcloud, keypoints)
        if pred_minmax_norm:
            min_v = pred_scores.min(dim=1, keepdim=True)[0]
            max_v = pred_scores.max(dim=1, keepdim=True)[0]
            pred_scores = (pred_scores - min_v) / (max_v - min_v + 1e-6)
        
        # Compute loss
        loss = criterion_reg(pred_scores, target_scores)
        # Optional: top-k regression upweighting
        if aux_cfg is not None and int(aux_cfg.get('topk_reg', 0)) > 0:
            k = int(aux_cfg['topk_reg'])
            w = float(aux_cfg.get('topk_reg_weight', 3.0))
            B, N = target_scores.shape
            k_eff = max(1, min(k, N))
            topk_vals, topk_idx = torch.topk(target_scores, k=k_eff, dim=1)
            batch_idx = torch.arange(B, device=device).unsqueeze(-1).expand(-1, k_eff)
            pred_topk = pred_scores[batch_idx, topk_idx]
            tgt_topk = target_scores[batch_idx, topk_idx]
            loss_topk = F.mse_loss(pred_topk, tgt_topk)
            loss = loss + (w - 1.0) * loss_topk
        cls_loss_val = 0.0
        if aux_cfg is not None and aux_cfg.get('enabled', False):
            logits = getattr(model, 'last_raw_scores', None)
            if logits is None:
                logits = torch.logit(torch.clamp(pred_scores, 1e-6, 1-1e-6))
            with torch.no_grad():
                if aux_cfg.get('dynq_cls', False):
                    # per-batch dynamic quantile thresholding on targets
                    q = float(aux_cfg.get('dynq', 0.9))
                    thr = torch.quantile(target_scores.view(-1), q).item()
                    cls_targets = (target_scores > thr).float()
                else:
                    cls_targets = (target_scores > aux_cfg['threshold']).float()
            bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(aux_cfg['pos_weight'], device=device))
            cls_loss = bce(logits, cls_targets)
            loss = loss + aux_cfg['lambda_cls'] * cls_loss
            cls_loss_val = cls_loss.item()

        # Optional: margin loss to push high targets above margin_value
        if aux_cfg is not None and float(aux_cfg.get('lambda_margin', 0.0)) > 0.0:
            margin_thr = float(aux_cfg.get('margin_high_thr', 0.5))
            desired = float(aux_cfg.get('margin_value', 0.5))
            lambda_m = float(aux_cfg['lambda_margin'])
            high_mask = (target_scores > margin_thr).float()
            if high_mask.any():
                # hinge-like on predictions: penalize if pred < desired
                margin_term = F.relu(desired - pred_scores) * high_mask
                loss = loss + lambda_m * margin_term.mean()
        
        # Backward pass with gradient clipping
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            mse = F.mse_loss(pred_scores, target_scores)
            mae = F.l1_loss(pred_scores, target_scores)
        
        total_loss += loss.item()
        total_mse += mse.item()
        total_mae += mae.item()
        total_cls += cls_loss_val
        num_batches += 1
        
        if batch_idx % 5 == 0:
            extra = f", Cls={cls_loss_val:.4f}" if aux_cfg is not None and aux_cfg.get('enabled', False) else ""
            print(f'  Batch {batch_idx:2d}/{len(dataloader)}: '
                  f'Loss={loss.item():.4f}, MSE={mse.item():.4f}, MAE={mae.item():.4f}{extra}, '
                  f'Pred=[{pred_scores.min():.3f},{pred_scores.max():.3f}], '
                  f'GradNorm={grad_norm:.2f}')
    
    return {
        'loss': total_loss / num_batches,
        'mse': total_mse / num_batches,
        'mae': total_mae / num_batches,
        'cls': (total_cls / num_batches) if (aux_cfg is not None and aux_cfg.get('enabled', False)) else None
    }


def validate_epoch(model, dataloader, device, pred_minmax_norm: bool = False):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            pointcloud = batch['pointcloud'].to(device)
            keypoints = batch['keypoints'].to(device)
            target_scores = batch['affordance_scores'].to(device)
            
            pred_scores = model(pointcloud, keypoints)
            if pred_minmax_norm:
                min_v = pred_scores.min(dim=1, keepdim=True)[0]
                max_v = pred_scores.max(dim=1, keepdim=True)[0]
                pred_scores = (pred_scores - min_v) / (max_v - min_v + 1e-6)
            
            # Use standard MSE for validation (not focal)
            loss = F.mse_loss(pred_scores, target_scores)
            mse = F.mse_loss(pred_scores, target_scores)
            mae = F.l1_loss(pred_scores, target_scores)
            
            total_loss += loss.item()
            total_mse += mse.item()
            total_mae += mae.item()
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'mse': total_mse / num_batches,
        'mae': total_mae / num_batches
    }


def main():
    """Main training function"""
    args = parse_args()
    
    print("=" * 60)
    print("SecAff Affordance Model Training")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Model: {args.extractor_type.upper()}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Focal loss: α={args.focal_alpha}, γ={args.focal_gamma}")
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Create dataloader
    print("\nCreating dataloader...")
    train_loader, val_loader = create_dataloaders(args)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Batches per epoch: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = SecAffModel(
        pointcloud_extractor_type=args.extractor_type,
        pointcloud_feature_dim=args.pointcloud_feature_dim,
        keypoint_encoder_dim=args.keypoint_encoder_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        num_points=args.max_points
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer and loss function
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Choose regression loss function
    if args.use_mse:
        criterion_reg = nn.MSELoss()
        print("Using plain MSE loss for training")
    else:
        # Use Focal MSE Loss to handle extreme data imbalance
        criterion_reg = FocalMSELoss(alpha=max(args.focal_alpha, 20.0), gamma=max(args.focal_gamma, 3.0))

    # Aux classification loss config
    aux_cfg = None
    if args.aux_cls:
        aux_cfg = {
            'enabled': True,
            'threshold': float(args.cls_threshold),
            'pos_weight': float(args.cls_pos_weight),
            'lambda_cls': float(args.lambda_cls),
        }
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=8, verbose=True, min_lr=1e-6
    )
    
    # Training loop
    print("\nStarting training...")
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(args.max_epochs):
        print(f"\nEpoch {epoch+1}/{args.max_epochs}")
        print("-" * 50)
        
        # Train
        start_time = time.time()
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion_reg, device, args.max_grad_norm, aux_cfg, pred_minmax_norm=bool(getattr(args, 'pred_minmax_norm', False))
        )
        train_time = time.time() - start_time
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device, pred_minmax_norm=bool(getattr(args, 'pred_minmax_norm', False)))
        
        # Update scheduler
        scheduler.step(val_metrics['loss'])
        
        # Print metrics
        cls_str = f", CLS: {train_metrics['cls']:.6f}" if train_metrics.get('cls') is not None else ""
        print(f"Train - Loss: {train_metrics['loss']:.6f}, "
              f"MSE: {train_metrics['mse']:.6f}, MAE: {train_metrics['mae']:.6f}{cls_str}")
        print(f"Val   - Loss: {val_metrics['loss']:.6f}, "
              f"MSE: {val_metrics['mse']:.6f}, MAE: {val_metrics['mae']:.6f}")
        print(f"Time: {train_time:.2f}s")
        
        # Save best model
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            patience_counter = 0
            
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f"✅ New best model saved: {checkpoint_path}")
        else:
            patience_counter += 1
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            periodic_path = os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_metrics['loss'],
                'args': vars(args)
            }, periodic_path)
            print(f"📁 Periodic checkpoint saved: {periodic_path}")
        
        # Early stopping
        if not getattr(args, 'no_early_stop', False) and patience_counter >= max_patience:
            print(f"\n⏹️  Early stopping after {epoch+1} epochs (patience: {max_patience})")
            break
    
    print(f"\n🎉 Training completed!")
    print(f"Best validation loss: {best_loss:.6f}")
    print(f"Best model saved at: {os.path.join(args.output_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()

