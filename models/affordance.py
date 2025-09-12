"""
Affordance Prediction Model for BiDexHand
Predicts affordance scores for each point in a point cloud given keypoint features and global point cloud features.

Author: Assistant
"""

import sys
import os
from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
# Removed pytorch_lightning dependency for better compatibility

# Add PointNet2 to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../third_party/Pointnet2_PyTorch'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../third_party/Pointnet2_PyTorch/pointnet2_ops_lib'))

try:
    from .pointcloud_feature_extractor import PointNet2SSGExtractor, PointNet2MSGExtractor
except ImportError:
    from pointcloud_feature_extractor import PointNet2SSGExtractor, PointNet2MSGExtractor


# Legacy classes removed - functionality integrated into SecAffModel


class FocalMSELoss(nn.Module):
    """
    Focal MSE Loss to handle imbalanced affordance regression
    Focuses more on high-value affordance points
    """
    def __init__(self, alpha=5.0, gamma=2.0):
        super(FocalMSELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        mse = (pred - target) ** 2
        # Weight higher for high-value targets
        weights = 1.0 + self.alpha * (target ** self.gamma)
        weighted_mse = weights * mse
        return weighted_mse.mean()


class SecAffModel(nn.Module):
    """
    Complete SecAff model for affordance prediction
    Optimized architecture with LayerNorm and proper initialization
    """
    def __init__(
        self,
        pointcloud_extractor_type: str = 'ssg',
        pointcloud_feature_dim: int = 512,
        keypoint_encoder_dim: int = 128,
        fusion_hidden_dim: int = 256,
        num_points: int = 8192
    ):
        super(SecAffModel, self).__init__()
        self.last_raw_scores: Optional[torch.Tensor] = None
        
        # Point cloud feature extractor
        if pointcloud_extractor_type.lower() == 'ssg':
            self.pointcloud_extractor = PointNet2SSGExtractor(
                input_channels=0,
                output_dim=pointcloud_feature_dim,
                use_xyz=True
            )
        elif pointcloud_extractor_type.lower() == 'msg':
            self.pointcloud_extractor = PointNet2MSGExtractor(
                input_channels=0,
                output_dim=pointcloud_feature_dim,
                use_xyz=True
            )
        else:
            raise ValueError(f"Unknown extractor type: {pointcloud_extractor_type}")
        
        # Improved keypoint encoder with LayerNorm (uses coords + sampled SA1 features)
        keypoint_input_dim = 3 + int(getattr(self.pointcloud_extractor, 'sa1_out_dim', 0) or 0)
        self.keypoint_encoder = nn.Sequential(
            nn.Linear(keypoint_input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(True),
            nn.Linear(64, keypoint_encoder_dim),
            nn.LayerNorm(keypoint_encoder_dim),
            nn.ReLU(True)
        )
        
        # Improved fusion network
        fusion_input_dim = keypoint_encoder_dim + pointcloud_feature_dim
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.LayerNorm(fusion_hidden_dim // 2),
            nn.ReLU(True)
        )
        # Sample-level logit bias to adjust global activation per sample
        self.logit_bias_head = nn.Linear(fusion_hidden_dim // 2, 1)
        # Sample-level logit scale to expand/shrink activation range per sample
        self.logit_scale_head = nn.Linear(fusion_hidden_dim // 2, 1)
        
        # Per-point affordance prediction head
        # Takes each point's (x,y,z) + interpolated SA1 features + broadcasted fused global features
        self.per_point_feat_dim = int(getattr(self.pointcloud_extractor, 'sa1_out_dim', 0) or 0)
        # Do not include point-to-keypoint distance stats (min/mean); not helpful for single keypoint
        self.include_point_keypoint_dists = False
        point_keypoint_extra = 0
        point_head_input_dim = (fusion_hidden_dim // 2) + 3 + self.per_point_feat_dim + point_keypoint_extra
        self.point_head = nn.Sequential(
            nn.Linear(point_head_input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )
        
        # Initialize weights for stable training
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights to handle extreme data imbalance"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # More aggressive initialization for better learning of rare high-value points
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)  # Small positive bias for stability
    
    def encode_keypoints(self, keypoints: torch.Tensor) -> torch.Tensor:
        """Encode keypoint coordinates to features"""
        B, N_kps, _ = keypoints.shape
        keypoints_flat = keypoints.view(-1, keypoints.size(-1))
        features_flat = self.keypoint_encoder(keypoints_flat)
        features = features_flat.view(B, N_kps, -1)
        return features
        
    def forward(
        self, 
        pointcloud: torch.Tensor, 
        keypoints: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            pointcloud: (B, N, 3) point cloud coordinates
            keypoints: (B, N_kps, 3) keypoint coordinates
            
        Returns:
            affordance_scores: (B, num_points) predicted affordance scores
        """
        # Extract global + intermediate point cloud features
        extractor_out = self.pointcloud_extractor(pointcloud)
        if isinstance(extractor_out, tuple) and len(extractor_out) == 2:
            global_features, intermediates = extractor_out
        else:
            # Backward compatibility if extractor returns only global features
            global_features, intermediates = extractor_out, {}
        
        # Sample SA1 features at keypoint locations (nearest neighbor on SA1 xyz)
        sa1_xyz = intermediates.get('sa1_xyz', None)
        sa1_features = intermediates.get('sa1_features', None)  # (B, C1, n1)
        if sa1_xyz is not None and sa1_features is not None and keypoints.size(-1) == 3:
            # Compute nearest SA1 point for each keypoint
            # keypoints: (B, K, 3), sa1_xyz: (B, n1, 3)
            dists = torch.cdist(keypoints, sa1_xyz)  # (B, K, n1)
            nn_idx = torch.argmin(dists, dim=2)  # (B, K)
            B, K = nn_idx.shape
            C1 = sa1_features.size(1)
            idx_expanded = nn_idx.unsqueeze(1).expand(-1, C1, -1)  # (B, C1, K)
            sampled_kp_feat = torch.gather(sa1_features, 2, idx_expanded)  # (B, C1, K)
            sampled_kp_feat = sampled_kp_feat.permute(0, 2, 1).contiguous()  # (B, K, C1)
            keypoint_inputs = torch.cat([keypoints, sampled_kp_feat], dim=2)  # (B, K, 3+C1)
        else:
            keypoint_inputs = keypoints
        
        # Encode keypoint features
        keypoint_features = self.encode_keypoints(keypoint_inputs)
        
        # Average keypoint features
        keypoint_features_avg = keypoint_features.mean(dim=1)
        
        # Concatenate features
        fused_features = torch.cat([keypoint_features_avg, global_features], dim=1)
        
        # Pass through fusion network
        fused_features = self.fusion_layers(fused_features)
        
        # Per-point features via SA1 interpolation if available (3-NN inverse distance)
        sa1_xyz = intermediates.get('sa1_xyz', None)
        sa1_features = intermediates.get('sa1_features', None)  # (B, C1, n1)
        B, N, _ = pointcloud.shape
        per_point_feats = None
        if sa1_xyz is not None and sa1_features is not None:
            dists = torch.cdist(pointcloud, sa1_xyz) + 1e-8  # (B, N, n1)
            K = min(3, sa1_xyz.size(1))
            knn_dists, knn_idx = torch.topk(dists, k=K, largest=False, dim=2)  # (B, N, K)
            weights = 1.0 / knn_dists
            weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)
            C1 = sa1_features.size(1)
            sa1_f = sa1_features.permute(0, 2, 1)  # (B, n1, C1)
            sa1_f_exp = sa1_f.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, n1, C1)
            idx_exp = knn_idx.unsqueeze(-1).expand(-1, -1, -1, C1)  # (B, N, K, C1)
            gathered = torch.gather(sa1_f_exp, 2, idx_exp)  # (B, N, K, C1)
            per_point_feats = (gathered * weights.unsqueeze(-1)).sum(dim=2)  # (B, N, C1)

        # Per-point prediction: concatenate point coords (+ optional feats) with broadcasted fused features
        fused_broadcast = fused_features.unsqueeze(1).expand(-1, N, -1)  # (B, N, C)
        if per_point_feats is not None:
            parts = [pointcloud, per_point_feats, fused_broadcast]
        else:
            parts = [pointcloud, fused_broadcast]
        per_point_input = torch.cat(parts, dim=-1)
        per_point_input = per_point_input.reshape(B * N, -1)
        raw_scores = self.point_head(per_point_input).reshape(B, N)  # (B, N)
        # Add sample-level logit bias and scale
        logit_bias = self.logit_bias_head(fused_features).squeeze(-1)  # (B,)
        logit_scale = F.softplus(self.logit_scale_head(fused_features)).squeeze(-1) + 1e-3  # (B,)
        raw_scores = raw_scores * logit_scale.unsqueeze(1) + logit_bias.unsqueeze(1)
        self.last_raw_scores = raw_scores
        
        # Apply sigmoid to map to [0,1]
        affordance_scores = torch.sigmoid(raw_scores)
        
        return affordance_scores


class AffordanceDataset(Dataset):
    """
    Dataset for affordance prediction training
    """
    def __init__(
        self, 
        data_path: str,
        max_points: int = 8192,
        augment: bool = True
    ):
        super(AffordanceDataset, self).__init__()
        
        self.data_path = data_path
        self.max_points = max_points
        self.augment = augment
        
        # Load data
        self.data = np.load(data_path, allow_pickle=True).item()
        self.pointcloud = self.data['points']  # (N, 3)
        self.pairs = self.data['pairs']  # Dict of pairs
        
        # Convert pairs to list for indexing
        self.pair_list = []
        for pair_id, pair_data in self.pairs.items():
            # Extract keypoints and affordance scores
            left_kps = pair_data.get('left_kps', np.array([]).reshape(0, 3))
            right_kps = pair_data.get('right_kps', np.array([]).reshape(0, 3))
            aff_scores_left = pair_data.get('aff_scores_left', np.array([]))
            aff_scores_right = pair_data.get('aff_scores_right', np.array([]))
            
            # Add left hand data if available
            if len(left_kps) > 0 and len(aff_scores_left) > 0:
                self.pair_list.append({
                    'keypoints': left_kps,
                    'affordance_scores': aff_scores_left,
                    'hand': 'left'
                })
            
            # Add right hand data if available (most common case)
            if len(left_kps) > 0 and len(aff_scores_right) > 0:
                self.pair_list.append({
                    'keypoints': left_kps,  # Use left keypoints for right affordance
                    'affordance_scores': aff_scores_right,
                    'hand': 'right'
                })
            
            # Also add right keypoints if they exist
            if len(right_kps) > 0 and len(aff_scores_right) > 0:
                self.pair_list.append({
                    'keypoints': right_kps,
                    'affordance_scores': aff_scores_right,
                    'hand': 'right'
                })
        
        # Precompute deterministic subsampling indices per sample for stability
        self._total_points = len(self.pointcloud)
        self.fixed_indices = []
        for i in range(len(self.pair_list)):
            rs = np.random.RandomState(i + 12345)
            replace = self._total_points < self.max_points
            idx = rs.choice(self._total_points, self.max_points, replace=replace)
            self.fixed_indices.append(idx)
        
        print(f"Loaded {len(self.pair_list)} training samples from {data_path}")
        
    def __len__(self) -> int:
        return len(self.pair_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pair_list[idx]
        
        # Get point cloud and apply deterministic subsampling/upsampling
        indices = self.fixed_indices[idx]
        pointcloud = self.pointcloud[indices]  # (max_points, 3)
        affordance_full = pair['affordance_scores']
        # If affordance_full shorter than total points, pad to total to index safely
        if len(affordance_full) < self._total_points:
            pad_len = self._total_points - len(affordance_full)
            affordance_full = np.concatenate([affordance_full, np.zeros(pad_len)])
        affordance_scores = affordance_full[indices]
        
        # Get keypoints
        keypoints = pair['keypoints'].copy()  # (N_kps, 3)
        
        # Data augmentation
        if self.augment:
            # Random rotation around Z axis
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            pointcloud = pointcloud @ rotation_matrix.T
            keypoints = keypoints @ rotation_matrix.T
            
            # Random translation
            translation = np.random.uniform(-0.02, 0.02, 3)
            pointcloud += translation
            keypoints += translation
            
            # Random scaling
            scale = np.random.uniform(0.9, 1.1)
            pointcloud *= scale
            keypoints *= scale
        
        # Convert to tensors
        sample = {
            'pointcloud': torch.from_numpy(pointcloud).float(),
            'keypoints': torch.from_numpy(keypoints).float(),
            'affordance_scores': torch.from_numpy(affordance_scores).float(),
            'hand': pair['hand']
        }
        
        return sample


def create_dataloader(
    data_path: str,
    batch_size: int = 8,
    max_points: int = 8192,
    augment: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create DataLoader for affordance training
    """
    dataset = AffordanceDataset(
        data_path=data_path,
        max_points=max_points,
        augment=augment
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


if __name__ == "__main__":
    # Test the model
    print("Testing SecAff Model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = SecAffModel(
        pointcloud_extractor_type='ssg',
        pointcloud_feature_dim=256,
        keypoint_encoder_dim=64,
        num_points=1024
    ).to(device)
    
    # Test data
    batch_size = 2
    num_points = 1024
    num_keypoints = 3
    
    pointcloud = torch.randn(batch_size, num_points, 3).to(device)
    keypoints = torch.randn(batch_size, num_keypoints, 3).to(device)
    
    # Forward pass
    with torch.no_grad():
        affordance_scores = model(pointcloud, keypoints)
        
    print(f"Input pointcloud shape: {pointcloud.shape}")
    print(f"Input keypoints shape: {keypoints.shape}")
    print(f"Output affordance scores shape: {affordance_scores.shape}")
    print(f"Affordance score range: [{affordance_scores.min():.4f}, {affordance_scores.max():.4f}]")
    
    # Test dataset loading
    print("\nTesting dataset loading...")
    try:
        dataset = AffordanceDataset(
            'aff_sec_result/2_of_Jenga_Classic_Game/aff_sec_pairs.npy',
            max_points=1024
        )
        print(f"Dataset size: {len(dataset)}")
        
        sample = dataset[0]
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
            else:
                print(f"{key}: {value}")
                
    except Exception as e:
        print(f"Dataset test failed: {e}")
    
    print("\n✅ SecAff model test completed!")
