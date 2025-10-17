import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

try:
    # PointNet++ ops (should be installed from third_party/Pointnet2_PyTorch)
    from pointnet2_ops import pointnet2_utils
    from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
except Exception as e:
    raise ImportError(
        "Failed to import pointnet2_ops. Please install third_party/Pointnet2_PyTorch/pointnet2_ops_lib."  # noqa: E501
    ) from e


class AffordancePointNet2SSG(nn.Module):
    """
    PointNet++ SSG-based network for per-point affordance prediction with optional conditioning.

    Input conventions:
      - If called with a single tensor (pointcloud), it must be of shape (B, N, 3 + C),
        where the first 3 channels are xyz, and the next C are extra per-point features.
      - Alternatively, you can provide (points_xyz: (B, N, 3)) and (centers_xyz: (B, 3)).
        In this case, conditioning features will be built according to condition_mode.

    Conditioning features:
      - condition_mode = 'none' -> C = 0 (no extra features)
      - condition_mode = 'r'    -> C = 3 (relative vector r = p - c)
      - condition_mode = 'rd'   -> C = 4 (r plus distance d = ||p - c||)

    This module reuses PointNet++ building blocks directly and mirrors the standard
    semantic segmentation decoder (FP) so that each original input point receives a feature
    and a per-point prediction.
    """

    def __init__(
        self,
        use_condition: bool = True,
        condition_mode: str = "rd",  # {'none','r','rd'}
        use_xyz: bool = True,
    ) -> None:
        super().__init__()

        self.use_condition = bool(use_condition)
        if condition_mode not in {"none", "r", "rd", "kp"}:
            raise ValueError("condition_mode must be one of {'none','r','rd','kp'}")
        self.condition_mode = condition_mode
        self.use_xyz = bool(use_xyz)

        # Determine feature channels C from conditioning mode
        self.extra_feat_channels = self._compute_extra_channels(use_condition, condition_mode)

        self.use_center_feature = bool(self.use_condition and self.condition_mode == "kp")
        if self.use_center_feature and self.extra_feat_channels != 0:
            raise ValueError("Keypoint feature mode should not use extra per-point channels.")

        # Build SA (encoder) and FP (decoder) stacks following pointnet2_ssg_sem style
        self.SA_modules = nn.ModuleList()
        self.sa_out_channels = []
        # SA1: input channels for MLP must include xyz if use_xyz=True
        # IMPORTANT: PointnetSAModule internally concatenates xyz to features when use_xyz=True.
        # Therefore, the first MLP input channel should be ONLY the extra feature channels (C),
        # and the module will add +3 internally if use_xyz=True.
        sa1_in_channels = self.extra_feat_channels
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[sa1_in_channels, 32, 32, 64],
                use_xyz=self.use_xyz,
            )
        )
        self.sa_out_channels.append(64)
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=self.use_xyz,
            )
        )
        self.sa_out_channels.append(128)
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=self.use_xyz,
            )
        )
        self.sa_out_channels.append(256)
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=self.use_xyz,
            )
        )
        self.sa_out_channels.append(512)

        # FP stack mirrors pointnet2_ssg_sem.py
        self.FP_modules = nn.ModuleList()
        # FP1 skip uses the original input feature channels (C), NOT including xyz.
        fp1_skip_channels = self.extra_feat_channels
        self.FP_modules.append(PointnetFPModule(mlp=[128 + fp1_skip_channels, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        # Per-point prediction head (logits)
        self.point_feature_channels = 128
        head_in_channels = self.point_feature_channels

        if self.use_center_feature:
            center_total_channels = sum(self.sa_out_channels) + self.point_feature_channels
            self.center_out_channels = 128
            self.center_fusion = nn.Sequential(
                nn.Conv1d(center_total_channels, 256, kernel_size=1, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Conv1d(256, self.center_out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(self.center_out_channels),
                nn.ReLU(inplace=True),
            )
            head_in_channels += self.center_out_channels

        self.head = nn.Sequential(
            nn.Conv1d(head_in_channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1, kernel_size=1),
        )

    @staticmethod
    def _compute_extra_channels(use_condition: bool, condition_mode: str) -> int:
        if not use_condition or condition_mode == "none":
            return 0
        if condition_mode == "r":
            return 3
        if condition_mode == "rd":
            return 4
        if condition_mode == "kp":
            return 0
        # Should not reach here due to validation
        return 0

    @staticmethod
    def _break_up_pc(pc: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Split a (B, N, 3 + C) pointcloud into xyz and features.

        Returns:
            xyz: (B, N, 3)
            features: (B, C, N) or None
        """
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].contiguous()
        if features.numel() == 0:
            features = None
        else:
            features = features.transpose(1, 2).contiguous()
        return xyz, features

    @staticmethod
    def _compute_condition_features(points_xyz: torch.Tensor, centers_xyz: torch.Tensor, mode: str) -> Optional[torch.Tensor]:
        """Compute per-point conditioning features based on centers.

        Args:
            points_xyz: (B, N, 3)
            centers_xyz: (B, 3)
            mode: 'none' | 'r' | 'rd'

        Returns:
            features: (B, N, C) or None
        """
        if mode == "none" or mode == "kp":
            return None
        # Compute r = p - c
        r = points_xyz - centers_xyz.unsqueeze(1)  # (B, N, 3)
        if mode == "r":
            return r
        # mode == 'rd'
        d = torch.norm(r, dim=-1, keepdim=True)  # (B, N, 1)
        return torch.cat([r, d], dim=-1)  # (B, N, 4)

    def _forward_with_pointcloud(
        self,
        pointcloud: torch.Tensor,
        centers_xyz: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward given a single (B, N, 3 + C) tensor as in the library examples."""
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz = [xyz]
        l_features: List[Optional[torch.Tensor]] = [features]
        sa_features_at_center: List[torch.Tensor] = [] if self.use_center_feature else []
        center_xyz: Optional[torch.Tensor] = None
        if self.use_center_feature:
            if centers_xyz is None:
                raise ValueError("centers_xyz must be provided when condition_mode='kp'.")
            center_xyz = centers_xyz.unsqueeze(1)

        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

            if self.use_center_feature and center_xyz is not None:
                assert li_xyz is not None and li_features is not None
                dist, idx = pointnet2_utils.three_nn(center_xyz, li_xyz)
                weight = 1.0 / (dist + 1e-8)
                weight = weight / torch.sum(weight, dim=-1, keepdim=True)
                center_feat_cur = pointnet2_utils.three_interpolate(li_features, idx, weight)
                sa_features_at_center.append(center_feat_cur)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        point_level_features = l_features[0]

        if self.use_center_feature:
            assert center_xyz is not None
            dist_low, idx_low = pointnet2_utils.three_nn(center_xyz, l_xyz[0])
            weight_low = 1.0 / (dist_low + 1e-8)
            weight_low = weight_low / torch.sum(weight_low, dim=-1, keepdim=True)
            center_feat_low = pointnet2_utils.three_interpolate(point_level_features, idx_low, weight_low)
            center_features = torch.cat(sa_features_at_center + [center_feat_low], dim=1)
            center_features = self.center_fusion(center_features)
            center_features = center_features.mean(dim=-1, keepdim=True)
            center_features = center_features.expand(-1, -1, point_level_features.shape[-1])
            point_level_features = torch.cat([point_level_features, center_features], dim=1)

        logits = self.head(point_level_features)  # (B, 1, N)
        return logits

    def forward(
        self,
        points_or_pointcloud: torch.Tensor,
        centers_xyz: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward API supporting two calling patterns:
          1) points_or_pointcloud is (B, N, 3 + C) (xyz then features). centers_xyz must be None.
          2) points_or_pointcloud is (B, N, 3) (xyz only) and centers_xyz is (B, 3).

        Returns:
            logits: (B, 1, N)
        """
        if points_or_pointcloud.dim() != 3:
            raise ValueError("Input must be of shape (B, N, 3) or (B, N, 3 + C)")

        B, N, D = points_or_pointcloud.shape
        if D > 3 and centers_xyz is None:
            # Already packed as (B, N, 3 + C)
            return self._forward_with_pointcloud(points_or_pointcloud)

        if D != 3:
            raise ValueError("When centers are provided, input must have shape (B, N, 3)")
        if centers_xyz is None:
            # No condition; treat as 'none'
            cond_mode = "none"
        else:
            cond_mode = self.condition_mode if self.use_condition else "none"

        # Build conditioning features according to the configured mode
        extra = self._compute_condition_features(points_or_pointcloud, centers_xyz, cond_mode)
        if extra is None:
            pointcloud = points_or_pointcloud
        else:
            pointcloud = torch.cat([points_or_pointcloud, extra], dim=-1)

        if cond_mode == "kp":
            return self._forward_with_pointcloud(pointcloud, centers_xyz=centers_xyz)
        return self._forward_with_pointcloud(pointcloud)


__all__ = ["AffordancePointNet2SSG"]


