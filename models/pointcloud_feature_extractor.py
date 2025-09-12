"""
Point cloud feature extractors based on PointNet2
Author: Assistant
"""

import sys
import os
from typing import Optional, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add PointNet2 to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../third_party/Pointnet2_PyTorch'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../third_party/Pointnet2_PyTorch/pointnet2_ops_lib'))

try:
    from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG
except ImportError:
    raise ImportError("Failed to import PointNet2 modules. Please ensure PointNet2_PyTorch is properly installed.")


class PointCloudFeatureExtractor(nn.Module):
    """
    Base class for point cloud feature extractors
    """
    def __init__(self):
        super(PointCloudFeatureExtractor, self).__init__()
        
    def _break_up_pc(self, pointcloud: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Break up point cloud into xyz coordinates and features
        
        Args:
            pointcloud: (B, N, 3+C) tensor where C is the number of input features
            
        Returns:
            xyz: (B, N, 3) coordinates
            features: (B, C, N) features or None if no features
        """
        xyz = pointcloud[..., 0:3].contiguous()
        features = pointcloud[..., 3:].transpose(1, 2).contiguous() if pointcloud.size(-1) > 3 else None
        return xyz, features
        
    def forward(self, pointcloud: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - to be implemented by subclasses
        
        Args:
            pointcloud: (B, N, 3+C) input point cloud
            
        Returns:
            features: (B, feature_dim) extracted features
        """
        raise NotImplementedError("Subclasses must implement forward method")


class PointNet2SSGExtractor(PointCloudFeatureExtractor):
    """
    PointNet2 Single-Scale Grouping (SSG) feature extractor
    """
    def __init__(
        self, 
        input_channels: int = 0,
        use_xyz: bool = True,
        output_dim: Optional[int] = None,
        sa_configs: Optional[List[dict]] = None
    ):
        """
        Initialize PointNet2 SSG feature extractor
        
        Args:
            input_channels: Number of input feature channels (excluding xyz)
            use_xyz: Whether to use xyz coordinates as features
            output_dim: Output feature dimension. If None, uses the last SA module output
            sa_configs: List of SA module configurations. If None, uses default configuration
        """
        super(PointNet2SSGExtractor, self).__init__()
        
        self.use_xyz = use_xyz
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.sa1_out_dim = None  # will be inferred from config
        
        # Default SA module configurations
        if sa_configs is None:
            # Note: PointNet2 modules automatically add 3 to the first layer when use_xyz=True
            # So we only specify the input feature channels, not xyz
            sa_configs = [
                {
                    'npoint': 512,
                    'radius': 0.2,
                    'nsample': 64,
                    'mlp': [input_channels, 64, 64, 128]
                },
                {
                    'npoint': 128,
                    'radius': 0.4,
                    'nsample': 64,
                    'mlp': [128, 128, 128, 256]
                },
                {
                    'npoint': None,  # Global pooling
                    'mlp': [256, 256, 512, 1024]
                }
            ]
        
        # Build SA modules
        self.SA_modules = nn.ModuleList()
        for i, config in enumerate(sa_configs):
            if config.get('npoint') is None:
                # Global pooling layer
                self.SA_modules.append(
                    PointnetSAModule(
                        mlp=config['mlp'],
                        use_xyz=self.use_xyz
                    )
                )
            else:
                self.SA_modules.append(
                    PointnetSAModule(
                        npoint=config['npoint'],
                        radius=config['radius'],
                        nsample=config['nsample'],
                        mlp=config['mlp'],
                        use_xyz=self.use_xyz
                    )
                )
            # Record SA1 output dim (last layer of first SA mlp)
            if i == 0:
                self.sa1_out_dim = config['mlp'][-1]
        
        # Optional output projection layer
        if self.output_dim is not None:
            final_dim = sa_configs[-1]['mlp'][-1]
            self.output_projection = nn.Sequential(
                nn.Linear(final_dim, self.output_dim),
                nn.BatchNorm1d(self.output_dim),
                nn.ReLU(True)
            )
        else:
            self.output_projection = None
            
    def forward(self, pointcloud: torch.Tensor):
        """
        Forward pass through PointNet2 SSG
        
        Args:
            pointcloud: (B, N, 3+C) input point cloud
            
        Returns:
            global_features: (B, feature_dim)
            intermediates: dict with keys 'sa1_xyz', 'sa1_features', 'sa2_xyz', 'sa2_features'
        """
        xyz, features = self._break_up_pc(pointcloud)
        
        # Pass through SA modules
        intermediates = {}
        for i, module in enumerate(self.SA_modules):
            xyz, features = module(xyz, features)
            if i == 0:
                intermediates['sa1_xyz'] = xyz  # (B, 512, 3)
                intermediates['sa1_features'] = features  # (B, C1, 512)
            if i == 1:
                intermediates['sa2_xyz'] = xyz  # (B, 128, 3)
                intermediates['sa2_features'] = features  # (B, C2, 128)
            
        # Global features should be (B, C, 1) after final SA module
        global_features = features.squeeze(-1)  # (B, C)
        
        # Optional output projection
        if self.output_projection is not None:
            global_features = self.output_projection(global_features)
            
        return global_features, intermediates


class PointNet2MSGExtractor(PointCloudFeatureExtractor):
    """
    PointNet2 Multi-Scale Grouping (MSG) feature extractor
    """
    def __init__(
        self,
        input_channels: int = 0,
        use_xyz: bool = True,
        output_dim: Optional[int] = None,
        sa_configs: Optional[List[dict]] = None
    ):
        """
        Initialize PointNet2 MSG feature extractor
        
        Args:
            input_channels: Number of input feature channels (excluding xyz)
            use_xyz: Whether to use xyz coordinates as features
            output_dim: Output feature dimension. If None, uses the last SA module output
            sa_configs: List of SA module configurations. If None, uses default configuration
        """
        super(PointNet2MSGExtractor, self).__init__()
        
        self.use_xyz = use_xyz
        self.input_channels = input_channels
        self.output_dim = output_dim
        # For MSG, SA1 output channels are sum of last dims for each scale
        self.sa1_out_dim = None
        
        # Default MSG SA module configurations
        if sa_configs is None:
            # Note: PointNet2 modules automatically add 3 to the first layer when use_xyz=True
            # So we only specify the input feature channels, not xyz
            sa_configs = [
                {
                    'npoint': 512,
                    'radii': [0.1, 0.2, 0.4],
                    'nsamples': [16, 32, 128],
                    'mlps': [
                        [input_channels, 32, 32, 64],
                        [input_channels, 64, 64, 128], 
                        [input_channels, 64, 96, 128]
                    ]
                },
                {
                    'npoint': 128,
                    'radii': [0.2, 0.4, 0.8],
                    'nsamples': [32, 64, 128],
                    'mlps': [
                        [64 + 128 + 128, 64, 64, 128],
                        [64 + 128 + 128, 128, 128, 256],
                        [64 + 128 + 128, 128, 128, 256]
                    ]
                },
                {
                    'npoint': None,  # Global pooling
                    'mlp': [128 + 256 + 256, 256, 512, 1024]
                }
            ]
        
        # Build SA modules
        self.SA_modules = nn.ModuleList()
        for i, config in enumerate(sa_configs):
            if config.get('npoint') is None:
                # Global pooling layer (regular SA module)
                self.SA_modules.append(
                    PointnetSAModule(
                        mlp=config['mlp'],
                        use_xyz=self.use_xyz
                    )
                )
            else:
                # Multi-scale SA module
                self.SA_modules.append(
                    PointnetSAModuleMSG(
                        npoint=config['npoint'],
                        radii=config['radii'],
                        nsamples=config['nsamples'],
                        mlps=config['mlps'],
                        use_xyz=self.use_xyz
                    )
                )
            if i == 0:
                if 'mlps' in config:
                    self.sa1_out_dim = sum(mlp[-1] for mlp in config['mlps'])
                else:
                    self.sa1_out_dim = config['mlp'][-1]
        
        # Optional output projection layer
        if self.output_dim is not None:
            final_dim = sa_configs[-1]['mlp'][-1]
            self.output_projection = nn.Sequential(
                nn.Linear(final_dim, self.output_dim),
                nn.BatchNorm1d(self.output_dim),
                nn.ReLU(True)
            )
        else:
            self.output_projection = None
            
    def forward(self, pointcloud: torch.Tensor):
        """
        Forward pass through PointNet2 MSG
        
        Args:
            pointcloud: (B, N, 3+C) input point cloud
            
        Returns:
            global_features: (B, feature_dim)
            intermediates: dict with keys 'sa1_xyz', 'sa1_features', 'sa2_xyz', 'sa2_features'
        """
        xyz, features = self._break_up_pc(pointcloud)
        
        # Pass through SA modules
        intermediates = {}
        for i, module in enumerate(self.SA_modules):
            xyz, features = module(xyz, features)
            if i == 0:
                intermediates['sa1_xyz'] = xyz
                intermediates['sa1_features'] = features
            if i == 1:
                intermediates['sa2_xyz'] = xyz
                intermediates['sa2_features'] = features
            
        # Global features should be (B, C, 1) after final SA module
        global_features = features.squeeze(-1)  # (B, C)
        
        # Optional output projection
        if self.output_projection is not None:
            global_features = self.output_projection(global_features)
            
        return global_features, intermediates


# Factory function for easy instantiation
def create_pointnet2_extractor(
    extractor_type: str = 'ssg',
    input_channels: int = 0,
    use_xyz: bool = True,
    output_dim: Optional[int] = None,
    **kwargs
) -> PointCloudFeatureExtractor:
    """
    Factory function to create PointNet2 feature extractors
    
    Args:
        extractor_type: Type of extractor ('ssg' or 'msg')
        input_channels: Number of input feature channels (excluding xyz)
        use_xyz: Whether to use xyz coordinates as features
        output_dim: Output feature dimension
        **kwargs: Additional arguments passed to the extractor
        
    Returns:
        Feature extractor instance
    """
    if extractor_type.lower() == 'ssg':
        return PointNet2SSGExtractor(
            input_channels=input_channels,
            use_xyz=use_xyz,
            output_dim=output_dim,
            **kwargs
        )
    elif extractor_type.lower() == 'msg':
        return PointNet2MSGExtractor(
            input_channels=input_channels,
            use_xyz=use_xyz,
            output_dim=output_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}. Choose 'ssg' or 'msg'")


if __name__ == "__main__":
    # Simple test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test SSG extractor
    print("Testing PointNet2 SSG Extractor...")
    ssg_extractor = PointNet2SSGExtractor(input_channels=0, output_dim=512)
    ssg_extractor.to(device)
    
    # Test input: batch_size=2, num_points=1024, xyz only
    test_input = torch.randn(2, 1024, 3).to(device)
    ssg_output = ssg_extractor(test_input)
    print(f"SSG Input shape: {test_input.shape}")
    print(f"SSG Output shape: {ssg_output.shape}")
    
    # Test MSG extractor
    print("\nTesting PointNet2 MSG Extractor...")
    msg_extractor = PointNet2MSGExtractor(input_channels=0, output_dim=512)
    msg_extractor.to(device)
    
    msg_output = msg_extractor(test_input)
    print(f"MSG Input shape: {test_input.shape}")
    print(f"MSG Output shape: {msg_output.shape}")
    
    print("\nAll tests passed!")
