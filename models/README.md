# Point Cloud Feature Extractors

This module provides PointNet2-based point cloud feature extractors for the BiDexHand affordance project.

## Features

- **PointNet2 SSG (Single-Scale Grouping)**: Standard PointNet2 with single-scale grouping
- **PointNet2 MSG (Multi-Scale Grouping)**: PointNet2 with multi-scale grouping for better feature capture
- **Flexible Configuration**: Customizable SA module configurations and output dimensions
- **CUDA Support**: Optimized for GPU acceleration

## Quick Start

### Basic Usage

```python
import torch
from models.pointcloud_feature_extractor import PointNet2SSGExtractor, PointNet2MSGExtractor

# Create test point cloud data (batch_size=4, num_points=1024, xyz coordinates)
pointcloud = torch.randn(4, 1024, 3)

# Initialize SSG extractor
ssg_extractor = PointNet2SSGExtractor(
    input_channels=0,    # Only xyz coordinates
    output_dim=256,      # Output feature dimension
    use_xyz=True         # Use xyz as features
)

# Extract features
features = ssg_extractor(pointcloud)  # Shape: (4, 256)
```

### With Additional Features

```python
# Point cloud with RGB features (xyz + rgb)
pointcloud_with_rgb = torch.randn(4, 1024, 6)  # 3 xyz + 3 rgb

# Initialize extractor with additional input channels
extractor = PointNet2SSGExtractor(
    input_channels=3,    # RGB features
    output_dim=512
)

features = extractor(pointcloud_with_rgb)  # Shape: (4, 512)
```

### Using Factory Function

```python
from models.pointcloud_feature_extractor import create_pointnet2_extractor

# Create SSG extractor
ssg_extractor = create_pointnet2_extractor(
    extractor_type='ssg',
    input_channels=0,
    output_dim=256
)

# Create MSG extractor
msg_extractor = create_pointnet2_extractor(
    extractor_type='msg',
    input_channels=0,
    output_dim=256
)
```

### Custom SA Module Configuration

```python
# Custom SA module configuration for SSG
custom_sa_configs = [
    {
        'npoint': 256,
        'radius': 0.1,
        'nsample': 32,
        'mlp': [0, 32, 32, 64]  # input_channels will be added automatically
    },
    {
        'npoint': 64,
        'radius': 0.2,
        'nsample': 32,
        'mlp': [64, 64, 128]
    },
    {
        'npoint': None,  # Global pooling
        'mlp': [128, 128, 256]
    }
]

extractor = PointNet2SSGExtractor(
    input_channels=0,
    sa_configs=custom_sa_configs,
    output_dim=128
)
```

## API Reference

### PointNet2SSGExtractor

Single-Scale Grouping PointNet2 feature extractor.

**Parameters:**
- `input_channels` (int): Number of input feature channels (excluding xyz)
- `use_xyz` (bool): Whether to use xyz coordinates as features
- `output_dim` (Optional[int]): Output feature dimension. If None, uses the last SA module output
- `sa_configs` (Optional[List[dict]]): Custom SA module configurations

### PointNet2MSGExtractor

Multi-Scale Grouping PointNet2 feature extractor.

**Parameters:**
- `input_channels` (int): Number of input feature channels (excluding xyz)
- `use_xyz` (bool): Whether to use xyz coordinates as features
- `output_dim` (Optional[int]): Output feature dimension. If None, uses the last SA module output
- `sa_configs` (Optional[List[dict]]): Custom SA module configurations

### create_pointnet2_extractor

Factory function to create PointNet2 feature extractors.

**Parameters:**
- `extractor_type` (str): Type of extractor ('ssg' or 'msg')
- `input_channels` (int): Number of input feature channels (excluding xyz)
- `use_xyz` (bool): Whether to use xyz coordinates as features
- `output_dim` (Optional[int]): Output feature dimension
- `**kwargs`: Additional arguments passed to the extractor

## Testing

Run the test script to verify functionality:

```bash
conda activate pn  # or your PointNet2 environment
python test_pointcloud_extractor.py
```

The test script includes:
- Basic functionality tests
- Tests with additional input features
- Factory function tests
- Different point count tests

## Requirements

- PyTorch
- PointNet2_PyTorch (included in third_party)
- CUDA (optional, for GPU acceleration)

## Notes

- The extractors automatically handle the addition of xyz coordinates to the first layer when `use_xyz=True`
- For batch size = 1, make sure to set the model to eval mode to avoid BatchNorm issues
- The default configurations are optimized for common use cases but can be customized as needed
