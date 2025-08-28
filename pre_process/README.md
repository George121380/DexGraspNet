# Pre-processing Module

This module contains scripts for preprocessing BimanGrasp dataset and visualizing the results.

## Files

### `preprocess_bimangrasp_data.py`
Data preprocessing script that converts BimanGrasp dataset into training-ready format.

**Usage:**
```bash
python preprocess_bimangrasp_data.py --data_root third_party/BimanGrasp-Dataset --output_dir processed_data --num_points 1024
```

**Arguments:**
- `--data_root`: Path to BimanGrasp dataset root directory
- `--output_dir`: Output directory for processed data
- `--num_points`: Number of points to sample from each mesh (default: 1024)

**Output:**
- `processed_data/train/`: Training samples
- `processed_data/val/`: Validation samples  
- `processed_data/test/`: Test samples
- `processed_data/dataset_info.json`: Dataset statistics

### `visualize_data_3d.py`
3D point cloud visualization script using plotly.

**Usage:**
```bash
# Basic visualization (saves HTML files only)
python visualize_data_3d.py --data_dir processed_data --num_samples 3

# Auto-open in browser (uses plotly's built-in show() method)
python visualize_data_3d.py --data_dir processed_data --num_samples 3 --auto_open
```

**Arguments:**
- `--data_dir`: Directory containing processed data (default: processed_data)
- `--num_samples`: Number of samples to visualize (default: 3)
- `--split`: Data split to visualize (train/val/test, default: train)
- `--auto_open`: Automatically open visualization in browser

**Output:**
- `point_cloud_visualization.html`: 3D point cloud visualization
- `affordance_statistics.html`: Affordance score distribution

## Workflow

1. **Preprocess data:**
   ```bash
   cd pre_process
   python preprocess_bimangrasp_data.py
   ```

2. **Visualize results:**
   ```bash
   python visualize_data_3d.py --auto_open
   ```

## Data Format

Each processed sample contains:
- `point_cloud`: 3D point coordinates (N, 3)
- `left_affordance`: Left hand affordance scores (N,)
- `right_affordance`: Right hand affordance scores (N,)
- `object_name`: Original object name

## Dependencies

- torch
- numpy
- trimesh
- plotly
- scikit-learn
- tqdm
