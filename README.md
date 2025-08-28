# Affordance-BiDex

Bimanual Dexterous Hand Affordance Learning on BimanGrasp-Dataset

## Overview

This project implements affordance learning for bimanual dexterous hand grasping using the BimanGrasp-Dataset. The goal is to learn visual affordances that guide dual-hand robot grasping strategies.

## Features

- **Data Preprocessing**: Efficient processing of BimanGrasp-Dataset for affordance learning
- **3D Visualization**: Interactive 3D point cloud visualization with affordance heatmaps
- **PointNet2 Integration**: GPU-accelerated point cloud processing
- **Dual-Hand Affordance**: Separate affordance prediction for left and right hands
- **Modular Design**: Clean code structure with separate preprocessing and training modules

## Project Structure

```
affordance-bidex/
├── pre_process/                    # Data preprocessing module
│   ├── preprocess_bimangrasp_data.py
│   ├── visualize_data_3d.py
│   └── README.md
├── models/                         # Model implementations (to be added)
├── data/                          # Data loaders (to be added)
├── configs/                       # Configuration files (to be added)
├── utils/                         # Utility functions (to be added)
├── train_affordance.py            # Training script (to be added)
├── evaluate_affordance.py         # Evaluation script (to be added)
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Installation

### Prerequisites

- Python 3.7+
- CUDA 10.2+ (for GPU support)
- PyTorch 1.5.1+

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/George121380/affordance-bidex.git
   cd affordance-bidex
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install PointNet2 (GPU version):**
   ```bash
   cd third_party/Pointnet2_PyTorch/pointnet2_ops_lib
   python setup.py install
   ```

## Usage

### Data Preprocessing

1. **Prepare BimanGrasp-Dataset:**
   - Download BimanGrasp-Dataset and place in `third_party/BimanGrasp-Dataset/`
   - Extract the dataset files

2. **Run preprocessing:**
   ```bash
   cd pre_process
   python preprocess_bimangrasp_data.py --data_root ../third_party/BimanGrasp-Dataset --output_dir ../processed_data --num_points 1024
   ```

### Visualization

**Interactive 3D visualization with auto-open:**
```bash
python visualize_data_3d.py --data_dir ../processed_data --num_samples 3 --auto_open
```

**Basic visualization (save HTML only):**
```bash
python visualize_data_3d.py --data_dir ../processed_data --num_samples 3
```

## Data Format

Each processed sample contains:
- **Point Cloud**: 1024 points with 3D coordinates (x, y, z)
- **Left Affordance**: 1024 scores for left hand grasping
- **Right Affordance**: 1024 scores for right hand grasping
- **Object Name**: Original object identifier

## Visualization Features

- **3D Point Cloud Display**: Interactive 3D scatter plots
- **Affordance Heatmaps**: Color-coded affordance scores (red=left, blue=right)
- **Hover Information**: Point details on mouse hover
- **Statistics**: Distribution histograms for affordance scores
- **Auto-Open**: Automatic browser opening for immediate viewing

## Development Status

- ✅ **Stage 1**: Environment setup and data preprocessing
- 🔄 **Stage 2**: Model architecture implementation (in progress)
- ⏳ **Stage 3**: Training script development
- ⏳ **Stage 4**: Training and optimization
- ⏳ **Stage 5**: Evaluation and analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{affordance-bidex,
  title={Affordance-BiDex: Bimanual Dexterous Hand Affordance Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/George121380/affordance-bidex}
}
```

## Acknowledgments

- BimanGrasp-Dataset for providing the bimanual grasping dataset
- PointNet2 authors for the point cloud processing framework
- DualAfford for inspiration on affordance learning approaches
