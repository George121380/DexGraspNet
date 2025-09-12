# Script Directory - Function Overview

This directory contains utility scripts for the BiDexHand affordance project pipeline.

## 📁 Script Functions

### 🔄 Data Preprocessing Scripts

#### `run_preprocess.sh`
**FUNCTION**: Single object preprocessing script
- Extracts point cloud surface and bimanual grasp pose pairs for one specific object
- Converts BimanGrasp dataset format to training-ready point cloud + grasp data

**Usage**:
```bash
bash script/run_preprocess.sh [OBJECT_NAME] [NUM]
# Example: bash script/run_preprocess.sh 3D_Dollhouse_Sofa 0
```

#### `run_preprocess_batch.sh`
**FUNCTION**: Batch preprocessing script for BimanGrasp dataset
- Processes first N objects from BimanGrasp dataset, extracting point clouds and grasp pairs
- Prepares data for affordance training by sampling object surfaces and hand poses

**Usage**:
```bash
bash script/run_preprocess_batch.sh [TOP_K]
# Example: bash script/run_preprocess_batch.sh 5
```

#### `run_aff_sec_batch.sh`
**FUNCTION**: Batch processing script for generating affordance training data
- Processes multiple objects to create left-hand keypoints and right-hand affordance scores
- Outputs training pairs for SecAff model training

**Usage**:
```bash
bash script/run_aff_sec_batch.sh [IN_ROOT] [OUT_ROOT] [SIGMA]
# Example: bash script/run_aff_sec_batch.sh
```

### 🎨 Visualization Scripts

#### `run_affordance_vis.sh`
**FUNCTION**: Affordance visualization script
- Generates interactive 3D HTML visualizations showing bimanual grasp affordance
- Creates dual-hand affordance heatmaps for specified objects

**Usage**:
```bash
bash script/run_affordance_vis.sh [OBJECT_NAME] [NUM]
# Example: bash script/run_affordance_vis.sh 3D_Dollhouse_Sofa 0
```

### 🤖 Actor Model Scripts

#### `run_train_actor.sh`
**FUNCTION**: Basic actor model training script
- Trains the actor model on standard preprocessed grasp data without affordance conditioning
- Uses distance-based target mode for bimanual grasp prediction

**Usage**:
```bash
bash script/run_train_actor.sh [DEVICE] [BATCH] [EPOCHS]
# Example: bash script/run_train_actor.sh cuda 2 20
```

#### `run_train_actor_aff_sec.sh`
**FUNCTION**: Actor model training script using affordance-based data
- Trains the actor model on preprocessed aff_sec dataset with dual affordance features
- Uses frozen backbones and affordance conditioning for bimanual grasp learning

**Usage**:
```bash
bash script/run_train_actor_aff_sec.sh [DEVICE] [BATCH] [EPOCHS]
# Example: bash script/run_train_actor_aff_sec.sh cuda 2 20
```

#### `run_eval_actor_aff_sec.sh`
**FUNCTION**: Actor model evaluation script for affordance-based data
- Evaluates trained actor model performance on aff_sec preprocessed dataset
- Tests model's ability to predict affordance scores from point clouds

**Usage**:
```bash
bash script/run_eval_actor_aff_sec.sh [CKPT_PATH] [DEVICE]
# Example: bash script/run_eval_actor_aff_sec.sh outputs/actor.pt cuda
```

#### `run_visualize_actor_aff_sec.sh`
**FUNCTION**: Actor model prediction visualization script
- Visualizes trained actor model predictions vs ground truth on aff_sec data
- Generates comparison plots showing model performance and prediction quality

**Usage**:
```bash
bash script/run_visualize_actor_aff_sec.sh [CKPT_PATH] [DEVICE]
# Example: bash script/run_visualize_actor_aff_sec.sh outputs/actor.pt cpu
```

### 🧪 Testing Scripts

#### `test_pointcloud_pointnet.py`
**FUNCTION**: PointNet feature extraction testing script
- Tests PointNet backbone on point cloud data for feature extraction validation
- Verifies model inference speed, output shapes, and numerical stability

**Usage**:
```bash
python script/test_pointcloud_pointnet.py --path data.npy --device cuda
```

## 🔗 Script Dependencies

### Data Flow Pipeline
```
BimanGrasp Dataset 
    ↓ (run_preprocess_batch.sh)
Point Clouds + Grasp Pairs
    ↓ (run_aff_sec_batch.sh)  
Affordance Training Data
    ↓ (../train.py)
Trained SecAff Model
    ↓ (../evaluate.py)
Evaluation Results + Visualizations
```

### Actor Model Pipeline
```
Preprocessed Data
    ↓ (run_train_actor*.sh)
Trained Actor Model
    ↓ (run_eval_actor*.sh)
Performance Metrics
    ↓ (run_visualize_actor*.sh)
Prediction Visualizations
```

## 🎯 Usage Notes

- All scripts assume specific conda environment (`affordance` or `pn`)
- Paths are configured for the original project structure
- Most scripts include default parameters for quick testing
- Use `--help` flag where available for detailed parameter information

---
*Updated: 2024年9月10日*  
*Purpose: Pipeline automation and workflow management*
