#!/bin/bash
# =============================================================================
# SecAff Affordance Model Training Script
# 
# This script provides convenient training configurations for the SecAff model
# with various parameter settings and use cases.
# =============================================================================

set -e  # Exit on any error

# Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Resolve important paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Ensure Python can import top-level packages (e.g., models)
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Function to check if conda environment is activated
check_environment() {
    if [[ "$CONDA_DEFAULT_ENV" != "pn" ]]; then
        print_warning "Conda environment 'pn' is not activated"
        print_info "Please run: conda activate pn"
        exit 1
    fi
    print_success "Conda environment 'pn' is active"
}

# Function to check if required data exists
check_data() {
    local data_path="$1"
    if [[ ! -f "$data_path" ]]; then
        print_error "Data file not found: $data_path"
        print_info "Please ensure the data file exists or update the path"
        exit 1
    fi
    print_success "Data file found: $data_path"
}

# Function to display help
show_help() {
    cat << EOF
=============================================================================
SecAff Affordance Model Training Script
=============================================================================

USAGE:
    ./train.sh [CONFIGURATION] [OPTIONS]

CONFIGURATIONS:
    quick       - Quick training for testing (10 epochs, small batch)
    standard    - Standard training configuration (100 epochs)
    robust      - Robust training with higher focal loss parameters
    custom      - Use custom parameters (requires manual specification)

OPTIONS:
    --data_path PATH        Path to training data (.npy file)
    --output_dir PATH       Output directory for checkpoints
    --batch_size N          Batch size (default: 4)
    --max_epochs N          Maximum epochs (default: 100)
    --learning_rate FLOAT   Learning rate (default: 1e-3)
    --device DEVICE         Device to use (auto/cuda/cpu)
    --help                  Show this help message

EXAMPLES:
    # Quick test training
    ./train.sh quick --data_path aff_sec_result/sample/aff_sec_pairs.npy

    # Standard training with custom data path
    ./train.sh standard --data_path aff_sec_result/2_of_Jenga_Classic_Game/aff_sec_pairs.npy

    # Robust training for difficult datasets
    ./train.sh robust --data_path aff_sec_result/complex_object/aff_sec_pairs.npy

    # Custom training with specific parameters
    ./train.sh custom --data_path data.npy --batch_size 8 --max_epochs 50 --learning_rate 5e-4

=============================================================================
EOF
}

# Default parameters
DEFAULT_DATA_PATH="aff_sec_result/2_of_Jenga_Classic_Game/aff_sec_pairs.npy"
DEFAULT_OUTPUT_DIR="./checkpoints"
DEFAULT_DEVICE="auto"

# Parse configuration
CONFIG="$1"
shift || true

# Initialize parameters based on configuration
case "$CONFIG" in
    "quick")
        print_info "Using QUICK training configuration"
        BATCH_SIZE=2
        MAX_EPOCHS=10
        LEARNING_RATE=1e-3
        FOCAL_ALPHA=15.0
        FOCAL_GAMMA=2.0
        SAVE_INTERVAL=5
        ;;
    "standard")
        print_info "Using STANDARD training configuration"
        BATCH_SIZE=4
        MAX_EPOCHS=100
        LEARNING_RATE=1e-3
        FOCAL_ALPHA=25.0
        FOCAL_GAMMA=3.0
        SAVE_INTERVAL=10
        ;;
    "robust")
        print_info "Using ROBUST training configuration"
        BATCH_SIZE=4
        MAX_EPOCHS=150
        LEARNING_RATE=8e-4
        FOCAL_ALPHA=35.0
        FOCAL_GAMMA=4.0
        SAVE_INTERVAL=15
        ;;
    "custom")
        print_info "Using CUSTOM training configuration"
        BATCH_SIZE=4
        MAX_EPOCHS=100
        LEARNING_RATE=1e-3
        FOCAL_ALPHA=25.0
        FOCAL_GAMMA=3.0
        SAVE_INTERVAL=10
        ;;
    "--help" | "-h" | "help")
        show_help
        exit 0
        ;;
    "")
        print_error "No configuration specified"
        show_help
        exit 1
        ;;
    *)
        print_error "Unknown configuration: $CONFIG"
        show_help
        exit 1
        ;;
esac

# Set default values
DATA_PATH="$DEFAULT_DATA_PATH"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
DEVICE="$DEFAULT_DEVICE"

# Parse additional options
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Pre-flight checks
print_info "Performing pre-flight checks..."
check_environment

# Normalize paths relative to repo root if not absolute
if [[ "$DATA_PATH" != /* ]]; then
    FULL_DATA_PATH="$REPO_ROOT/$DATA_PATH"
else
    FULL_DATA_PATH="$DATA_PATH"
fi

if [[ "$OUTPUT_DIR" != /* ]]; then
    FULL_OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR"
else
    FULL_OUTPUT_DIR="$OUTPUT_DIR"
fi

check_data "$FULL_DATA_PATH"

# Create output directory
mkdir -p "$FULL_OUTPUT_DIR"
print_success "Output directory ready: $FULL_OUTPUT_DIR"

# Display training configuration
print_info "Training Configuration:"
echo "  Data Path:        $DATA_PATH"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Batch Size:       $BATCH_SIZE"
echo "  Max Epochs:       $MAX_EPOCHS"
echo "  Learning Rate:    $LEARNING_RATE"
echo "  Focal Alpha:      $FOCAL_ALPHA"
echo "  Focal Gamma:      $FOCAL_GAMMA"
echo "  Save Interval:    $SAVE_INTERVAL"
echo "  Device:           $DEVICE"

# Confirmation prompt (skip for quick config)
if [[ "$CONFIG" != "quick" ]]; then
    echo
    read -p "Proceed with training? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Training cancelled"
        exit 0
    fi
fi

# Start training
print_success "Starting training..."
echo "============================================================================="

python "$REPO_ROOT/train_eval/train.py" \
    --data_path "$FULL_DATA_PATH" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --max_epochs "$MAX_EPOCHS" \
    --focal_alpha "$FOCAL_ALPHA" \
    --focal_gamma "$FOCAL_GAMMA" \
    --output_dir "$FULL_OUTPUT_DIR" \
    --save_interval "$SAVE_INTERVAL" \
    --device "$DEVICE" \
    --max_grad_norm 1.0 \
    --weight_decay 1e-5

# Check training result
if [[ $? -eq 0 ]]; then
    print_success "Training completed successfully!"
    print_info "Best model saved at: $FULL_OUTPUT_DIR/best_model.pth"
    print_info "To evaluate the model, run:"
    echo "  $SCRIPT_DIR/eval.sh standard --model_path $FULL_OUTPUT_DIR/best_model.pth --data_path $FULL_DATA_PATH"
else
    print_error "Training failed!"
    exit 1
fi

# =============================================================================
# USAGE EXAMPLES:
# 
# 1. Quick test training (10 epochs):
#    ./train.sh quick --data_path aff_sec_result/sample/aff_sec_pairs.npy
#
# 2. Standard training with default parameters:
#    ./train.sh standard
#
# 3. Robust training for challenging datasets:
#    ./train.sh robust --data_path aff_sec_result/complex_object/aff_sec_pairs.npy
#
# 4. Custom training with specific parameters:
#    ./train.sh custom --batch_size 8 --max_epochs 50 --learning_rate 5e-4
#
# 5. Training on specific GPU:
#    ./train.sh standard --device cuda:0
#
# 6. Training with custom output directory:
#    ./train.sh standard --output_dir ./my_checkpoints
#
# CONFIGURATION DETAILS:
# - quick:    Fast training for testing and debugging
# - standard: Balanced configuration for most use cases
# - robust:   Intensive training for difficult or imbalanced datasets
# - custom:   Flexible configuration with manual parameter specification
#
# TIPS:
# - Use 'quick' configuration first to verify everything works
# - Monitor GPU memory usage with larger batch sizes
# - The script automatically creates the output directory
# - Training can be resumed by running the same command again
# =============================================================================
