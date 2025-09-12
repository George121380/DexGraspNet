#!/bin/bash
# =============================================================================
# SecAff Affordance Model Evaluation Script
# 
# This script provides comprehensive evaluation options for the SecAff model
# including quantitative metrics and interactive 3D visualizations.
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

# Function to check if required files exist
check_files() {
    local model_path="$1"
    local data_path="$2"
    
    if [[ ! -f "$model_path" ]]; then
        print_error "Model file not found: $model_path"
        print_info "Please ensure the model checkpoint exists"
        exit 1
    fi
    print_success "Model file found: $model_path"
    
    if [[ ! -f "$data_path" ]]; then
        print_error "Data file not found: $data_path"
        print_info "Please ensure the data file exists"
        exit 1
    fi
    print_success "Data file found: $data_path"
}

# Function to display help
show_help() {
    cat << EOF
=============================================================================
SecAff Affordance Model Evaluation Script
=============================================================================

USAGE:
    ./eval.sh [MODE] [OPTIONS]

MODES:
    quick       - Quick evaluation without visualizations
    standard    - Standard evaluation with basic metrics
    visual      - Full evaluation with interactive 3D visualizations
    batch       - Batch evaluation for multiple models

REQUIRED OPTIONS:
    --model_path PATH       Path to trained model checkpoint (.pth file)
    --data_path PATH        Path to test data (.npy file)

OPTIONAL OPTIONS:
    --output_dir PATH       Output directory for results (default: ./evaluation_results)
    --batch_size N          Batch size for evaluation (default: 1)
    --max_vis_samples N     Maximum samples to visualize (default: 5)
    --device DEVICE         Device to use (auto/cuda/cpu, default: auto)
    --no_open               Do not auto-open browser when using visual mode
    --open_page PAGE        Which page to open: index|summary|top|all (default: index)
    --help                  Show this help message

EXAMPLES:
    # Quick evaluation
    ./eval.sh quick --model_path checkpoints/best_model.pth --data_path test_data.npy

    # Standard evaluation with metrics
    ./eval.sh standard --model_path checkpoints/best_model.pth --data_path aff_sec_result/2_of_Jenga_Classic_Game/aff_sec_pairs.npy

    # Full evaluation with interactive visualizations
    ./eval.sh visual --model_path checkpoints/best_model.pth --data_path test_data.npy --max_vis_samples 10

    # Batch evaluation with custom output directory
    ./eval.sh batch --model_path checkpoints/model_epoch_50.pth --data_path test_data.npy --output_dir ./results_epoch50

    # Evaluation on specific device
    ./eval.sh visual --model_path best_model.pth --data_path data.npy --device cuda:0

=============================================================================
EOF
}

# Default parameters
DEFAULT_OUTPUT_DIR="./evaluation_results"
DEFAULT_BATCH_SIZE=1
DEFAULT_MAX_VIS_SAMPLES=5
DEFAULT_DEVICE="auto"
DEFAULT_OPEN_PAGE="index"

# Parse mode
MODE="$1"
shift || true

# Initialize parameters based on mode
case "$MODE" in
    "quick")
        print_info "Using QUICK evaluation mode"
        VISUALIZE=false
        MAX_VIS_SAMPLES=0
        ;;
    "standard")
        print_info "Using STANDARD evaluation mode"
        VISUALIZE=false
        MAX_VIS_SAMPLES=0
        ;;
    "visual")
        print_info "Using VISUAL evaluation mode"
        VISUALIZE=true
        MAX_VIS_SAMPLES="$DEFAULT_MAX_VIS_SAMPLES"
        ;;
    "batch")
        print_info "Using BATCH evaluation mode"
        VISUALIZE=false
        MAX_VIS_SAMPLES=0
        ;;
    "--help" | "-h" | "help")
        show_help
        exit 0
        ;;
    "")
        print_error "No evaluation mode specified"
        show_help
        exit 1
        ;;
    *)
        print_error "Unknown evaluation mode: $MODE"
        show_help
        exit 1
        ;;
esac

# Set default values
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
DEVICE="$DEFAULT_DEVICE"
MODEL_PATH=""
DATA_PATH=""
AUTO_OPEN=false
OPEN_PAGE="$DEFAULT_OPEN_PAGE"

# Parse additional options
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
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
        --max_vis_samples)
            MAX_VIS_SAMPLES="$2"
            if [[ "$VISUALIZE" == "false" ]]; then
                VISUALIZE=true
                print_info "Enabling visualizations due to --max_vis_samples"
            fi
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --no_open)
            AUTO_OPEN=false
            shift 1
            ;;
        --open_page)
            OPEN_PAGE="$2"
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

# Check required parameters
if [[ -z "$MODEL_PATH" ]]; then
    print_error "Model path is required. Use --model_path option."
    show_help
    exit 1
fi

if [[ -z "$DATA_PATH" ]]; then
    print_error "Data path is required. Use --data_path option."
    show_help
    exit 1
fi

# Pre-flight checks
print_info "Performing pre-flight checks..."
check_environment
check_files "$MODEL_PATH" "$DATA_PATH"

# Create output directory
# Normalize paths relative to repo root if not absolute
if [[ "$OUTPUT_DIR" != /* ]]; then
    FULL_OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR"
else
    FULL_OUTPUT_DIR="$OUTPUT_DIR"
fi

mkdir -p "$FULL_OUTPUT_DIR"
print_success "Output directory ready: $FULL_OUTPUT_DIR"

# Display evaluation configuration
print_info "Evaluation Configuration:"
echo "  Mode:             $MODE"
echo "  Model Path:       $MODEL_PATH"
echo "  Data Path:        $DATA_PATH"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Batch Size:       $BATCH_SIZE"
echo "  Visualizations:   $([ "$VISUALIZE" == "true" ] && echo "Enabled ($MAX_VIS_SAMPLES samples)" || echo "Disabled")"
echo "  Device:           $DEVICE"
if [[ "$VISUALIZE" == "true" ]]; then
    # Default to auto-open unless explicitly disabled
    if [[ "$AUTO_OPEN" == "false" ]]; then
        : # keep false as set by --no_open
    else
        AUTO_OPEN=true
    fi
    echo "  Auto Open:        $AUTO_OPEN (page: $OPEN_PAGE)"
fi

# Normalize model and data paths relative to repo root if not absolute
if [[ "$MODEL_PATH" != /* ]]; then
    FULL_MODEL_PATH="$REPO_ROOT/$MODEL_PATH"
else
    FULL_MODEL_PATH="$MODEL_PATH"
fi

if [[ "$DATA_PATH" != /* ]]; then
    FULL_DATA_PATH="$REPO_ROOT/$DATA_PATH"
else
    FULL_DATA_PATH="$DATA_PATH"
fi

# Build evaluation command
EVAL_CMD="python \"$REPO_ROOT/train_eval/evaluate.py\""
EVAL_CMD="$EVAL_CMD --model_path \"$FULL_MODEL_PATH\""
EVAL_CMD="$EVAL_CMD --data_path \"$FULL_DATA_PATH\""
EVAL_CMD="$EVAL_CMD --output_dir \"$FULL_OUTPUT_DIR\""
EVAL_CMD="$EVAL_CMD --batch_size $BATCH_SIZE"
EVAL_CMD="$EVAL_CMD --device \"$DEVICE\""

if [[ "$VISUALIZE" == "true" ]]; then
    EVAL_CMD="$EVAL_CMD --visualize --max_vis_samples $MAX_VIS_SAMPLES"
fi

# Confirmation prompt (skip for quick mode)
if [[ "$MODE" != "quick" ]]; then
    echo
    read -p "Proceed with evaluation? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Evaluation cancelled"
        exit 0
    fi
fi

# Start evaluation
print_success "Starting evaluation..."
echo "============================================================================="

eval $EVAL_CMD

# Check evaluation result
if [[ $? -eq 0 ]]; then
    print_success "Evaluation completed successfully!"
    print_info "Results saved to: $FULL_OUTPUT_DIR"
    
    # Display results summary
    if [[ -f "$FULL_OUTPUT_DIR/metrics.json" ]]; then
        print_info "Metrics Summary:"
        python -c "
import json
try:
    with open('$FULL_OUTPUT_DIR/metrics.json', 'r') as f:
        metrics = json.load(f)
    print(f'  MSE:         {metrics[\"mse\"]:.6f}')
    print(f'  MAE:         {metrics[\"mae\"]:.6f}')
    print(f'  RMSE:        {metrics[\"rmse\"]:.6f}')
    print(f'  Correlation: {metrics[\"correlation\"]:.6f}')
    print(f'  Pred Range:  [{metrics[\"pred_range\"][0]:.3f}, {metrics[\"pred_range\"][1]:.3f}]')
except Exception as e:
    print(f'  Could not load metrics: {e}')
"
    fi
    
    # Display visualization info
    if [[ "$VISUALIZE" == "true" ]] && [[ -f "$FULL_OUTPUT_DIR/visualizations/index.html" ]]; then
        print_success "Interactive visualizations created!"
        print_info "Open the following file in your browser:"
        echo "  file://$FULL_OUTPUT_DIR/visualizations/index.html"
    fi
    
    # Display file locations
    print_info "Generated Files:"
    echo "  Predictions:      $FULL_OUTPUT_DIR/predictions.npy"
    echo "  Targets:          $FULL_OUTPUT_DIR/targets.npy"
    echo "  Metrics:          $FULL_OUTPUT_DIR/metrics.json"
    if [[ "$VISUALIZE" == "true" ]]; then
        echo "  Visualizations:   $FULL_OUTPUT_DIR/visualizations/"
    fi

    # Auto-open browser windows if requested
    if [[ "$VISUALIZE" == "true" ]] && [[ "$AUTO_OPEN" == "true" ]]; then
        open_in_browser() {
            local fpath="$1"
            if [[ -f "$fpath" ]]; then
                if command -v xdg-open >/dev/null 2>&1; then
                    xdg-open "file://$fpath" >/dev/null 2>&1 &
                elif command -v sensible-browser >/dev/null 2>&1; then
                    sensible-browser "file://$fpath" >/dev/null 2>&1 &
                else
                    python - <<PY
import webbrowser
webbrowser.open('file://' + r'''$fpath''')
PY
                fi
            else
                print_warning "File not found for opening: $fpath"
            fi
        }

        case "$OPEN_PAGE" in
            all)
                open_in_browser "$FULL_OUTPUT_DIR/visualizations/index.html"
                open_in_browser "$FULL_OUTPUT_DIR/visualizations/summary.html"
                open_in_browser "$FULL_OUTPUT_DIR/visualizations/top_errors.html"
                ;;
            summary)
                open_in_browser "$FULL_OUTPUT_DIR/visualizations/summary.html"
                ;;
            top)
                open_in_browser "$FULL_OUTPUT_DIR/visualizations/top_errors.html"
                ;;
            index|*)
                open_in_browser "$FULL_OUTPUT_DIR/visualizations/index.html"
                ;;
        esac
    fi
    
else
    print_error "Evaluation failed!"
    exit 1
fi

# =============================================================================
# USAGE EXAMPLES:
# 
# 1. Quick evaluation for testing:
#    ./eval.sh quick --model_path checkpoints/best_model.pth --data_path test_data.npy
#
# 2. Standard evaluation with detailed metrics:
#    ./eval.sh standard --model_path checkpoints/best_model.pth --data_path aff_sec_result/2_of_Jenga_Classic_Game/aff_sec_pairs.npy
#
# 3. Full evaluation with interactive 3D visualizations:
#    ./eval.sh visual --model_path checkpoints/best_model.pth --data_path test_data.npy --max_vis_samples 10
#
# 4. Batch evaluation with custom output directory:
#    ./eval.sh batch --model_path checkpoints/model_epoch_50.pth --data_path test_data.npy --output_dir ./results_epoch50
#
# 5. Evaluation on specific GPU:
#    ./eval.sh visual --model_path best_model.pth --data_path data.npy --device cuda:0
#
# 6. Large-scale evaluation with bigger batch size:
#    ./eval.sh standard --model_path best_model.pth --data_path large_dataset.npy --batch_size 16
#
# MODE DETAILS:
# - quick:    Fast evaluation with basic metrics only
# - standard: Comprehensive metrics without visualizations
# - visual:   Full evaluation with interactive 3D visualizations
# - batch:    Optimized for evaluating multiple models/datasets
#
# VISUALIZATION FEATURES:
# - Interactive 3D point clouds with affordance coloring
# - Side-by-side comparison of predictions vs ground truth
# - Error visualization with color-coded differences
# - HTML-based interface that works in any modern browser
# - Hover tooltips with exact values for detailed analysis
#
# OUTPUT FILES:
# - predictions.npy:   Model predictions for all test samples
# - targets.npy:       Ground truth affordance scores
# - metrics.json:      Comprehensive evaluation metrics
# - visualizations/    Interactive HTML visualizations (if enabled)
#
# TIPS:
# - Use 'quick' mode for rapid testing during development
# - Use 'visual' mode to understand model behavior qualitatively
# - Visualizations help identify systematic errors and biases
# - Check correlation coefficient to assess overall prediction quality
# =============================================================================