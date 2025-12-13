#!/bin/bash
# ============================================================================
# Evaluation Script for Micro-RL GRPO Experiment
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Default values
CHECKPOINT_PATH=""
CONFIG_FILE="${PROJECT_DIR}/configs/grpo_config.yaml"
NUM_SAMPLES=100
OUTPUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint PATH   Path to model checkpoint (required)"
            echo "  --config FILE       Configuration file"
            echo "  --num_samples N     Number of samples to evaluate"
            echo "  --output_dir DIR    Output directory for results"
            echo "  --help              Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate checkpoint
if [ -z "$CHECKPOINT_PATH" ]; then
    log_error "Checkpoint path required. Use --checkpoint PATH"

    # List available checkpoints
    log_info "Available checkpoints:"
    find "${PROJECT_DIR}/runs" -name "checkpoint-*" -o -name "best" -o -name "final" 2>/dev/null | head -10

    exit 1
fi

if [ ! -d "$CHECKPOINT_PATH" ]; then
    log_error "Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

# Set output directory
if [ -z "$OUTPUT_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${PROJECT_DIR}/eval_results/eval_${TIMESTAMP}"
fi

mkdir -p "$OUTPUT_DIR"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              Micro-RL GRPO Evaluation                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

log_info "Checkpoint: $CHECKPOINT_PATH"
log_info "Config: $CONFIG_FILE"
log_info "Samples: $NUM_SAMPLES"
log_info "Output: $OUTPUT_DIR"
echo ""

# Run evaluation
python "${PROJECT_DIR}/run.py" \
    --config "$CONFIG_FILE" \
    --mode eval \
    --checkpoint "$CHECKPOINT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"

# Run analytics
log_info "Generating evaluation report..."
python "${PROJECT_DIR}/analytics/analyze.py" \
    --log_dir "$OUTPUT_DIR" \
    --output_dir "${OUTPUT_DIR}/analysis"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                 Evaluation Complete!                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
