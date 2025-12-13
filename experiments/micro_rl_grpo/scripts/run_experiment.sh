#!/bin/bash
# ============================================================================
# Main Script to Run Micro-RL GRPO Experiment
# GigaChat-Lightning Base + Kimi-Dev paradigm
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# Default values
CONFIG_FILE="${PROJECT_DIR}/configs/grpo_config.yaml"
MODE="train"
WANDB_ENABLED=false
DEBUG_MODE=false
SEED=42

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --wandb)
            WANDB_ENABLED=true
            shift
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config FILE    Configuration file (default: configs/grpo_config.yaml)"
            echo "  --mode MODE      Run mode: train, eval, both (default: train)"
            echo "  --wandb          Enable WandB logging"
            echo "  --debug          Enable debug mode (small dataset)"
            echo "  --seed SEED      Random seed (default: 42)"
            echo "  --help           Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         Micro-RL GRPO Experiment Runner                    ║"
echo "║         GigaChat-Lightning Base (10B-A1.8B)                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Check prerequisites
log_step "Step 1: Checking prerequisites..."

# Check Python
if ! command -v python &> /dev/null; then
    log_error "Python not found. Please install Python 3.11+"
    exit 1
fi

# Check CUDA
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    log_error "CUDA not available. Please check your GPU setup."
    exit 1
fi

# Print GPU info
log_info "GPU Information:"
python -c "
import torch
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)')
"

# Check model
MODEL_DIR="${PROJECT_DIR}/models/gigachat-lightning-base"
if [ ! -d "$MODEL_DIR" ] || [ ! -f "$MODEL_DIR/config.json" ]; then
    log_warn "Model not found at $MODEL_DIR"
    log_info "Please run: ./setup/download_model.sh"

    read -p "Download model now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        bash "${PROJECT_DIR}/setup/download_model.sh"
    else
        log_error "Model required. Exiting."
        exit 1
    fi
fi

log_info "Model found: $MODEL_DIR"

# Step 2: Setup directories
log_step "Step 2: Setting up directories..."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${PROJECT_DIR}/runs/run_${TIMESTAMP}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
LOG_DIR="${RUN_DIR}/logs"
ROLLOUT_DIR="${RUN_DIR}/rollouts"

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$ROLLOUT_DIR"

log_info "Run directory: $RUN_DIR"

# Copy config
cp "$CONFIG_FILE" "${RUN_DIR}/config.yaml"

# Step 3: Build command
log_step "Step 3: Building run command..."

CMD="python ${PROJECT_DIR}/run.py"
CMD="$CMD --config ${RUN_DIR}/config.yaml"
CMD="$CMD --mode $MODE"
CMD="$CMD --seed $SEED"
CMD="$CMD --output_dir $CHECKPOINT_DIR"

if [ "$WANDB_ENABLED" = true ]; then
    CMD="$CMD --wandb"
    log_info "WandB logging enabled"
fi

if [ "$DEBUG_MODE" = true ]; then
    CMD="$CMD --debug"
    log_info "Debug mode enabled (small dataset)"
fi

# Step 4: Run experiment
log_step "Step 4: Starting experiment..."
echo ""
log_info "Command: $CMD"
echo ""

# Create run info file
cat > "${RUN_DIR}/run_info.json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "config_file": "$CONFIG_FILE",
    "mode": "$MODE",
    "seed": $SEED,
    "wandb_enabled": $WANDB_ENABLED,
    "debug_mode": $DEBUG_MODE,
    "hostname": "$(hostname)",
    "user": "$(whoami)",
    "gpu_info": "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
}
EOF

# Run with logging
$CMD 2>&1 | tee "${LOG_DIR}/training.log"

EXIT_CODE=${PIPESTATUS[0]}

# Step 5: Post-processing
log_step "Step 5: Post-processing..."

if [ $EXIT_CODE -eq 0 ]; then
    log_info "Experiment completed successfully!"

    # Run analytics
    log_info "Generating analytics..."
    python "${PROJECT_DIR}/analytics/analyze.py" \
        --log_dir "$LOG_DIR" \
        --output_dir "${RUN_DIR}/analysis"

    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                    Experiment Complete!                    ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Results saved to: $RUN_DIR"
    echo ""
    echo "Key files:"
    echo "  - Checkpoints: $CHECKPOINT_DIR"
    echo "  - Logs: $LOG_DIR"
    echo "  - Analysis: ${RUN_DIR}/analysis"
    echo ""

    # Print summary if available
    if [ -f "${RUN_DIR}/analysis/analysis_report.md" ]; then
        echo "--- Analysis Summary ---"
        cat "${RUN_DIR}/analysis/analysis_report.md"
    fi
else
    log_error "Experiment failed with exit code $EXIT_CODE"
    log_error "Check logs at: ${LOG_DIR}/training.log"
    exit $EXIT_CODE
fi
