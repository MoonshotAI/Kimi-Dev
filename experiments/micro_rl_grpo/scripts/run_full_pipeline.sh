#!/bin/bash
# ============================================================================
# Full Pipeline: Setup → Download → Train → Evaluate → Analyze
# One-click experiment runner
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${CYAN}[====]${NC} $1"; }

# Timing
TOTAL_START=$(date +%s)

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                                                                    ║"
echo "║          MICRO-RL GRPO FULL PIPELINE                               ║"
echo "║          GigaChat-Lightning Base + Kimi-Dev Paradigm               ║"
echo "║                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Parse arguments
SKIP_SETUP=false
SKIP_DOWNLOAD=false
SKIP_DOCKER=false
WANDB_ENABLED=false
DEBUG_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-docker)
            SKIP_DOCKER=true
            shift
            ;;
        --wandb)
            WANDB_ENABLED=true
            shift
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-setup     Skip environment setup"
            echo "  --skip-download  Skip model download"
            echo "  --skip-docker    Skip Docker setup"
            echo "  --wandb          Enable WandB logging"
            echo "  --debug          Debug mode (small dataset)"
            echo "  --help           Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Phase 1: Environment Setup
# ============================================================================
log_step "PHASE 1: Environment Setup"
PHASE_START=$(date +%s)

if [ "$SKIP_SETUP" = false ]; then
    log_info "Running environment setup..."
    bash "${PROJECT_DIR}/setup/install.sh"
else
    log_info "Skipping environment setup (--skip-setup)"
fi

PHASE_END=$(date +%s)
log_info "Phase 1 completed in $((PHASE_END - PHASE_START)) seconds"
echo ""

# ============================================================================
# Phase 2: Model Download
# ============================================================================
log_step "PHASE 2: Model Download"
PHASE_START=$(date +%s)

MODEL_DIR="${PROJECT_DIR}/models/gigachat-lightning-base"

if [ "$SKIP_DOWNLOAD" = false ]; then
    if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
        log_info "Model already exists at $MODEL_DIR"
    else
        log_info "Downloading GigaChat-Lightning Base..."
        bash "${PROJECT_DIR}/setup/download_model.sh"
    fi
else
    log_info "Skipping model download (--skip-download)"
fi

PHASE_END=$(date +%s)
log_info "Phase 2 completed in $((PHASE_END - PHASE_START)) seconds"
echo ""

# ============================================================================
# Phase 3: Docker Sandbox Setup
# ============================================================================
log_step "PHASE 3: Docker Sandbox Setup"
PHASE_START=$(date +%s)

if [ "$SKIP_DOCKER" = false ]; then
    if command -v docker &> /dev/null; then
        log_info "Setting up Docker sandbox..."
        bash "${PROJECT_DIR}/setup/setup_docker.sh"
    else
        log_warn "Docker not found. Sandbox features will be limited."
    fi
else
    log_info "Skipping Docker setup (--skip-docker)"
fi

PHASE_END=$(date +%s)
log_info "Phase 3 completed in $((PHASE_END - PHASE_START)) seconds"
echo ""

# ============================================================================
# Phase 4: Training
# ============================================================================
log_step "PHASE 4: GRPO Training"
PHASE_START=$(date +%s)

TRAIN_ARGS="--mode train"

if [ "$WANDB_ENABLED" = true ]; then
    TRAIN_ARGS="$TRAIN_ARGS --wandb"
fi

if [ "$DEBUG_MODE" = true ]; then
    TRAIN_ARGS="$TRAIN_ARGS --debug"
fi

log_info "Starting GRPO training..."
bash "${PROJECT_DIR}/scripts/run_experiment.sh" $TRAIN_ARGS

PHASE_END=$(date +%s)
log_info "Phase 4 completed in $((PHASE_END - PHASE_START)) seconds"
echo ""

# ============================================================================
# Phase 5: Final Summary
# ============================================================================
log_step "PHASE 5: Final Summary"

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                     PIPELINE COMPLETE!                             ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Total runtime: $(printf '%02d:%02d:%02d' $((TOTAL_TIME/3600)) $((TOTAL_TIME%3600/60)) $((TOTAL_TIME%60)))"
echo ""
echo "Results location: ${PROJECT_DIR}/runs/"
echo ""

# List recent runs
log_info "Recent runs:"
ls -lt "${PROJECT_DIR}/runs/" 2>/dev/null | head -5 || echo "  No runs found"
echo ""

log_info "To analyze results:"
echo "  python analytics/analyze.py --log_dir runs/<run_dir>/logs"
echo ""

log_info "To evaluate a checkpoint:"
echo "  python run.py --mode eval --checkpoint runs/<run_dir>/checkpoints/best"
echo ""
