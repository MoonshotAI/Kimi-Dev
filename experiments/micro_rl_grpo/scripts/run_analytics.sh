#!/bin/bash
# ============================================================================
# Analytics Script for Micro-RL GRPO Experiment
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# Default values
LOG_DIR=""
OUTPUT_DIR=""
NO_PLOTS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --log_dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no_plots)
            NO_PLOTS=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --log_dir DIR      Directory with training logs (required)"
            echo "  --output_dir DIR   Output directory for analysis"
            echo "  --no_plots         Skip plot generation"
            echo "  --help             Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Find log directory
if [ -z "$LOG_DIR" ]; then
    # Try to find most recent run
    LATEST_RUN=$(ls -td "${PROJECT_DIR}/runs/"*/ 2>/dev/null | head -1)

    if [ -n "$LATEST_RUN" ]; then
        LOG_DIR="${LATEST_RUN}logs"
        log_info "Using latest run: $LOG_DIR"
    else
        echo "Usage: $0 --log_dir <path_to_logs>"
        echo ""
        echo "Available runs:"
        ls -lt "${PROJECT_DIR}/runs/" 2>/dev/null || echo "  No runs found"
        exit 1
    fi
fi

if [ ! -d "$LOG_DIR" ]; then
    log_warn "Log directory not found: $LOG_DIR"
    exit 1
fi

# Set output directory
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${LOG_DIR}/../analysis"
fi

mkdir -p "$OUTPUT_DIR"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              Micro-RL GRPO Analytics                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

log_info "Log directory: $LOG_DIR"
log_info "Output directory: $OUTPUT_DIR"
echo ""

# Build command
CMD="python ${PROJECT_DIR}/analytics/analyze.py --log_dir $LOG_DIR --output_dir $OUTPUT_DIR"

if [ "$NO_PLOTS" = true ]; then
    CMD="$CMD --no_plots"
fi

# Run analytics
$CMD

echo ""
echo "Analysis complete!"
echo ""
echo "Generated files:"
ls -la "$OUTPUT_DIR"
echo ""

# Display report if available
REPORT_FILE="${OUTPUT_DIR}/analysis_report.md"
if [ -f "$REPORT_FILE" ]; then
    echo "═══════════════════════════════════════════════════════════════"
    cat "$REPORT_FILE"
    echo "═══════════════════════════════════════════════════════════════"
fi
