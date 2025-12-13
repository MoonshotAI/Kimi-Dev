#!/bin/bash
# ============================================================================
# Micro-RL GRPO Experiment Setup Script
# GigaChat-Lightning Base + Kimi-Dev paradigm
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "  Micro-RL GRPO Experiment Setup"
echo "  GigaChat-Lightning Base (10B-A1.8B)"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check CUDA
check_cuda() {
    log_info "Checking CUDA installation..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
        log_info "CUDA is available"
    else
        log_error "CUDA not found. Please install NVIDIA drivers."
        exit 1
    fi
}

# Create conda environment
setup_conda_env() {
    ENV_NAME="micro_rl_grpo"

    log_info "Setting up conda environment: $ENV_NAME"

    if conda env list | grep -q "^$ENV_NAME "; then
        log_warn "Environment $ENV_NAME already exists. Updating..."
        conda activate $ENV_NAME || source activate $ENV_NAME
    else
        log_info "Creating new environment..."
        conda create -n $ENV_NAME python=3.11 -y
        conda activate $ENV_NAME || source activate $ENV_NAME
    fi
}

# Install PyTorch with CUDA support
install_pytorch() {
    log_info "Installing PyTorch with CUDA 12.4 support..."
    pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
}

# Install core dependencies
install_dependencies() {
    log_info "Installing core dependencies..."

    pip install --upgrade pip

    # Core ML libraries
    pip install \
        transformers>=4.46.0 \
        accelerate>=1.0.0 \
        datasets>=3.0.0 \
        huggingface_hub>=0.26.0 \
        safetensors>=0.4.0

    # RL training
    pip install \
        trl>=0.12.0 \
        peft>=0.13.0

    # Flash Attention 2
    log_info "Installing Flash Attention 2..."
    pip install flash-attn --no-build-isolation

    # vLLM for fast inference
    log_info "Installing vLLM..."
    pip install vllm>=0.6.0

    # Docker SDK
    pip install docker>=7.0.0

    # Logging and analytics
    pip install \
        wandb>=0.18.0 \
        tensorboard>=2.18.0 \
        matplotlib>=3.9.0 \
        seaborn>=0.13.0 \
        pandas>=2.2.0 \
        numpy>=1.26.0 \
        tqdm>=4.66.0 \
        rich>=13.9.0

    # Code processing
    pip install \
        gitpython>=3.1.0 \
        unidiff>=0.7.0

    # YAML config
    pip install \
        pyyaml>=6.0 \
        omegaconf>=2.3.0
}

# Install Kimi-Dev dependencies
install_kimidev() {
    log_info "Installing Kimi-Dev package..."
    cd "$PROJECT_DIR/../.."
    pip install -e .
    cd "$PROJECT_DIR"
}

# Setup directories
setup_directories() {
    log_info "Setting up project directories..."

    mkdir -p "$PROJECT_DIR/logs"
    mkdir -p "$PROJECT_DIR/checkpoints"
    mkdir -p "$PROJECT_DIR/data/swebench"
    mkdir -p "$PROJECT_DIR/data/rollouts"
    mkdir -p "$PROJECT_DIR/analytics/plots"
    mkdir -p "$PROJECT_DIR/analytics/reports"

    log_info "Directories created successfully"
}

# Download SWE-bench data
download_swebench() {
    log_info "Downloading SWE-bench Verified dataset..."

    python -c "
from datasets import load_dataset
import json
import os

data_dir = '$PROJECT_DIR/data/swebench'

# Load SWE-bench Verified
dataset = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')

# Save to JSON
with open(os.path.join(data_dir, 'swebench_verified.json'), 'w') as f:
    json.dump([dict(item) for item in dataset], f, indent=2)

print(f'Downloaded {len(dataset)} instances')
"
    log_info "SWE-bench data downloaded"
}

# Main setup
main() {
    log_info "Starting setup..."

    check_cuda
    setup_conda_env
    install_pytorch
    install_dependencies
    install_kimidev
    setup_directories
    download_swebench

    echo ""
    echo "=============================================="
    echo "  Setup completed successfully!"
    echo "=============================================="
    echo ""
    echo "Next steps:"
    echo "  1. Download model: ./setup/download_model.sh"
    echo "  2. Setup Docker:   ./setup/setup_docker.sh"
    echo "  3. Run experiment: ./scripts/run_experiment.sh"
    echo ""
}

main "$@"
