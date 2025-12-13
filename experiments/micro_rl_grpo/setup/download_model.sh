#!/bin/bash
# ============================================================================
# Download GigaChat-Lightning Base Model from HuggingFace
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_DIR/models"

# Model configuration
MODEL_ID="ai-sage/GigaChat3-10B-A1.8B-base"
MODEL_NAME="gigachat-lightning-base"

echo "=============================================="
echo "  Downloading GigaChat-Lightning Base"
echo "  Model: $MODEL_ID"
echo "=============================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# Create model directory
mkdir -p "$MODEL_DIR"

# Check if model already exists
if [ -d "$MODEL_DIR/$MODEL_NAME" ] && [ -f "$MODEL_DIR/$MODEL_NAME/config.json" ]; then
    log_warn "Model already downloaded at $MODEL_DIR/$MODEL_NAME"
    read -p "Re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Using existing model"
        exit 0
    fi
fi

# Download model
log_info "Downloading model from HuggingFace..."
log_info "This may take a while (~20GB)..."

python -c "
import os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

model_id = '$MODEL_ID'
model_dir = '$MODEL_DIR/$MODEL_NAME'

print(f'Downloading {model_id}...')

# Download all files
snapshot_download(
    repo_id=model_id,
    local_dir=model_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)

print('Download complete!')

# Verify model can be loaded
print('Verifying model...')
config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
print(f'Model config loaded: {config.model_type}')
print(f'Hidden size: {config.hidden_size}')
print(f'Num layers: {config.num_hidden_layers}')
print(f'Num attention heads: {config.num_attention_heads}')

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
print(f'Tokenizer loaded: vocab_size={tokenizer.vocab_size}')

print('Model verification successful!')
"

# Create model info file
cat > "$MODEL_DIR/$MODEL_NAME/MODEL_INFO.md" << EOF
# GigaChat-Lightning Base

- **Source**: $MODEL_ID
- **Architecture**: MoE (Mixture of Experts)
- **Total Parameters**: ~10B
- **Active Parameters**: ~1.8B
- **Context Length**: 128K tokens
- **Downloaded**: $(date)

## Features
- Multi-head Latent Attention (MLA)
- Multi-Token Prediction (MTP)
- Optimized for Russian language

## Usage
\`\`\`python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "$MODEL_DIR/$MODEL_NAME",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "$MODEL_DIR/$MODEL_NAME",
    trust_remote_code=True
)
\`\`\`
EOF

echo ""
echo "=============================================="
echo "  Model downloaded successfully!"
echo "  Location: $MODEL_DIR/$MODEL_NAME"
echo "=============================================="
