# Micro-RL GRPO Experiment

**GigaChat-Lightning Base (10B-A1.8B) + Kimi-Dev Paradigm on Single B200 GPU**

This experiment implements GRPO (Group Relative Policy Optimization) training for code repair, following the Kimi-Dev agentless training paradigm.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │   Problem    │    │   File       │    │   Code       │     │
│   │   Statement  │───▶│   Localize   │───▶│   Edit       │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│                              │                    │              │
│                              ▼                    ▼              │
│                       ┌──────────────────────────────┐          │
│                       │     SEARCH/REPLACE Patch     │          │
│                       └──────────────────────────────┘          │
│                                      │                          │
│                                      ▼                          │
│                       ┌──────────────────────────────┐          │
│                       │   Docker Test Execution      │          │
│                       │   (F2P + P2P metrics)        │          │
│                       └──────────────────────────────┘          │
│                                      │                          │
│                                      ▼                          │
│                       ┌──────────────────────────────┐          │
│                       │     GRPO Reward Signal       │          │
│                       │   (Test-based feedback)      │          │
│                       └──────────────────────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### One-Click Setup and Training

```bash
# Full pipeline: setup → download → train → analyze
./scripts/run_full_pipeline.sh

# With WandB logging
./scripts/run_full_pipeline.sh --wandb

# Debug mode (small dataset for testing)
./scripts/run_full_pipeline.sh --debug
```

### Step-by-Step

```bash
# 1. Setup environment
./setup/install.sh

# 2. Download GigaChat-Lightning Base from HuggingFace
./setup/download_model.sh

# 3. Setup Docker sandbox (optional but recommended)
./setup/setup_docker.sh

# 4. Run training
./scripts/run_experiment.sh

# 5. Evaluate checkpoint
./scripts/run_eval.sh --checkpoint checkpoints/best

# 6. Generate analytics
./scripts/run_analytics.sh --log_dir runs/<run_dir>/logs
```

## Project Structure

```
micro_rl_grpo/
├── setup/
│   ├── install.sh           # Environment setup
│   ├── download_model.sh    # Model download from HuggingFace
│   └── setup_docker.sh      # Docker sandbox setup
│
├── configs/
│   └── grpo_config.yaml     # Main configuration
│
├── src/
│   ├── __init__.py
│   ├── model.py             # Model loading and management
│   ├── grpo_trainer.py      # GRPO training loop
│   ├── reward.py            # Test-based reward computation
│   ├── data.py              # SWE-bench data loading
│   ├── metrics.py           # Logging and metrics tracking
│   └── rollout.py           # Rollout generation
│
├── scripts/
│   ├── run_full_pipeline.sh # One-click full pipeline
│   ├── run_experiment.sh    # Training script
│   ├── run_eval.sh          # Evaluation script
│   └── run_analytics.sh     # Analytics generation
│
├── analytics/
│   └── analyze.py           # Analysis and visualization
│
├── docker/
│   └── (generated)          # Docker sandbox files
│
├── run.py                   # Main entry point
└── README.md
```

## Configuration

Edit `configs/grpo_config.yaml` to customize:

```yaml
# Model
model:
  name: "ai-sage/GigaChat3-10B-A1.8B-base"
  max_seq_length: 49152  # 48K context

# GRPO Training
grpo:
  num_generations: 8     # Rollouts per prompt
  temperature: 1.0
  kl_coef: 0.05
  batch_size: 2
  learning_rate: 1.0e-6
  max_steps: 1000

# Data
data:
  dataset: "princeton-nlp/SWE-bench_Verified"
  subset_size: 100       # Problems for micro experiment

# Evaluation
evaluation:
  eval_steps: 100
  save_steps: 200
```

## Hardware Requirements

| Configuration | Memory | Notes |
|---------------|--------|-------|
| **B200 (Recommended)** | 192 GB | Full GRPO with 48K context |
| H100/H200 | 80 GB | Reduced batch size |
| A100 | 80 GB | Use gradient checkpointing |

### Memory Estimate for B200

```
Actor model (BF16):              ~20 GB
Reference model (BF16):          ~20 GB
Gradients:                       ~20 GB
Optimizer (Adam):                ~40 GB
Activations (grad checkpoint):   ~30 GB
KV-cache (48K context):          ~25 GB
───────────────────────────────────────
Total:                           ~155 GB ✓
```

## Key Concepts

### GRPO (Group Relative Policy Optimization)

GRPO is a variant of PPO that eliminates the critic network by using group-relative advantage estimation:

1. Generate multiple completions per prompt (group)
2. Compute rewards via test execution
3. Normalize rewards relative to group mean
4. Update policy using clipped objective

```
Loss = -E[min(r * A, clip(r) * A)] + kl_coef * KL
```

### Two-Stage Agentless Framework

Following Kimi-Dev paradigm:

1. **File Localization**: Identify relevant files based on problem statement
2. **Code Edit**: Generate SEARCH/REPLACE patches for identified files

### Reward Signal

Test-based reward using Docker sandbox:

- **Fail-to-Pass (F2P)**: Tests that should pass after fix
- **Pass-to-Pass (P2P)**: Tests that should remain passing
- **Resolution**: F2P=1 AND P2P=1

## Outputs

### Training Outputs

```
runs/<run_id>/
├── checkpoints/
│   ├── best/              # Best checkpoint by reward
│   ├── final/             # Final checkpoint
│   └── step-<N>/          # Intermediate checkpoints
├── logs/
│   ├── training.log       # Training log
│   ├── training_history.json
│   └── eval_history.json
├── analysis/
│   ├── plots/             # Training curves
│   ├── analysis.json      # Detailed metrics
│   └── analysis_report.md # Summary report
└── config.yaml            # Run configuration
```

### Metrics Tracked

- Loss (total, policy, KL)
- Rewards (mean, std, min, max)
- Resolution rate
- F2P and P2P scores
- Pass@K metrics
- Generation/evaluation timing

## Logging

### WandB Integration

```bash
# Enable WandB
./scripts/run_experiment.sh --wandb

# Or set in config
logging:
  use_wandb: true
  wandb_project: "micro-rl-grpo"
```

### TensorBoard

```bash
tensorboard --logdir runs/<run_id>/logs/tensorboard
```

## Troubleshooting

### Out of Memory

1. Reduce `batch_size` in config
2. Reduce `max_seq_length`
3. Enable `gradient_checkpointing: true`
4. Reduce `num_generations`

### Slow Training

1. Reduce `max_concurrent` Docker containers
2. Use smaller `subset_size` for testing
3. Enable Flash Attention 2

### Model Download Issues

```bash
# Manual download
huggingface-cli download ai-sage/GigaChat3-10B-A1.8B-base \
    --local-dir models/gigachat-lightning-base
```

## References

- [Kimi-Dev Paper](https://arxiv.org/abs/2509.23045)
- [GigaChat-Lightning](https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B-base)
- [SWE-bench](https://www.swebench.com/)
- [GRPO (DeepSeekMath)](https://arxiv.org/abs/2402.03300)

## License

This experiment code is provided for research purposes. Model weights are subject to their respective licenses.
