#!/usr/bin/env python3
"""
Main Entry Point for Micro-RL GRPO Experiment

GigaChat-Lightning Base + Kimi-Dev paradigm on single B200 GPU.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
import yaml
import json
import random

import torch
import numpy as np

# Add project to path
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from src.model import ModelManager, load_model, load_tokenizer
from src.grpo_trainer import GRPOTrainer, GRPOConfig
from src.reward import RewardComputer, create_reward_function
from src.data import SWEBenchDataset, create_dataloaders, PromptBuilder
from src.metrics import MetricsTracker
from src.rollout import RolloutGenerator

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path, log_level: str = "INFO"):
    """Setup logging configuration."""

    log_dir.mkdir(parents=True, exist_ok=True)

    # Create formatters
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(console_formatter)

    # File handler
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")


def set_seed(seed: int):
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info(f"Random seed set to {seed}")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded from {config_path}")

    return config


def check_gpu():
    """Check GPU availability and print info."""

    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    logger.info(f"Found {num_gpus} GPU(s)")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        logger.info(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")

    # Check if B200 or similar
    gpu_name = torch.cuda.get_device_properties(0).name
    if "B200" in gpu_name or "B100" in gpu_name:
        logger.info("Detected Blackwell GPU - optimal configuration!")
    elif "H100" in gpu_name or "H200" in gpu_name:
        logger.info("Detected Hopper GPU - good configuration!")
    elif "A100" in gpu_name:
        logger.info("Detected Ampere GPU - acceptable configuration")

    return num_gpus


def run_training(config: dict):
    """Run GRPO training loop."""

    logger.info("=" * 60)
    logger.info("Starting Micro-RL GRPO Training")
    logger.info("=" * 60)

    experiment_config = config.get("experiment", {})
    model_config = config.get("model", {})
    grpo_config_dict = config.get("grpo", {})
    logging_config = config.get("logging", {})

    # Directories
    output_dir = Path(experiment_config.get("output_dir", "./checkpoints"))
    logging_dir = Path(experiment_config.get("logging_dir", "./logs"))

    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(
        output_dir=str(logging_dir),
        experiment_name=experiment_config.get("name", "micro_rl_grpo"),
        use_wandb=logging_config.get("use_wandb", False),
        wandb_project=logging_config.get("wandb_project"),
        wandb_entity=logging_config.get("wandb_entity"),
        config=config,
    )

    # Load model
    logger.info("Loading model...")
    model_manager = ModelManager(config)
    model, ref_model, tokenizer = model_manager.setup()

    logger.info(f"Model loaded on device: {next(model.parameters()).device}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader, eval_dataloader = create_dataloaders(config)

    # Create reward function
    logger.info("Setting up reward computation...")
    reward_fn = create_reward_function(config)

    # Create GRPO config
    grpo_config = GRPOConfig(
        num_generations=grpo_config_dict.get("num_generations", 8),
        temperature=grpo_config_dict.get("temperature", 1.0),
        top_p=grpo_config_dict.get("top_p", 0.95),
        kl_coef=grpo_config_dict.get("kl_coef", 0.05),
        clip_range=grpo_config_dict.get("clip_range", 0.2),
        batch_size=grpo_config_dict.get("batch_size", 2),
        gradient_accumulation_steps=grpo_config_dict.get("gradient_accumulation_steps", 8),
        learning_rate=grpo_config_dict.get("learning_rate", 1e-6),
        weight_decay=grpo_config_dict.get("weight_decay", 0.01),
        warmup_ratio=grpo_config_dict.get("warmup_ratio", 0.1),
        max_steps=grpo_config_dict.get("max_steps", 1000),
        num_train_epochs=grpo_config_dict.get("num_train_epochs", 3),
        max_new_tokens=model_config.get("max_new_tokens", 16384),
        max_seq_length=model_config.get("max_seq_length", 49152),
        gradient_checkpointing=grpo_config_dict.get("gradient_checkpointing", True),
        bf16=grpo_config_dict.get("bf16", True),
    )

    # Create trainer
    logger.info("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        config=grpo_config,
        reward_fn=reward_fn,
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
    )

    # Training loop
    eval_config = config.get("evaluation", {})

    logger.info("Starting training...")
    trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        eval_steps=eval_config.get("eval_steps", 100),
        save_steps=eval_config.get("save_steps", 200),
        log_steps=logging_config.get("log_every_n_steps", 10),
    )

    # Final evaluation
    logger.info("Running final evaluation...")
    final_metrics = trainer.evaluate(eval_dataloader)

    metrics_tracker.log_eval_step(
        step=trainer.global_step,
        metrics=final_metrics,
    )

    # Save final results
    metrics_tracker.save()
    metrics_tracker.close()

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Best reward: {trainer.best_reward:.4f}")
    logger.info(f"Checkpoints saved to: {output_dir}")
    logger.info(f"Logs saved to: {logging_dir}")
    logger.info("=" * 60)

    return trainer.best_reward


def run_evaluation(config: dict, checkpoint_path: str):
    """Run evaluation on a checkpoint."""

    logger.info("=" * 60)
    logger.info("Running Evaluation")
    logger.info("=" * 60)

    # Load model from checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    tokenizer = load_tokenizer(checkpoint_path)
    model = load_model(
        checkpoint_path,
        torch_dtype=config.get("model", {}).get("torch_dtype", "bfloat16"),
    )

    # Create dataloader
    _, eval_dataloader = create_dataloaders(config)

    # Create reward function
    reward_fn = create_reward_function(config)

    # Create rollout generator
    rollout_gen = RolloutGenerator(
        model=model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        num_generations=config.get("grpo", {}).get("num_generations", 8),
        output_dir="./eval_rollouts",
    )

    # Generate rollouts
    all_batches = []
    for batch in eval_dataloader:
        batches = rollout_gen.generate_rollouts(
            prompts=batch["prompts"],
            problem_data_list=batch["problem_data"],
        )
        all_batches.extend(batches)

    # Compute metrics
    total_resolved = sum(
        any(r.resolved for r in batch.rollouts)
        for batch in all_batches
    )

    resolve_rate = total_resolved / len(all_batches) if all_batches else 0

    logger.info(f"Evaluation Results:")
    logger.info(f"  Total instances: {len(all_batches)}")
    logger.info(f"  Resolved: {total_resolved}")
    logger.info(f"  Resolve rate: {resolve_rate:.2%}")

    # Save rollouts
    rollout_gen.save_rollouts(all_batches)

    return resolve_rate


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Micro-RL GRPO Experiment with GigaChat-Lightning"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/grpo_config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "both"],
        default="train",
        help="Run mode: train, eval, or both"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for evaluation"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable WandB logging"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    args = parser.parse_args()

    # Load config
    config_path = PROJECT_DIR / args.config
    config = load_config(str(config_path))

    # Override config with command line args
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed

    if args.output_dir is not None:
        config["experiment"]["output_dir"] = args.output_dir

    if args.wandb:
        config["logging"]["use_wandb"] = True

    if args.debug:
        config["logging"]["log_level"] = "DEBUG"
        config["grpo"]["max_steps"] = 10
        config["data"]["subset_size"] = 5

    # Setup
    logging_dir = Path(config.get("experiment", {}).get("logging_dir", "./logs"))
    log_level = config.get("logging", {}).get("log_level", "INFO")
    setup_logging(logging_dir, log_level)

    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    check_gpu()

    # Print banner
    logger.info("")
    logger.info("╔════════════════════════════════════════════════════════════╗")
    logger.info("║         Micro-RL GRPO Experiment                           ║")
    logger.info("║         GigaChat-Lightning Base (10B-A1.8B)                ║")
    logger.info("║         Following Kimi-Dev Paradigm                        ║")
    logger.info("╚════════════════════════════════════════════════════════════╝")
    logger.info("")

    # Run
    if args.mode in ["train", "both"]:
        best_reward = run_training(config)

        if args.mode == "both":
            # Use best checkpoint for evaluation
            checkpoint_path = Path(config["experiment"]["output_dir"]) / "best"
            run_evaluation(config, str(checkpoint_path))

    elif args.mode == "eval":
        if not args.checkpoint:
            logger.error("--checkpoint required for evaluation mode")
            sys.exit(1)
        run_evaluation(config, args.checkpoint)


if __name__ == "__main__":
    main()
