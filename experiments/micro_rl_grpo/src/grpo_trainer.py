"""
GRPO (Group Relative Policy Optimization) Trainer

Implementation of GRPO for code repair following Kimi-Dev paradigm.
GRPO eliminates the need for a critic network by using group-based
advantage estimation.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
)
from torch.optim import AdamW

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    # Core GRPO parameters
    num_generations: int = 8
    temperature: float = 1.0
    top_p: float = 0.95
    kl_coef: float = 0.05
    clip_range: float = 0.2

    # Reward settings
    reward_baseline: str = "mean"  # "mean" or "min"
    reward_scale: float = 1.0

    # Training parameters
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_steps: int = 1000
    num_train_epochs: int = 3

    # Generation parameters
    max_new_tokens: int = 16384
    max_seq_length: int = 49152

    # Memory optimization
    gradient_checkpointing: bool = True
    bf16: bool = True


@dataclass
class GRPOBatch:
    """A batch for GRPO training."""

    prompts: List[str]
    prompt_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    completions: List[List[str]]
    completion_ids: torch.Tensor
    completion_attention_mask: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    old_log_probs: torch.Tensor
    ref_log_probs: torch.Tensor


class GRPOTrainer:
    """
    GRPO Trainer for code repair.

    GRPO (Group Relative Policy Optimization) is a variant of PPO that:
    1. Generates multiple completions per prompt (group)
    2. Computes advantages relative to the group mean
    3. Eliminates the need for a separate critic network
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        ref_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: GRPOConfig,
        reward_fn: callable,
        output_dir: str = "./checkpoints",
        logging_dir: str = "./logs",
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.output_dir = Path(output_dir)
        self.logging_dir = Path(logging_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logging_dir.mkdir(parents=True, exist_ok=True)

        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = None  # Will be created when we know total steps

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_reward = float("-inf")

        # Metrics history
        self.history = {
            "steps": [],
            "loss": [],
            "policy_loss": [],
            "kl_loss": [],
            "rewards": [],
            "advantages": [],
            "learning_rate": [],
        }

        logger.info(f"GRPOTrainer initialized")
        logger.info(f"  num_generations: {config.num_generations}")
        logger.info(f"  kl_coef: {config.kl_coef}")
        logger.info(f"  clip_range: {config.clip_range}")

    def _create_optimizer(self) -> AdamW:
        """Create optimizer with weight decay."""

        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "LayerNorm" in name or "layernorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_params = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = AdamW(
            optimizer_grouped_params,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        return optimizer

    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler."""

        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        self.scheduler = get_scheduler(
            name=self.config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        logger.info(f"Scheduler created: {self.config.lr_scheduler_type}")
        logger.info(f"  warmup_steps: {num_warmup_steps}")
        logger.info(f"  total_steps: {num_training_steps}")

    @torch.no_grad()
    def generate_completions(
        self,
        prompts: List[str],
        num_generations: int = None,
    ) -> List[List[str]]:
        """
        Generate multiple completions for each prompt.

        Args:
            prompts: List of prompt strings
            num_generations: Number of completions per prompt

        Returns:
            List of lists of completion strings
        """

        num_generations = num_generations or self.config.num_generations

        self.model.eval()

        all_completions = []

        for prompt in prompts:
            # Tokenize prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_length - self.config.max_new_tokens,
            ).to(self.model.device)

            # Generate multiple completions
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                num_return_sequences=num_generations,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # Decode completions (remove prompt)
            prompt_length = inputs["input_ids"].shape[1]
            completions = []

            for output in outputs:
                completion_ids = output[prompt_length:]
                completion = self.tokenizer.decode(
                    completion_ids,
                    skip_special_tokens=True,
                )
                completions.append(completion)

            all_completions.append(completions)

        self.model.train()

        return all_completions

    def compute_log_probs(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities for given sequences."""

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = outputs.logits[:, :-1, :]  # Shift for next token prediction
        labels = labels[:, 1:]  # Shift labels

        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=labels.unsqueeze(-1),
        ).squeeze(-1)

        # Mask padding
        mask = (labels != self.tokenizer.pad_token_id).float()
        token_log_probs = token_log_probs * mask

        # Sum log probs per sequence
        seq_log_probs = token_log_probs.sum(dim=-1)

        return seq_log_probs

    def compute_advantages(
        self,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GRPO advantages using group-relative baseline.

        Args:
            rewards: Tensor of shape (batch_size, num_generations)

        Returns:
            advantages: Tensor of same shape with normalized advantages
        """

        if self.config.reward_baseline == "mean":
            baseline = rewards.mean(dim=-1, keepdim=True)
        elif self.config.reward_baseline == "min":
            baseline = rewards.min(dim=-1, keepdim=True).values
        else:
            baseline = torch.zeros_like(rewards[:, :1])

        advantages = rewards - baseline

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        return advantages

    def compute_loss(
        self,
        batch: GRPOBatch,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO loss.

        Loss = -E[min(r * A, clip(r, 1-eps, 1+eps) * A)] + kl_coef * KL

        Where:
            r = exp(log_prob - old_log_prob) is the probability ratio
            A = advantages
            KL = KL divergence from reference model
        """

        # Compute current log probs
        current_log_probs = self.compute_log_probs(
            self.model,
            batch.completion_ids,
            batch.completion_attention_mask,
            batch.completion_ids,
        )

        # Compute probability ratio
        ratio = torch.exp(current_log_probs - batch.old_log_probs)

        # Clipped surrogate objective
        surr1 = ratio * batch.advantages
        surr2 = torch.clamp(
            ratio,
            1.0 - self.config.clip_range,
            1.0 + self.config.clip_range,
        ) * batch.advantages

        policy_loss = -torch.min(surr1, surr2).mean()

        # KL divergence from reference model
        kl_div = (batch.old_log_probs - batch.ref_log_probs).mean()
        kl_loss = self.config.kl_coef * kl_div

        # Total loss
        total_loss = policy_loss + kl_loss

        # Metrics
        metrics = {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "kl_div": kl_div.item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_std": ratio.std().item(),
        }

        return total_loss, metrics

    def train_step(
        self,
        prompts: List[str],
        problem_data: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Perform one GRPO training step.

        Args:
            prompts: List of prompts (problem statements)
            problem_data: Additional data for reward computation

        Returns:
            Dictionary of metrics
        """

        self.model.train()

        # 1. Generate completions
        logger.debug("Generating completions...")
        completions = self.generate_completions(prompts)

        # 2. Compute rewards via test execution
        logger.debug("Computing rewards...")
        rewards_list = []
        for i, (prompt, comps, data) in enumerate(zip(prompts, completions, problem_data)):
            prompt_rewards = []
            for comp in comps:
                reward = self.reward_fn(
                    problem_data=data,
                    completion=comp,
                )
                prompt_rewards.append(reward)
            rewards_list.append(prompt_rewards)

        rewards = torch.tensor(rewards_list, device=self.model.device)

        # 3. Compute advantages
        advantages = self.compute_advantages(rewards)

        # 4. Prepare batch for training
        # Flatten completions for batch processing
        flat_completions = [c for comps in completions for c in comps]
        flat_prompts = [p for p in prompts for _ in range(self.config.num_generations)]

        # Tokenize
        full_texts = [p + c for p, c in zip(flat_prompts, flat_completions)]
        encodings = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
        ).to(self.model.device)

        # Compute old log probs (before update)
        with torch.no_grad():
            old_log_probs = self.compute_log_probs(
                self.model,
                encodings["input_ids"],
                encodings["attention_mask"],
                encodings["input_ids"],
            )

            ref_log_probs = self.compute_log_probs(
                self.ref_model,
                encodings["input_ids"],
                encodings["attention_mask"],
                encodings["input_ids"],
            )

        # Create batch
        batch = GRPOBatch(
            prompts=flat_prompts,
            prompt_ids=None,
            prompt_attention_mask=None,
            completions=flat_completions,
            completion_ids=encodings["input_ids"],
            completion_attention_mask=encodings["attention_mask"],
            rewards=rewards.flatten(),
            advantages=advantages.flatten(),
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
        )

        # 5. Compute loss and update
        loss, metrics = self.compute_loss(batch)

        # Backward pass
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()

        # Update weights
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()

        # Add reward metrics
        metrics["reward_mean"] = rewards.mean().item()
        metrics["reward_std"] = rewards.std().item()
        metrics["reward_max"] = rewards.max().item()
        metrics["reward_min"] = rewards.min().item()
        metrics["advantage_mean"] = advantages.mean().item()
        metrics["advantage_std"] = advantages.std().item()

        if self.scheduler:
            metrics["learning_rate"] = self.scheduler.get_last_lr()[0]

        self.global_step += 1

        return metrics

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        eval_steps: int = 100,
        save_steps: int = 200,
        log_steps: int = 10,
    ):
        """
        Main training loop.

        Args:
            train_dataloader: Training data loader
            eval_dataloader: Optional evaluation data loader
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps
            log_steps: Log metrics every N steps
        """

        # Calculate total steps
        total_steps = min(
            len(train_dataloader) * self.config.num_train_epochs,
            self.config.max_steps,
        )

        self._create_scheduler(total_steps)

        logger.info("=" * 60)
        logger.info("Starting GRPO Training")
        logger.info("=" * 60)
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        logger.info("=" * 60)

        progress_bar = tqdm(total=total_steps, desc="Training")

        for epoch in range(self.config.num_train_epochs):
            self.epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_train_epochs}")

            for batch_idx, batch in enumerate(train_dataloader):
                if self.global_step >= self.config.max_steps:
                    break

                # Training step
                prompts = batch["prompts"]
                problem_data = batch["problem_data"]

                metrics = self.train_step(prompts, problem_data)

                # Update history
                self.history["steps"].append(self.global_step)
                self.history["loss"].append(metrics["loss"])
                self.history["policy_loss"].append(metrics["policy_loss"])
                self.history["kl_loss"].append(metrics["kl_loss"])
                self.history["rewards"].append(metrics["reward_mean"])
                self.history["advantages"].append(metrics["advantage_mean"])
                if "learning_rate" in metrics:
                    self.history["learning_rate"].append(metrics["learning_rate"])

                # Logging
                if self.global_step % log_steps == 0:
                    self._log_metrics(metrics)

                # Evaluation
                if eval_dataloader and self.global_step % eval_steps == 0:
                    eval_metrics = self.evaluate(eval_dataloader)
                    self._log_eval_metrics(eval_metrics)

                    # Save best model
                    if eval_metrics["reward_mean"] > self.best_reward:
                        self.best_reward = eval_metrics["reward_mean"]
                        self.save_checkpoint("best")

                # Save checkpoint
                if self.global_step % save_steps == 0:
                    self.save_checkpoint(f"step-{self.global_step}")

                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "reward": f"{metrics['reward_mean']:.4f}",
                })

            if self.global_step >= self.config.max_steps:
                break

        progress_bar.close()

        # Final save
        self.save_checkpoint("final")
        self._save_history()

        logger.info("Training complete!")
        logger.info(f"Best reward: {self.best_reward:.4f}")

    @torch.no_grad()
    def evaluate(
        self,
        eval_dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate model on evaluation set."""

        self.model.eval()

        all_rewards = []
        all_metrics = []

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            prompts = batch["prompts"]
            problem_data = batch["problem_data"]

            # Generate completions
            completions = self.generate_completions(prompts, num_generations=1)

            # Compute rewards
            for i, (data, comps) in enumerate(zip(problem_data, completions)):
                reward = self.reward_fn(
                    problem_data=data,
                    completion=comps[0],
                )
                all_rewards.append(reward)

        self.model.train()

        eval_metrics = {
            "reward_mean": sum(all_rewards) / len(all_rewards),
            "reward_std": torch.tensor(all_rewards).std().item(),
            "reward_max": max(all_rewards),
            "reward_min": min(all_rewards),
            "num_samples": len(all_rewards),
        }

        return eval_metrics

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log training metrics."""

        log_str = f"Step {self.global_step}"
        for key, value in metrics.items():
            log_str += f" | {key}: {value:.4f}"

        logger.info(log_str)

    def _log_eval_metrics(self, metrics: Dict[str, float]):
        """Log evaluation metrics."""

        logger.info("=" * 40)
        logger.info(f"Evaluation at step {self.global_step}")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        logger.info("=" * 40)

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""

        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        torch.save({
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_reward": self.best_reward,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
        }, checkpoint_dir / "training_state.pt")

        # Save config
        with open(checkpoint_dir / "config.json", "w") as f:
            json.dump(vars(self.config), f, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_dir}")

    def _save_history(self):
        """Save training history."""

        history_path = self.logging_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"Training history saved: {history_path}")


if __name__ == "__main__":
    # Test GRPO trainer
    logging.basicConfig(level=logging.INFO)

    config = GRPOConfig(
        num_generations=4,
        batch_size=1,
        max_steps=10,
    )

    print(f"GRPO Config: {config}")
