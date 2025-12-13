"""
Metrics Tracking and Logging for GRPO Training

Handles WandB integration, TensorBoard logging, and local metrics storage.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics for a training step."""

    step: int
    epoch: int
    timestamp: float

    # Loss metrics
    loss: float
    policy_loss: float
    kl_loss: float
    kl_div: float

    # Reward metrics
    reward_mean: float
    reward_std: float
    reward_max: float
    reward_min: float

    # Advantage metrics
    advantage_mean: float
    advantage_std: float

    # Training metrics
    learning_rate: float
    ratio_mean: float = 0.0
    ratio_std: float = 0.0

    # Resolution metrics
    resolve_rate: float = 0.0
    f2p_mean: float = 0.0
    p2p_mean: float = 0.0


@dataclass
class EvalMetrics:
    """Metrics for evaluation."""

    step: int
    timestamp: float

    # Core metrics
    reward_mean: float
    reward_std: float
    resolve_rate: float
    f2p_mean: float
    p2p_mean: float

    # Pass@K metrics
    pass_at_1: float = 0.0
    pass_at_5: float = 0.0
    pass_at_10: float = 0.0

    # Additional metrics
    num_samples: int = 0
    patch_validity_rate: float = 0.0
    file_localization_acc: float = 0.0


class MetricsTracker:
    """
    Tracks and logs training metrics.

    Supports:
    - WandB logging
    - TensorBoard logging
    - Local JSON logging
    - Rich console output
    """

    def __init__(
        self,
        output_dir: str,
        experiment_name: str,
        use_wandb: bool = True,
        use_tensorboard: bool = True,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.config = config or {}

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # History storage
        self.training_history: List[TrainingMetrics] = []
        self.eval_history: List[EvalMetrics] = []
        self.custom_metrics: Dict[str, List[Any]] = defaultdict(list)

        # Running averages
        self.running_loss = RunningAverage(window=100)
        self.running_reward = RunningAverage(window=100)

        # Initialize loggers
        self._init_wandb(wandb_project, wandb_entity)
        self._init_tensorboard()
        self._init_rich_console()

        logger.info(f"MetricsTracker initialized: {experiment_name}")

    def _init_wandb(self, project: Optional[str], entity: Optional[str]):
        """Initialize WandB logging."""

        if not self.use_wandb:
            self.wandb_run = None
            return

        try:
            import wandb

            self.wandb_run = wandb.init(
                project=project or "micro-rl-grpo",
                entity=entity,
                name=self.experiment_name,
                config=self.config,
                dir=str(self.output_dir),
                reinit=True,
            )
            logger.info(f"WandB initialized: {self.wandb_run.url}")

        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
            self.wandb_run = None
            self.use_wandb = False

    def _init_tensorboard(self):
        """Initialize TensorBoard logging."""

        if not self.use_tensorboard:
            self.tb_writer = None
            return

        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = self.output_dir / "tensorboard"
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            logger.info(f"TensorBoard initialized: {tb_dir}")

        except Exception as e:
            logger.warning(f"Failed to initialize TensorBoard: {e}")
            self.tb_writer = None
            self.use_tensorboard = False

    def _init_rich_console(self):
        """Initialize rich console for pretty output."""

        try:
            from rich.console import Console
            from rich.table import Table

            self.console = Console()
            self.use_rich = True

        except ImportError:
            self.console = None
            self.use_rich = False

    def log_training_step(
        self,
        step: int,
        epoch: int,
        metrics: Dict[str, float],
    ):
        """Log metrics for a training step."""

        # Create metrics object
        training_metrics = TrainingMetrics(
            step=step,
            epoch=epoch,
            timestamp=time.time(),
            loss=metrics.get("loss", 0),
            policy_loss=metrics.get("policy_loss", 0),
            kl_loss=metrics.get("kl_loss", 0),
            kl_div=metrics.get("kl_div", 0),
            reward_mean=metrics.get("reward_mean", 0),
            reward_std=metrics.get("reward_std", 0),
            reward_max=metrics.get("reward_max", 0),
            reward_min=metrics.get("reward_min", 0),
            advantage_mean=metrics.get("advantage_mean", 0),
            advantage_std=metrics.get("advantage_std", 0),
            learning_rate=metrics.get("learning_rate", 0),
            ratio_mean=metrics.get("ratio_mean", 0),
            ratio_std=metrics.get("ratio_std", 0),
        )

        # Store in history
        self.training_history.append(training_metrics)

        # Update running averages
        self.running_loss.update(training_metrics.loss)
        self.running_reward.update(training_metrics.reward_mean)

        # Log to WandB
        if self.wandb_run:
            import wandb
            wandb.log(
                {f"train/{k}": v for k, v in asdict(training_metrics).items()},
                step=step,
            )

        # Log to TensorBoard
        if self.tb_writer:
            for key, value in asdict(training_metrics).items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f"train/{key}", value, step)

    def log_eval_step(
        self,
        step: int,
        metrics: Dict[str, float],
    ):
        """Log metrics for an evaluation step."""

        eval_metrics = EvalMetrics(
            step=step,
            timestamp=time.time(),
            reward_mean=metrics.get("reward_mean", 0),
            reward_std=metrics.get("reward_std", 0),
            resolve_rate=metrics.get("resolve_rate", 0),
            f2p_mean=metrics.get("f2p_mean", 0),
            p2p_mean=metrics.get("p2p_mean", 0),
            pass_at_1=metrics.get("pass_at_1", 0),
            pass_at_5=metrics.get("pass_at_5", 0),
            pass_at_10=metrics.get("pass_at_10", 0),
            num_samples=metrics.get("num_samples", 0),
            patch_validity_rate=metrics.get("patch_validity_rate", 0),
            file_localization_acc=metrics.get("file_localization_acc", 0),
        )

        self.eval_history.append(eval_metrics)

        # Log to WandB
        if self.wandb_run:
            import wandb
            wandb.log(
                {f"eval/{k}": v for k, v in asdict(eval_metrics).items()},
                step=step,
            )

        # Log to TensorBoard
        if self.tb_writer:
            for key, value in asdict(eval_metrics).items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f"eval/{key}", value, step)

        # Pretty print
        self._print_eval_metrics(eval_metrics)

    def log_custom(
        self,
        name: str,
        value: Any,
        step: int,
    ):
        """Log a custom metric."""

        self.custom_metrics[name].append({"step": step, "value": value})

        if self.wandb_run:
            import wandb
            wandb.log({name: value}, step=step)

        if self.tb_writer and isinstance(value, (int, float)):
            self.tb_writer.add_scalar(name, value, step)

    def _print_eval_metrics(self, metrics: EvalMetrics):
        """Pretty print evaluation metrics."""

        if self.use_rich:
            from rich.table import Table

            table = Table(title=f"Evaluation Results (Step {metrics.step})")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Reward Mean", f"{metrics.reward_mean:.4f}")
            table.add_row("Resolve Rate", f"{metrics.resolve_rate:.2%}")
            table.add_row("F2P Mean", f"{metrics.f2p_mean:.4f}")
            table.add_row("P2P Mean", f"{metrics.p2p_mean:.4f}")
            table.add_row("Pass@1", f"{metrics.pass_at_1:.2%}")
            table.add_row("Pass@5", f"{metrics.pass_at_5:.2%}")
            table.add_row("Num Samples", str(metrics.num_samples))

            self.console.print(table)
        else:
            logger.info("=" * 50)
            logger.info(f"Evaluation Results (Step {metrics.step})")
            logger.info(f"  Reward Mean: {metrics.reward_mean:.4f}")
            logger.info(f"  Resolve Rate: {metrics.resolve_rate:.2%}")
            logger.info(f"  F2P Mean: {metrics.f2p_mean:.4f}")
            logger.info(f"  P2P Mean: {metrics.p2p_mean:.4f}")
            logger.info("=" * 50)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""

        summary = {
            "experiment_name": self.experiment_name,
            "total_steps": len(self.training_history),
            "total_evals": len(self.eval_history),
        }

        if self.training_history:
            summary["final_loss"] = self.training_history[-1].loss
            summary["avg_loss"] = np.mean([m.loss for m in self.training_history])
            summary["avg_reward"] = np.mean([m.reward_mean for m in self.training_history])
            summary["max_reward"] = max(m.reward_mean for m in self.training_history)

        if self.eval_history:
            summary["best_resolve_rate"] = max(m.resolve_rate for m in self.eval_history)
            summary["best_reward"] = max(m.reward_mean for m in self.eval_history)

        return summary

    def save(self):
        """Save all metrics to files."""

        # Save training history
        training_path = self.output_dir / "training_history.json"
        with open(training_path, "w") as f:
            json.dump(
                [asdict(m) for m in self.training_history],
                f,
                indent=2,
            )

        # Save eval history
        eval_path = self.output_dir / "eval_history.json"
        with open(eval_path, "w") as f:
            json.dump(
                [asdict(m) for m in self.eval_history],
                f,
                indent=2,
            )

        # Save custom metrics
        custom_path = self.output_dir / "custom_metrics.json"
        with open(custom_path, "w") as f:
            json.dump(dict(self.custom_metrics), f, indent=2)

        # Save summary
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(self.get_summary(), f, indent=2)

        logger.info(f"Metrics saved to {self.output_dir}")

    def close(self):
        """Close all loggers and save final metrics."""

        self.save()

        if self.wandb_run:
            import wandb
            wandb.finish()

        if self.tb_writer:
            self.tb_writer.close()

        logger.info("MetricsTracker closed")


class RunningAverage:
    """Compute running average over a window."""

    def __init__(self, window: int = 100):
        self.window = window
        self.values: List[float] = []

    def update(self, value: float):
        self.values.append(value)
        if len(self.values) > self.window:
            self.values.pop(0)

    @property
    def avg(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    @property
    def latest(self) -> float:
        if not self.values:
            return 0.0
        return self.values[-1]


def compute_pass_at_k(
    results: List[Dict[str, Any]],
    k: int,
) -> float:
    """
    Compute Pass@K metric.

    Args:
        results: List of evaluation results with 'resolved' field
        k: Number of attempts

    Returns:
        Pass@K score
    """

    if not results:
        return 0.0

    # Group by instance
    instances = defaultdict(list)
    for result in results:
        instance_id = result.get("instance_id", "unknown")
        instances[instance_id].append(result.get("resolved", False))

    # Compute pass@k for each instance
    pass_count = 0
    for instance_id, resolved_list in instances.items():
        n = len(resolved_list)
        c = sum(resolved_list)

        if n < k:
            # Not enough samples, use what we have
            pass_at_k = 1.0 if c > 0 else 0.0
        else:
            # Compute 1 - (n-c choose k) / (n choose k)
            from math import comb
            if c >= k:
                pass_at_k = 1.0
            else:
                pass_at_k = 1.0 - comb(n - c, k) / comb(n, k)

        pass_count += pass_at_k

    return pass_count / len(instances)


if __name__ == "__main__":
    # Test metrics tracker
    logging.basicConfig(level=logging.INFO)

    tracker = MetricsTracker(
        output_dir="./test_logs",
        experiment_name="test_experiment",
        use_wandb=False,
        use_tensorboard=False,
    )

    # Log some training steps
    for step in range(10):
        tracker.log_training_step(
            step=step,
            epoch=0,
            metrics={
                "loss": 1.0 - step * 0.1,
                "policy_loss": 0.8 - step * 0.08,
                "kl_loss": 0.2 - step * 0.02,
                "kl_div": 0.1,
                "reward_mean": step * 0.1,
                "reward_std": 0.1,
                "reward_max": step * 0.15,
                "reward_min": step * 0.05,
                "advantage_mean": 0.0,
                "advantage_std": 1.0,
                "learning_rate": 1e-6,
            }
        )

    # Log eval
    tracker.log_eval_step(
        step=10,
        metrics={
            "reward_mean": 0.5,
            "reward_std": 0.2,
            "resolve_rate": 0.25,
            "f2p_mean": 0.6,
            "p2p_mean": 0.8,
            "num_samples": 20,
        }
    )

    # Print summary
    print(tracker.get_summary())

    tracker.close()
