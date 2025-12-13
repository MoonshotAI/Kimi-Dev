#!/usr/bin/env python3
"""
Analytics and Visualization for GRPO Training

Generates reports, plots, and analysis of training results.
"""

import json
import logging
import argparse
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


def load_training_history(log_dir: Path) -> List[Dict[str, Any]]:
    """Load training history from JSON file."""

    history_path = log_dir / "training_history.json"
    if not history_path.exists():
        logger.warning(f"Training history not found: {history_path}")
        return []

    with open(history_path, "r") as f:
        return json.load(f)


def load_eval_history(log_dir: Path) -> List[Dict[str, Any]]:
    """Load evaluation history from JSON file."""

    eval_path = log_dir / "eval_history.json"
    if not eval_path.exists():
        logger.warning(f"Eval history not found: {eval_path}")
        return []

    with open(eval_path, "r") as f:
        return json.load(f)


def load_rollouts(log_dir: Path) -> List[Dict[str, Any]]:
    """Load rollouts from JSONL file."""

    rollouts_path = log_dir / "rollouts.jsonl"
    if not rollouts_path.exists():
        logger.warning(f"Rollouts not found: {rollouts_path}")
        return []

    rollouts = []
    with open(rollouts_path, "r") as f:
        for line in f:
            rollouts.append(json.loads(line))

    return rollouts


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute basic statistics for a list of values."""

    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}

    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def analyze_training(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze training metrics."""

    if not history:
        return {}

    # Extract metrics
    losses = [h["loss"] for h in history]
    rewards = [h["reward_mean"] for h in history]
    kl_divs = [h.get("kl_div", 0) for h in history]

    # Compute smoothed curves (exponential moving average)
    def ema(values, alpha=0.1):
        result = []
        current = values[0] if values else 0
        for v in values:
            current = alpha * v + (1 - alpha) * current
            result.append(current)
        return result

    analysis = {
        "total_steps": len(history),
        "loss": compute_statistics(losses),
        "reward": compute_statistics(rewards),
        "kl_div": compute_statistics(kl_divs),
        "loss_trend": {
            "initial": losses[0] if losses else 0,
            "final": losses[-1] if losses else 0,
            "improvement": (losses[0] - losses[-1]) if losses else 0,
        },
        "reward_trend": {
            "initial": rewards[0] if rewards else 0,
            "final": rewards[-1] if rewards else 0,
            "improvement": (rewards[-1] - rewards[0]) if rewards else 0,
        },
        "convergence": {
            "loss_converged": np.std(losses[-100:]) < 0.01 if len(losses) > 100 else False,
            "steps_to_best_reward": int(np.argmax(rewards)) if rewards else 0,
        }
    }

    return analysis


def analyze_evaluation(eval_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze evaluation metrics."""

    if not eval_history:
        return {}

    resolve_rates = [h.get("resolve_rate", 0) for h in eval_history]
    rewards = [h.get("reward_mean", 0) for h in eval_history]
    f2p_scores = [h.get("f2p_mean", 0) for h in eval_history]
    p2p_scores = [h.get("p2p_mean", 0) for h in eval_history]

    analysis = {
        "num_evals": len(eval_history),
        "resolve_rate": {
            "best": max(resolve_rates) if resolve_rates else 0,
            "final": resolve_rates[-1] if resolve_rates else 0,
            "stats": compute_statistics(resolve_rates),
        },
        "reward": compute_statistics(rewards),
        "f2p": compute_statistics(f2p_scores),
        "p2p": compute_statistics(p2p_scores),
        "best_eval": max(eval_history, key=lambda x: x.get("resolve_rate", 0)) if eval_history else None,
    }

    return analysis


def analyze_rollouts(rollouts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze rollout statistics."""

    if not rollouts:
        return {}

    # Group by instance
    by_instance = defaultdict(list)
    for r in rollouts:
        by_instance[r.get("instance_id", "unknown")].append(r)

    # Compute per-instance metrics
    instance_stats = []
    for instance_id, instance_rollouts in by_instance.items():
        rewards = [r["reward"] for r in instance_rollouts]
        resolved = [r["resolved"] for r in instance_rollouts]
        instance_stats.append({
            "instance_id": instance_id,
            "num_rollouts": len(instance_rollouts),
            "best_reward": max(rewards),
            "mean_reward": np.mean(rewards),
            "any_resolved": any(resolved),
            "resolve_rate": sum(resolved) / len(resolved),
        })

    # Overall metrics
    all_rewards = [r["reward"] for r in rollouts]
    all_resolved = [r["resolved"] for r in rollouts]
    all_valid = [r.get("patch_valid", False) for r in rollouts]

    analysis = {
        "total_rollouts": len(rollouts),
        "unique_instances": len(by_instance),
        "rollouts_per_instance": len(rollouts) / len(by_instance) if by_instance else 0,
        "overall": {
            "reward": compute_statistics(all_rewards),
            "resolve_rate": sum(all_resolved) / len(all_resolved) if all_resolved else 0,
            "patch_validity_rate": sum(all_valid) / len(all_valid) if all_valid else 0,
        },
        "per_instance": {
            "any_resolved_rate": sum(1 for s in instance_stats if s["any_resolved"]) / len(instance_stats) if instance_stats else 0,
            "best_instances": sorted(instance_stats, key=lambda x: x["best_reward"], reverse=True)[:5],
            "worst_instances": sorted(instance_stats, key=lambda x: x["best_reward"])[:5],
        },
        "timing": {
            "avg_generation_time": np.mean([r.get("generation_time", 0) for r in rollouts]),
            "avg_eval_time": np.mean([r.get("eval_time", 0) for r in rollouts]),
        }
    }

    return analysis


def create_plots(
    training_history: List[Dict[str, Any]],
    eval_history: List[Dict[str, Any]],
    output_dir: Path,
):
    """Create visualization plots."""

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        logger.warning("matplotlib/seaborn not available, skipping plots")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training loss curve
    if training_history:
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = [h["step"] for h in training_history]
        losses = [h["loss"] for h in training_history]
        ax.plot(steps, losses, alpha=0.3, label="Raw")

        # Smoothed
        window = min(50, len(losses) // 10 + 1)
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(steps[window-1:], smoothed, label="Smoothed")

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        plt.savefig(plots_dir / "training_loss.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 2. Reward curve
    if training_history:
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = [h["step"] for h in training_history]
        rewards = [h["reward_mean"] for h in training_history]
        ax.plot(steps, rewards, alpha=0.3, label="Raw")

        window = min(50, len(rewards) // 10 + 1)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(steps[window-1:], smoothed, label="Smoothed")

        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Reward")
        ax.set_title("Training Reward")
        ax.legend()
        plt.savefig(plots_dir / "training_reward.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 3. Evaluation metrics
    if eval_history:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        eval_steps = [h["step"] for h in eval_history]

        # Resolve rate
        ax = axes[0, 0]
        resolve_rates = [h.get("resolve_rate", 0) for h in eval_history]
        ax.plot(eval_steps, resolve_rates, marker='o')
        ax.set_xlabel("Step")
        ax.set_ylabel("Resolve Rate")
        ax.set_title("Resolution Rate")
        ax.set_ylim(0, 1)

        # Reward
        ax = axes[0, 1]
        rewards = [h.get("reward_mean", 0) for h in eval_history]
        ax.plot(eval_steps, rewards, marker='o', color='green')
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Reward")
        ax.set_title("Evaluation Reward")

        # F2P and P2P
        ax = axes[1, 0]
        f2p = [h.get("f2p_mean", 0) for h in eval_history]
        p2p = [h.get("p2p_mean", 0) for h in eval_history]
        ax.plot(eval_steps, f2p, marker='o', label="F2P")
        ax.plot(eval_steps, p2p, marker='s', label="P2P")
        ax.set_xlabel("Step")
        ax.set_ylabel("Score")
        ax.set_title("F2P and P2P Scores")
        ax.legend()
        ax.set_ylim(0, 1)

        # KL divergence from training
        ax = axes[1, 1]
        if training_history:
            kl_steps = [h["step"] for h in training_history]
            kl_divs = [h.get("kl_div", 0) for h in training_history]
            ax.plot(kl_steps, kl_divs, alpha=0.5)
            ax.set_xlabel("Step")
            ax.set_ylabel("KL Divergence")
            ax.set_title("KL Divergence from Reference")

        plt.tight_layout()
        plt.savefig(plots_dir / "evaluation_metrics.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 4. Reward distribution
    if training_history:
        fig, ax = plt.subplots(figsize=(10, 6))
        rewards = [h["reward_mean"] for h in training_history]
        ax.hist(rewards, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.3f}')
        ax.set_xlabel("Reward")
        ax.set_ylabel("Frequency")
        ax.set_title("Reward Distribution")
        ax.legend()
        plt.savefig(plots_dir / "reward_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"Plots saved to {plots_dir}")


def generate_report(
    training_analysis: Dict[str, Any],
    eval_analysis: Dict[str, Any],
    rollout_analysis: Dict[str, Any],
    output_dir: Path,
) -> str:
    """Generate a markdown report."""

    report = []
    report.append("# GRPO Training Analysis Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Training Summary
    report.append("## Training Summary\n")
    if training_analysis:
        report.append(f"- **Total Steps**: {training_analysis.get('total_steps', 0)}")
        report.append(f"- **Final Loss**: {training_analysis.get('loss', {}).get('mean', 0):.4f}")
        report.append(f"- **Final Reward**: {training_analysis.get('reward', {}).get('mean', 0):.4f}")

        trend = training_analysis.get("reward_trend", {})
        report.append(f"- **Reward Improvement**: {trend.get('initial', 0):.4f} → {trend.get('final', 0):.4f} (+{trend.get('improvement', 0):.4f})")
    else:
        report.append("*No training data available*")

    # Evaluation Summary
    report.append("\n## Evaluation Summary\n")
    if eval_analysis:
        resolve = eval_analysis.get("resolve_rate", {})
        report.append(f"- **Best Resolve Rate**: {resolve.get('best', 0):.2%}")
        report.append(f"- **Final Resolve Rate**: {resolve.get('final', 0):.2%}")
        report.append(f"- **Mean F2P Score**: {eval_analysis.get('f2p', {}).get('mean', 0):.4f}")
        report.append(f"- **Mean P2P Score**: {eval_analysis.get('p2p', {}).get('mean', 0):.4f}")
    else:
        report.append("*No evaluation data available*")

    # Rollout Summary
    report.append("\n## Rollout Summary\n")
    if rollout_analysis:
        report.append(f"- **Total Rollouts**: {rollout_analysis.get('total_rollouts', 0)}")
        report.append(f"- **Unique Instances**: {rollout_analysis.get('unique_instances', 0)}")

        overall = rollout_analysis.get("overall", {})
        report.append(f"- **Overall Resolve Rate**: {overall.get('resolve_rate', 0):.2%}")
        report.append(f"- **Patch Validity Rate**: {overall.get('patch_validity_rate', 0):.2%}")

        per_instance = rollout_analysis.get("per_instance", {})
        report.append(f"- **Instances with Any Resolution**: {per_instance.get('any_resolved_rate', 0):.2%}")
    else:
        report.append("*No rollout data available*")

    # Performance Table
    report.append("\n## Key Metrics\n")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")

    if eval_analysis:
        report.append(f"| Best Resolve Rate | {eval_analysis.get('resolve_rate', {}).get('best', 0):.2%} |")
        report.append(f"| Best Reward | {eval_analysis.get('reward', {}).get('max', 0):.4f} |")

    if rollout_analysis:
        report.append(f"| Total Rollouts | {rollout_analysis.get('total_rollouts', 0)} |")
        report.append(f"| Avg Generation Time | {rollout_analysis.get('timing', {}).get('avg_generation_time', 0):.2f}s |")

    report_text = "\n".join(report)

    # Save report
    report_path = output_dir / "analysis_report.md"
    with open(report_path, "w") as f:
        f.write(report_text)

    logger.info(f"Report saved to {report_path}")

    return report_text


def main():
    """Main analysis function."""

    parser = argparse.ArgumentParser(description="Analyze GRPO training results")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory with training logs")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for analysis")
    parser.add_argument("--no_plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir) if args.output_dir else log_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Analyzing logs from: {log_dir}")

    # Load data
    training_history = load_training_history(log_dir)
    eval_history = load_eval_history(log_dir)
    rollouts = load_rollouts(log_dir)

    logger.info(f"Loaded: {len(training_history)} training steps, {len(eval_history)} evals, {len(rollouts)} rollouts")

    # Analyze
    training_analysis = analyze_training(training_history)
    eval_analysis = analyze_evaluation(eval_history)
    rollout_analysis = analyze_rollouts(rollouts)

    # Save analysis
    analysis = {
        "training": training_analysis,
        "evaluation": eval_analysis,
        "rollouts": rollout_analysis,
    }

    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    # Generate plots
    if not args.no_plots:
        create_plots(training_history, eval_history, output_dir)

    # Generate report
    report = generate_report(
        training_analysis,
        eval_analysis,
        rollout_analysis,
        output_dir,
    )

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    logger.info(f"Analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
