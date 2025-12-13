"""
Rollout Generation for GRPO Training

Generates completions and collects rewards for training.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import torch

logger = logging.getLogger(__name__)


@dataclass
class Rollout:
    """A single rollout (prompt + completion + reward)."""

    instance_id: str
    prompt: str
    completion: str
    reward: float
    resolved: bool
    f2p_score: float
    p2p_score: float
    patch_valid: bool
    generation_time: float
    eval_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutBatch:
    """A batch of rollouts for a single prompt."""

    instance_id: str
    prompt: str
    rollouts: List[Rollout]
    best_reward: float
    best_rollout_idx: int
    mean_reward: float
    std_reward: float


class RolloutGenerator:
    """
    Generates rollouts for GRPO training.

    Handles:
    - Multi-completion generation per prompt
    - Parallel reward computation
    - Rollout storage and logging
    """

    def __init__(
        self,
        model,
        tokenizer,
        reward_fn: callable,
        num_generations: int = 8,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_new_tokens: int = 16384,
        max_seq_length: int = 49152,
        max_workers: int = 10,
        output_dir: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.num_generations = num_generations
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_seq_length
        self.max_workers = max_workers
        self.output_dir = Path(output_dir) if output_dir else None

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.total_rollouts = 0
        self.total_resolved = 0
        self.generation_time_total = 0
        self.eval_time_total = 0

        logger.info(f"RolloutGenerator initialized")
        logger.info(f"  num_generations: {num_generations}")
        logger.info(f"  temperature: {temperature}")

    @torch.no_grad()
    def generate_completions(
        self,
        prompt: str,
        num_generations: Optional[int] = None,
    ) -> Tuple[List[str], float]:
        """
        Generate multiple completions for a prompt.

        Returns:
            (list of completions, generation time)
        """

        num_generations = num_generations or self.num_generations
        start_time = time.time()

        self.model.eval()

        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length - self.max_new_tokens,
        ).to(self.model.device)

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            num_return_sequences=num_generations,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode
        prompt_length = inputs["input_ids"].shape[1]
        completions = []

        for output in outputs:
            completion_ids = output[prompt_length:]
            completion = self.tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
            )
            completions.append(completion)

        generation_time = time.time() - start_time

        return completions, generation_time

    def evaluate_completions(
        self,
        completions: List[str],
        problem_data: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Evaluate completions in parallel.

        Returns:
            (list of results, evaluation time)
        """

        start_time = time.time()
        results = []

        # Parallel evaluation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for i, completion in enumerate(completions):
                future = executor.submit(
                    self._evaluate_single,
                    completion,
                    problem_data,
                )
                futures[future] = i

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    result["index"] = idx
                    results.append(result)
                except Exception as e:
                    logger.error(f"Evaluation error for completion {idx}: {e}")
                    results.append({
                        "index": idx,
                        "reward": -1.0,
                        "resolved": False,
                        "error": str(e),
                    })

        # Sort by index
        results.sort(key=lambda x: x["index"])

        eval_time = time.time() - start_time

        return results, eval_time

    def _evaluate_single(
        self,
        completion: str,
        problem_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate a single completion."""

        result = self.reward_fn(
            problem_data=problem_data,
            completion=completion,
        )

        # Handle different return types
        if isinstance(result, (int, float)):
            return {
                "reward": float(result),
                "resolved": result >= 0.99,
                "f2p_score": 0.0,
                "p2p_score": 0.0,
                "patch_valid": True,
            }
        elif isinstance(result, dict):
            return result
        else:
            # Assume it's a RewardResult dataclass
            return {
                "reward": result.reward,
                "resolved": result.resolved,
                "f2p_score": result.f2p_score,
                "p2p_score": result.p2p_score,
                "patch_valid": result.patch_valid,
            }

    def generate_rollout_batch(
        self,
        prompt: str,
        problem_data: Dict[str, Any],
        num_generations: Optional[int] = None,
    ) -> RolloutBatch:
        """
        Generate a batch of rollouts for a single prompt.

        Args:
            prompt: The prompt text
            problem_data: Problem data for reward computation
            num_generations: Number of completions to generate

        Returns:
            RolloutBatch with all rollouts
        """

        num_generations = num_generations or self.num_generations
        instance_id = problem_data.get("instance_id", "unknown")

        # Generate completions
        completions, gen_time = self.generate_completions(prompt, num_generations)

        # Evaluate completions
        results, eval_time = self.evaluate_completions(completions, problem_data)

        # Create rollouts
        rollouts = []
        for completion, result in zip(completions, results):
            rollout = Rollout(
                instance_id=instance_id,
                prompt=prompt,
                completion=completion,
                reward=result.get("reward", 0.0),
                resolved=result.get("resolved", False),
                f2p_score=result.get("f2p_score", 0.0),
                p2p_score=result.get("p2p_score", 0.0),
                patch_valid=result.get("patch_valid", False),
                generation_time=gen_time / num_generations,
                eval_time=eval_time / num_generations,
                metadata=result,
            )
            rollouts.append(rollout)

        # Update statistics
        self.total_rollouts += len(rollouts)
        self.total_resolved += sum(1 for r in rollouts if r.resolved)
        self.generation_time_total += gen_time
        self.eval_time_total += eval_time

        # Compute batch statistics
        rewards = [r.reward for r in rollouts]
        best_idx = max(range(len(rewards)), key=lambda i: rewards[i])

        batch = RolloutBatch(
            instance_id=instance_id,
            prompt=prompt,
            rollouts=rollouts,
            best_reward=max(rewards),
            best_rollout_idx=best_idx,
            mean_reward=sum(rewards) / len(rewards),
            std_reward=torch.tensor(rewards).std().item(),
        )

        return batch

    def generate_rollouts(
        self,
        prompts: List[str],
        problem_data_list: List[Dict[str, Any]],
        num_generations: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[RolloutBatch]:
        """
        Generate rollouts for multiple prompts.

        Args:
            prompts: List of prompts
            problem_data_list: List of problem data dicts
            num_generations: Completions per prompt
            show_progress: Show progress bar

        Returns:
            List of RolloutBatch objects
        """

        batches = []

        iterator = zip(prompts, problem_data_list)
        if show_progress:
            iterator = tqdm(
                list(iterator),
                desc="Generating rollouts",
                total=len(prompts),
            )

        for prompt, problem_data in iterator:
            batch = self.generate_rollout_batch(
                prompt=prompt,
                problem_data=problem_data,
                num_generations=num_generations,
            )
            batches.append(batch)

            # Log progress
            if show_progress:
                iterator.set_postfix({
                    "reward": f"{batch.mean_reward:.3f}",
                    "resolved": f"{self.total_resolved}/{self.total_rollouts}",
                })

        return batches

    def save_rollouts(
        self,
        batches: List[RolloutBatch],
        filename: str = "rollouts.jsonl",
    ):
        """Save rollouts to JSONL file."""

        if not self.output_dir:
            logger.warning("No output_dir specified, skipping save")
            return

        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            for batch in batches:
                for rollout in batch.rollouts:
                    record = {
                        "instance_id": rollout.instance_id,
                        "prompt_length": len(rollout.prompt),
                        "completion": rollout.completion,
                        "reward": rollout.reward,
                        "resolved": rollout.resolved,
                        "f2p_score": rollout.f2p_score,
                        "p2p_score": rollout.p2p_score,
                        "patch_valid": rollout.patch_valid,
                        "generation_time": rollout.generation_time,
                        "eval_time": rollout.eval_time,
                    }
                    f.write(json.dumps(record) + "\n")

        logger.info(f"Saved {sum(len(b.rollouts) for b in batches)} rollouts to {filepath}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get rollout generation statistics."""

        return {
            "total_rollouts": self.total_rollouts,
            "total_resolved": self.total_resolved,
            "resolve_rate": self.total_resolved / max(1, self.total_rollouts),
            "avg_generation_time": self.generation_time_total / max(1, self.total_rollouts),
            "avg_eval_time": self.eval_time_total / max(1, self.total_rollouts),
        }


class RolloutBuffer:
    """
    Buffer for storing rollouts during training.

    Supports prioritized sampling based on reward.
    """

    def __init__(
        self,
        capacity: int = 10000,
        prioritized: bool = True,
    ):
        self.capacity = capacity
        self.prioritized = prioritized
        self.buffer: List[Rollout] = []

    def add(self, rollout: Rollout):
        """Add a rollout to the buffer."""

        self.buffer.append(rollout)

        if len(self.buffer) > self.capacity:
            # Remove oldest
            self.buffer.pop(0)

    def add_batch(self, batch: RolloutBatch):
        """Add all rollouts from a batch."""

        for rollout in batch.rollouts:
            self.add(rollout)

    def sample(self, n: int) -> List[Rollout]:
        """Sample n rollouts from the buffer."""

        import random

        if len(self.buffer) < n:
            return self.buffer.copy()

        if self.prioritized:
            # Prioritize by reward (higher reward = higher probability)
            weights = [max(0.1, r.reward + 1) for r in self.buffer]
            total = sum(weights)
            probs = [w / total for w in weights]
            indices = random.choices(range(len(self.buffer)), weights=probs, k=n)
            return [self.buffer[i] for i in indices]
        else:
            return random.sample(self.buffer, n)

    def get_best(self, n: int = 10) -> List[Rollout]:
        """Get top n rollouts by reward."""

        sorted_buffer = sorted(self.buffer, key=lambda r: r.reward, reverse=True)
        return sorted_buffer[:n]

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()


if __name__ == "__main__":
    # Test rollout generation
    logging.basicConfig(level=logging.INFO)

    # Mock model and tokenizer for testing
    class MockModel:
        device = "cpu"

        def eval(self):
            pass

        def generate(self, **kwargs):
            import torch
            # Return mock outputs
            n = kwargs.get("num_return_sequences", 1)
            return [torch.tensor([1, 2, 3, 4, 5]) for _ in range(n)]

    class MockTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, text, **kwargs):
            import torch
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, ids, **kwargs):
            return "Mock completion"

    def mock_reward_fn(problem_data, completion):
        import random
        return random.random()

    generator = RolloutGenerator(
        model=MockModel(),
        tokenizer=MockTokenizer(),
        reward_fn=mock_reward_fn,
        num_generations=4,
        output_dir="./test_rollouts",
    )

    # Generate a batch
    batch = generator.generate_rollout_batch(
        prompt="Test prompt",
        problem_data={"instance_id": "test-001"},
    )

    print(f"Generated {len(batch.rollouts)} rollouts")
    print(f"Best reward: {batch.best_reward:.4f}")
    print(f"Mean reward: {batch.mean_reward:.4f}")
    print(f"Statistics: {generator.get_statistics()}")
