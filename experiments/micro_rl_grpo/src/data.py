"""
Data Loading and Processing for SWE-bench

Handles loading SWE-bench data and preparing it for GRPO training.
"""

import json
import logging
import random
from typing import Dict, List, Optional, Any, Iterator
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class SWEBenchInstance:
    """A single SWE-bench instance."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str
    created_at: str
    patch: str
    test_patch: str
    FAIL_TO_PASS: List[str]
    PASS_TO_PASS: List[str]
    environment_setup_commit: str
    version: str

    # Computed fields
    file_contents: Optional[Dict[str, str]] = None
    repo_structure: Optional[str] = None


class SWEBenchDataset(Dataset):
    """
    Dataset for SWE-bench instances.

    Loads and processes SWE-bench data for GRPO training.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        dataset_name: str = "princeton-nlp/SWE-bench_Verified",
        split: str = "test",
        subset_size: Optional[int] = None,
        repo_structure_path: Optional[str] = None,
        seed: int = 42,
        max_problem_length: int = 10000,
    ):
        """
        Initialize SWE-bench dataset.

        Args:
            data_path: Path to local JSON file (if available)
            dataset_name: HuggingFace dataset name
            split: Dataset split
            subset_size: Number of instances to use (None = all)
            repo_structure_path: Path to preprocessed repo structures
            seed: Random seed for shuffling
            max_problem_length: Maximum problem statement length
        """

        self.data_path = data_path
        self.dataset_name = dataset_name
        self.split = split
        self.subset_size = subset_size
        self.repo_structure_path = repo_structure_path
        self.seed = seed
        self.max_problem_length = max_problem_length

        # Load data
        self.instances = self._load_data()

        logger.info(f"Loaded {len(self.instances)} SWE-bench instances")

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load SWE-bench data from file or HuggingFace."""

        # Try local file first
        if self.data_path and Path(self.data_path).exists():
            logger.info(f"Loading data from local file: {self.data_path}")
            with open(self.data_path, "r") as f:
                data = json.load(f)
        else:
            # Load from HuggingFace
            logger.info(f"Loading data from HuggingFace: {self.dataset_name}")
            dataset = load_dataset(self.dataset_name, split=self.split)
            data = [dict(item) for item in dataset]

        # Shuffle and subset
        random.seed(self.seed)
        random.shuffle(data)

        if self.subset_size:
            data = data[:self.subset_size]

        # Process instances
        processed = []
        for item in data:
            # Parse FAIL_TO_PASS and PASS_TO_PASS
            f2p = item.get("FAIL_TO_PASS", "[]")
            p2p = item.get("PASS_TO_PASS", "[]")

            if isinstance(f2p, str):
                try:
                    f2p = json.loads(f2p)
                except:
                    f2p = []

            if isinstance(p2p, str):
                try:
                    p2p = json.loads(p2p)
                except:
                    p2p = []

            item["FAIL_TO_PASS"] = f2p
            item["PASS_TO_PASS"] = p2p

            # Truncate problem statement if needed
            if len(item.get("problem_statement", "")) > self.max_problem_length:
                item["problem_statement"] = item["problem_statement"][:self.max_problem_length] + "..."

            processed.append(item)

        return processed

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single instance."""
        return self.instances[idx]

    def get_instance_by_id(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get instance by ID."""
        for instance in self.instances:
            if instance.get("instance_id") == instance_id:
                return instance
        return None

    def split_train_eval(
        self,
        train_ratio: float = 0.8,
    ) -> tuple:
        """Split dataset into train and eval sets."""

        n_train = int(len(self.instances) * train_ratio)

        train_data = self.instances[:n_train]
        eval_data = self.instances[n_train:]

        train_dataset = SWEBenchDatasetFromList(train_data)
        eval_dataset = SWEBenchDatasetFromList(eval_data)

        logger.info(f"Split: {len(train_data)} train, {len(eval_data)} eval")

        return train_dataset, eval_dataset


class SWEBenchDatasetFromList(Dataset):
    """Dataset wrapper for a list of instances."""

    def __init__(self, instances: List[Dict[str, Any]]):
        self.instances = instances

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.instances[idx]


class PromptBuilder:
    """
    Builds prompts for the two-stage agentless framework.

    Stage 1: File Localization
    Stage 2: Code Edit
    """

    FILE_LOCALIZATION_TEMPLATE = """You are a software engineer tasked with identifying the files that need to be modified to fix a bug.

## Repository Structure
{repo_structure}

## Problem Statement
{problem_statement}

## Instructions
Based on the problem statement and repository structure, identify the most relevant files that need to be modified.
List up to {max_files} files, ordered by relevance (most relevant first).

## Output Format
Return a JSON list of file paths:
```json
["path/to/file1.py", "path/to/file2.py", ...]
```

## Your Response:
"""

    CODE_EDIT_TEMPLATE = """You are a software engineer tasked with fixing a bug in a codebase.

## Problem Statement
{problem_statement}

## Relevant File Contents
{file_contents}

## Instructions
Fix the bug described in the problem statement. Use the SEARCH/REPLACE format for your changes.

## Output Format
For each file that needs to be modified, use this format:

### path/to/file.py
<<<<<<< SEARCH
[exact code to find]
=======
[replacement code]
>>>>>>> REPLACE

You may make multiple changes to the same file or different files.
Only include the minimal changes necessary to fix the bug.

## Your Fix:
"""

    COMBINED_TEMPLATE = """You are a software engineer tasked with fixing a bug in a codebase.

## Repository Structure
{repo_structure}

## Problem Statement
{problem_statement}

## Relevant File Contents
{file_contents}

## Instructions
1. First, identify which files need to be modified based on the problem statement
2. Then, make the necessary code changes to fix the bug

## Output Format
Use the SEARCH/REPLACE format for your changes:

### path/to/file.py
<<<<<<< SEARCH
[exact code to find]
=======
[replacement code]
>>>>>>> REPLACE

Only include the minimal changes necessary to fix the bug.

## Your Fix:
"""

    def __init__(
        self,
        max_files: int = 5,
        include_repo_structure: bool = True,
        use_combined_prompt: bool = True,
    ):
        self.max_files = max_files
        self.include_repo_structure = include_repo_structure
        self.use_combined_prompt = use_combined_prompt

    def build_localization_prompt(
        self,
        instance: Dict[str, Any],
        repo_structure: str,
    ) -> str:
        """Build prompt for file localization stage."""

        return self.FILE_LOCALIZATION_TEMPLATE.format(
            repo_structure=repo_structure,
            problem_statement=instance["problem_statement"],
            max_files=self.max_files,
        )

    def build_edit_prompt(
        self,
        instance: Dict[str, Any],
        file_contents: Dict[str, str],
    ) -> str:
        """Build prompt for code edit stage."""

        # Format file contents
        formatted_contents = ""
        for file_path, content in file_contents.items():
            formatted_contents += f"\n### {file_path}\n```\n{content}\n```\n"

        return self.CODE_EDIT_TEMPLATE.format(
            problem_statement=instance["problem_statement"],
            file_contents=formatted_contents,
        )

    def build_combined_prompt(
        self,
        instance: Dict[str, Any],
        repo_structure: str,
        file_contents: Dict[str, str],
    ) -> str:
        """Build combined prompt for single-pass generation."""

        # Format file contents
        formatted_contents = ""
        for file_path, content in file_contents.items():
            formatted_contents += f"\n### {file_path}\n```\n{content}\n```\n"

        return self.COMBINED_TEMPLATE.format(
            repo_structure=repo_structure,
            problem_statement=instance["problem_statement"],
            file_contents=formatted_contents,
        )

    def build_prompt(
        self,
        instance: Dict[str, Any],
        repo_structure: Optional[str] = None,
        file_contents: Optional[Dict[str, str]] = None,
    ) -> str:
        """Build appropriate prompt based on configuration."""

        if self.use_combined_prompt:
            return self.build_combined_prompt(
                instance=instance,
                repo_structure=repo_structure or "",
                file_contents=file_contents or {},
            )
        else:
            # For two-stage, start with localization
            if file_contents:
                return self.build_edit_prompt(instance, file_contents)
            else:
                return self.build_localization_prompt(instance, repo_structure or "")


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for DataLoader.

    Groups instances into batches with prompts and problem data.
    """

    prompt_builder = PromptBuilder()

    prompts = []
    problem_data = []

    for instance in batch:
        # Build prompt
        prompt = prompt_builder.build_prompt(
            instance=instance,
            repo_structure=instance.get("repo_structure", ""),
            file_contents=instance.get("file_contents", {}),
        )

        prompts.append(prompt)
        problem_data.append(instance)

    return {
        "prompts": prompts,
        "problem_data": problem_data,
    }


def create_dataloaders(
    config: Dict[str, Any],
) -> tuple:
    """
    Create train and eval dataloaders.

    Args:
        config: Configuration dict

    Returns:
        (train_dataloader, eval_dataloader)
    """

    data_config = config.get("data", {})
    grpo_config = config.get("grpo", {})

    # Load dataset
    dataset = SWEBenchDataset(
        data_path=data_config.get("data_path"),
        dataset_name=data_config.get("dataset", "princeton-nlp/SWE-bench_Verified"),
        subset_size=data_config.get("subset_size"),
        seed=config.get("experiment", {}).get("seed", 42),
    )

    # Split into train/eval
    train_ratio = data_config.get("train_split", 0.8)
    train_dataset, eval_dataset = dataset.split_train_eval(train_ratio)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=grpo_config.get("batch_size", 2),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=grpo_config.get("batch_size", 2),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    logger.info(f"Created dataloaders:")
    logger.info(f"  Train batches: {len(train_dataloader)}")
    logger.info(f"  Eval batches: {len(eval_dataloader)}")

    return train_dataloader, eval_dataloader


if __name__ == "__main__":
    # Test data loading
    logging.basicConfig(level=logging.INFO)

    dataset = SWEBenchDataset(
        subset_size=10,
    )

    print(f"Dataset size: {len(dataset)}")

    # Test an instance
    instance = dataset[0]
    print(f"Instance ID: {instance['instance_id']}")
    print(f"Repo: {instance['repo']}")
    print(f"Problem: {instance['problem_statement'][:200]}...")

    # Test prompt builder
    builder = PromptBuilder()
    prompt = builder.build_prompt(instance)
    print(f"Prompt length: {len(prompt)}")
