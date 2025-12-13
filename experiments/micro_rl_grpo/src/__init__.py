"""
Micro-RL GRPO Experiment
========================

GRPO (Group Relative Policy Optimization) training for code repair
using GigaChat-Lightning Base model on SWE-bench.

Following the Kimi-Dev paradigm:
1. File Localization
2. Code Edit (SEARCH/REPLACE format)
3. Test-based reward signal
"""

__version__ = "0.1.0"
__author__ = "Kimi-Dev Experiment"

from .model import load_model, load_tokenizer
from .grpo_trainer import GRPOTrainer
from .reward import RewardComputer
from .rollout import RolloutGenerator
from .data import SWEBenchDataset
from .metrics import MetricsTracker

__all__ = [
    "load_model",
    "load_tokenizer",
    "GRPOTrainer",
    "RewardComputer",
    "RolloutGenerator",
    "SWEBenchDataset",
    "MetricsTracker",
]
