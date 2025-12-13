"""
Model Loading and Configuration for GigaChat-Lightning Base
"""

import torch
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)


def load_tokenizer(
    model_path: str,
    trust_remote_code: bool = True,
    padding_side: str = "left",
) -> AutoTokenizer:
    """Load tokenizer for GigaChat-Lightning."""

    logger.info(f"Loading tokenizer from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        padding_side=padding_side,
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    return tokenizer


def load_model(
    model_path: str,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    trust_remote_code: bool = True,
    use_flash_attention_2: bool = True,
    use_gradient_checkpointing: bool = True,
    quantization: Optional[str] = None,
    lora_config: Optional[Dict[str, Any]] = None,
) -> AutoModelForCausalLM:
    """
    Load GigaChat-Lightning Base model with optimizations.

    Args:
        model_path: Path to model or HuggingFace model ID
        torch_dtype: Data type (bfloat16, float16, float32)
        device_map: Device mapping strategy
        trust_remote_code: Trust remote code execution
        use_flash_attention_2: Use Flash Attention 2
        use_gradient_checkpointing: Enable gradient checkpointing
        quantization: Quantization type (None, "4bit", "8bit")
        lora_config: LoRA configuration dict

    Returns:
        Loaded model ready for training
    """

    logger.info(f"Loading model from {model_path}")
    logger.info(f"  dtype: {torch_dtype}")
    logger.info(f"  flash_attention_2: {use_flash_attention_2}")
    logger.info(f"  gradient_checkpointing: {use_gradient_checkpointing}")
    logger.info(f"  quantization: {quantization}")

    # Parse dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype_parsed = dtype_map.get(torch_dtype, torch.bfloat16)

    # Quantization config
    quantization_config = None
    if quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype_parsed,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quantization == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    # Model kwargs
    model_kwargs = {
        "pretrained_model_name_or_path": model_path,
        "torch_dtype": torch_dtype_parsed,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }

    # Add Flash Attention 2 if available
    if use_flash_attention_2:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception as e:
            logger.warning(f"Flash Attention 2 not available: {e}")

    # Add quantization if specified
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config

    # Load model
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    # Prepare for training if quantized
    if quantization:
        model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # Apply LoRA if config provided
    if lora_config:
        logger.info(f"Applying LoRA config: {lora_config}")
        peft_config = LoraConfig(
            r=lora_config.get("r", 64),
            lora_alpha=lora_config.get("lora_alpha", 128),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=lora_config.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            ),
            bias=lora_config.get("bias", "none"),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model loaded successfully:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")

    return model


def create_reference_model(
    model: AutoModelForCausalLM,
    device_map: str = "auto",
) -> AutoModelForCausalLM:
    """
    Create a frozen reference model for KL divergence computation.

    For GRPO, we need a reference model that stays fixed during training.
    """

    logger.info("Creating reference model (frozen copy)")

    # Get config and path from original model
    config = model.config

    # Load fresh copy
    ref_model = AutoModelForCausalLM.from_pretrained(
        config._name_or_path,
        torch_dtype=model.dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Freeze all parameters
    for param in ref_model.parameters():
        param.requires_grad = False

    ref_model.eval()

    logger.info("Reference model created and frozen")

    return ref_model


class ModelManager:
    """Manager for model loading, saving, and reference model handling."""

    def __init__(
        self,
        config: Dict[str, Any],
    ):
        self.config = config
        self.model = None
        self.ref_model = None
        self.tokenizer = None

    def setup(self):
        """Load all models and tokenizer."""

        model_config = self.config.get("model", {})

        # Determine model path
        model_path = model_config.get("local_path")
        if not Path(model_path).exists():
            model_path = model_config.get("name")
            logger.info(f"Local model not found, using HuggingFace: {model_path}")

        # Load tokenizer
        self.tokenizer = load_tokenizer(
            model_path,
            trust_remote_code=model_config.get("trust_remote_code", True),
        )

        # Load main model
        self.model = load_model(
            model_path,
            torch_dtype=model_config.get("torch_dtype", "bfloat16"),
            trust_remote_code=model_config.get("trust_remote_code", True),
            use_flash_attention_2=model_config.get("use_flash_attention_2", True),
            use_gradient_checkpointing=self.config.get("hardware", {}).get(
                "use_gradient_checkpointing", True
            ),
        )

        # Create reference model
        self.ref_model = create_reference_model(self.model)

        return self.model, self.ref_model, self.tokenizer

    def save_checkpoint(self, output_dir: str, step: int):
        """Save model checkpoint."""

        checkpoint_dir = Path(output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving checkpoint to {checkpoint_dir}")

        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save config
        import json
        with open(checkpoint_dir / "training_config.json", "w") as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_dir}")

        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir: str):
        """Load model from checkpoint."""

        logger.info(f"Loading checkpoint from {checkpoint_dir}")

        self.tokenizer = load_tokenizer(checkpoint_dir)
        self.model = load_model(
            checkpoint_dir,
            torch_dtype=self.config.get("model", {}).get("torch_dtype", "bfloat16"),
        )

        return self.model, self.tokenizer


if __name__ == "__main__":
    # Test model loading
    logging.basicConfig(level=logging.INFO)

    config = {
        "model": {
            "name": "ai-sage/GigaChat3-10B-A1.8B-base",
            "local_path": "./models/gigachat-lightning-base",
            "torch_dtype": "bfloat16",
            "trust_remote_code": True,
            "use_flash_attention_2": True,
        },
        "hardware": {
            "use_gradient_checkpointing": True,
        }
    }

    manager = ModelManager(config)
    model, ref_model, tokenizer = manager.setup()

    print(f"Model loaded: {type(model)}")
    print(f"Reference model loaded: {type(ref_model)}")
    print(f"Tokenizer loaded: {type(tokenizer)}")
