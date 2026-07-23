"""
GraphCodeBERT Vulnerability Classifier
---------------------------------------
Wraps microsoft/graphcodebert-base for binary vulnerability detection.

IMPORTANT: Full fine-tuning of GraphCodeBERT requires a GPU and
significant compute time. This module provides all infrastructure
for fine-tuning and inference but does NOT perform automatic
full fine-tuning on CPU.

Usage:
  - Training: python -m src.training.train_graphcodebert --help
  - Inference: Use GraphCodeBERTPredictor class below
"""

import os
import logging
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

def get_graphcodebert_model(
    model_name: str = "microsoft/graphcodebert-base",
    num_labels: int = 1,
    dropout: float = 0.1,
):
    """
    Returns a HuggingFace AutoModelForSequenceClassification instance.
    This provides native support for save_pretrained() and from_pretrained().
    """
    from transformers import AutoModelForSequenceClassification
    
    logger.info("Loading GraphCodeBERT sequence classifier backbone: %s", model_name)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )
        return model
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load GraphCodeBERT backbone '{model_name}'. "
            f"Ensure transformers is installed and you have an internet "
            f"connection (or a local cache). Error: {exc}"
        ) from exc


def load_graphcodebert_tokenizer(model_name: str = "microsoft/graphcodebert-base"):
    """
    Load the GraphCodeBERT tokenizer from HuggingFace Hub.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        AutoTokenizer instance.
    """
    from transformers import AutoTokenizer

    logger.info("Loading GraphCodeBERT tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("Tokenizer loaded. Vocab size: %d", tokenizer.vocab_size)
    return tokenizer


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_graphcodebert_dataset(
    code_samples: list,
    labels: list,
    tokenizer,
    max_length: int = 512,
) -> "torch.utils.data.Dataset":
    """
    Tokenize raw code strings and wrap them in a PyTorch Dataset.

    Args:
        code_samples: list of raw source code strings.
        labels: list of binary integer labels (0 or 1).
        tokenizer: HuggingFace tokenizer.
        max_length: maximum token sequence length.

    Returns:
        GraphCodeBERTDataset instance.
    """
    return GraphCodeBERTDataset(code_samples, labels, tokenizer, max_length)


class GraphCodeBERTDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for GraphCodeBERT fine-tuning."""

    def __init__(self, code_samples, labels, tokenizer, max_length: int = 512):
        self.encodings = tokenizer(
            code_samples,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_graphcodebert_checkpoint(
    model,
    tokenizer,
    output_dir: str,
    epoch: int,
    metrics: Dict[str, Any],
) -> None:
    """Save GraphCodeBERT fine-tuned checkpoint using HuggingFace format."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and tokenizer natively
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training metrics to a JSON file
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        import json
        metrics["epoch"] = epoch
        json.dump(metrics, f, indent=4)
        
    logger.info("Saved GraphCodeBERT checkpoint to %s", output_dir)


def load_graphcodebert_checkpoint(
    checkpoint_dir: str,
    device: str = "cpu",
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a fine-tuned GraphCodeBERT checkpoint.

    Args:
        checkpoint_dir: Path to the HuggingFace saved model directory.
        device: Target device ('cpu' or 'cuda').

    Returns:
        Tuple of (model, metrics_dict).
    """
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()

    # Attempt to load metrics if available
    metrics = {}
    metrics_path = os.path.join(checkpoint_dir, "training_metrics.json")
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    logger.info(
        "Loaded GraphCodeBERT checkpoint from %s (epoch %s)",
        checkpoint_dir, metrics.get("epoch", "?")
    )
    return model, metrics
