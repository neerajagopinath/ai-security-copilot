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

class GraphCodeBERTVulnerabilityClassifier(nn.Module):
    """
    Binary vulnerability classifier built on top of GraphCodeBERT
    (microsoft/graphcodebert-base).

    Architecture:
        Source Code
        → HuggingFace GraphCodeBERT tokenizer
        → GraphCodeBERT transformer encoder
        → [CLS] token representation
        → Dropout
        → Linear classification head
        → Vulnerability logit (binary)
    """

    def __init__(
        self,
        model_name: str = "microsoft/graphcodebert-base",
        num_labels: int = 1,
        dropout: float = 0.1,
        hidden_size: int = 768,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.hidden_size = hidden_size

        # Lazy import: only load transformers when the class is instantiated
        try:
            from transformers import RobertaModel
            self.encoder = RobertaModel.from_pretrained(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load GraphCodeBERT backbone '{model_name}'. "
                f"Ensure transformers is installed and you have an internet "
                f"connection (or a local cache). Error: {exc}"
            ) from exc

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len) token IDs.
            attention_mask: (batch_size, seq_len) attention mask.
            token_type_ids: optional token type IDs (not used by RoBERTa).

        Returns:
            logits: (batch_size, 1) raw classification logit.
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Use [CLS] token representation (first token)
        cls_repr = outputs.last_hidden_state[:, 0, :]
        cls_repr = self.dropout(cls_repr)
        logits = self.classifier(cls_repr)
        return logits


# ---------------------------------------------------------------------------
# Tokenizer loader
# ---------------------------------------------------------------------------

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
    model: GraphCodeBERTVulnerabilityClassifier,
    tokenizer,
    output_dir: str,
    epoch: int,
    metrics: Dict[str, Any],
) -> None:
    """Save GraphCodeBERT fine-tuned checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    # Save model state dict
    ckpt_path = os.path.join(output_dir, f"graphcodebert_epoch_{epoch}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "model_name": model.model_name,
        },
        ckpt_path,
    )
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    logger.info("Saved GraphCodeBERT checkpoint to %s", ckpt_path)


def load_graphcodebert_checkpoint(
    checkpoint_path: str,
    model_name: str = "microsoft/graphcodebert-base",
    device: str = "cpu",
) -> Tuple[GraphCodeBERTVulnerabilityClassifier, Dict[str, Any]]:
    """
    Load a fine-tuned GraphCodeBERT checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        model_name: Original base model name.
        device: Target device ('cpu' or 'cuda').

    Returns:
        Tuple of (model, checkpoint_dict).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = GraphCodeBERTVulnerabilityClassifier(
        model_name=checkpoint.get("model_name", model_name)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        "Loaded GraphCodeBERT checkpoint (epoch %d) from %s",
        checkpoint.get("epoch", "?"),
        checkpoint_path,
    )
    return model, checkpoint
