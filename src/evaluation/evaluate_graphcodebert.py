"""
GraphCodeBERT Evaluation Script
---------------------------------
Evaluates a fine-tuned GraphCodeBERT model on the test set.

Usage:
    python -m src.evaluation.evaluate_graphcodebert \\
        --checkpoint_path models/checkpoints/graphcodebert_epoch_3.pt \\
        --test_path data/processed/test.jsonl \\
        --device cuda
"""

import os
import json
import argparse
import logging
from typing import List, Tuple

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GraphCodeBERT Evaluation"
    )
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--tokenizer_dir", type=str, default=None,
                        help="Directory with saved tokenizer. Defaults to checkpoint dir.")
    parser.add_argument("--test_path", type=str, default="data/processed/test.jsonl")
    parser.add_argument("--model_name", type=str, default="microsoft/graphcodebert-base")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_jsonl(path: str) -> Tuple[List[str], List[int]]:
    sequences, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            token_ids = rec.get("input_ids", [])
            sequences.append(" ".join(map(str, token_ids)))
            labels.append(rec["label"])
    return sequences, labels


def main():
    args = parse_args()

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    )
    from src.models.graphcodebert import (
        load_graphcodebert_checkpoint,
        load_graphcodebert_tokenizer,
        prepare_graphcodebert_dataset,
    )

    # Load model
    logger.info("Loading checkpoint: %s", args.checkpoint_path)
    model, ckpt = load_graphcodebert_checkpoint(
        args.checkpoint_path, args.model_name, args.device
    )

    # Load tokenizer
    tok_dir = args.tokenizer_dir or os.path.dirname(args.checkpoint_path)
    if os.path.exists(os.path.join(tok_dir, "tokenizer_config.json")):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    else:
        tokenizer = load_graphcodebert_tokenizer(args.model_name)

    # Load test data
    logger.info("Loading test data: %s", args.test_path)
    sequences, labels = load_jsonl(args.test_path)
    dataset = prepare_graphcodebert_dataset(
        sequences, labels, tokenizer, args.max_length
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    # Evaluate
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"]

            logits = model(input_ids, attn_mask).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(batch_labels.numpy().tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= 0.5).astype(int)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0

    logger.info("=" * 50)
    logger.info("GraphCodeBERT Test Evaluation Results")
    logger.info("=" * 50)
    logger.info("  Accuracy:  %.4f", acc)
    logger.info("  Precision: %.4f", prec)
    logger.info("  Recall:    %.4f", rec)
    logger.info("  F1-score:  %.4f", f1)
    logger.info("  ROC-AUC:   %.4f", auc)
    logger.info("=" * 50)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
    }


if __name__ == "__main__":
    main()
