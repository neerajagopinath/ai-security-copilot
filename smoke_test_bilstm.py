"""
Bi-LSTM Smoke Test + Demo Checkpoint Generator
------------------------------------------------
Runs a very short (2-epoch, tiny-subset) training run to:
  1. Verify the full training pipeline works end-to-end.
  2. Generate a demo checkpoint for API fallback usage.

The generated checkpoint is saved as:
  models/checkpoints/demo_bilstm_model.pt

IMPORTANT LABEL:
  DEMO-TRAINED CHECKPOINT — NOT A FULLY TRAINED FINAL MODEL

This file is safe to run on CPU and completes in < 60 seconds.

Usage:
  python smoke_test_bilstm.py
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.bilstm import BiLSTMVulnerabilityDetector
from src.preprocessing.dataset import VulnerabilityDataset, get_dataloader
from src.training.train_utils import (
    calculate_pos_weight,
    compute_classification_metrics,
    save_checkpoint,
)
from src.utils.seed import set_seed


DEMO_CHECKPOINT_PATH = "models/checkpoints/demo_bilstm_model.pt"
PROCESSED_TRAIN_PATH = "data/processed/train.jsonl"
PROCESSED_VAL_PATH = "data/processed/validation.jsonl"
VOCAB_PATH = "data/processed/vocabulary.json"
CONFIG_PATH = "configs/config.yaml"


def load_jsonl_subset(file_path: str, n_samples: int):
    """Load first n_samples records from a JSONL file."""
    sequences, labels, lengths = [], [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            record = json.loads(line.strip())
            sequences.append(record["input_ids"])
            labels.append(record["label"])
            lengths.append(record["length"])
    return sequences, labels, lengths


def run_smoke_test():
    print("=" * 60)
    print("AI Security Copilot — Bi-LSTM Smoke Test")
    print("=" * 60)
    print()

    # Verify required files
    required = [PROCESSED_TRAIN_PATH, PROCESSED_VAL_PATH, VOCAB_PATH]
    for f in required:
        if not os.path.exists(f):
            print(f"[ERROR] Required file not found: {f}")
            print("Please run preprocessing first:")
            print(
                "  python -m src.preprocessing.preprocess "
                "--train_raw data/raw/train-00000-of-00001.parquet ..."
            )
            sys.exit(1)

    print("[OK] All required data files found.")

    # Set seed
    set_seed(42)
    device = torch.device("cpu")
    print(f"[OK] Device: {device}")

    # Load vocabulary
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    print(f"[OK] Vocabulary size: {vocab_size}")

    # Load SMALL subsets (16 train, 8 val) for demo
    N_TRAIN = 32
    N_VAL = 16
    print(f"\n[1/5] Loading {N_TRAIN} train + {N_VAL} val samples (smoke subset)...")
    train_seqs, train_labels, train_lengths = load_jsonl_subset(PROCESSED_TRAIN_PATH, N_TRAIN)
    val_seqs, val_labels, val_lengths = load_jsonl_subset(PROCESSED_VAL_PATH, N_VAL)
    print(f"[OK] Loaded: train={len(train_seqs)}, val={len(val_seqs)}")

    # Verify both classes present
    train_pos = sum(train_labels)
    train_neg = len(train_labels) - train_pos
    print(f"[OK] Train class distribution: {train_neg} negative, {train_pos} positive")
    if train_pos == 0 or train_neg == 0:
        print("[WARNING] Only one class in smoke subset — increasing subset size...")
        # Try with larger subset
        train_seqs, train_labels, train_lengths = load_jsonl_subset(PROCESSED_TRAIN_PATH, 200)
        val_seqs, val_labels, val_lengths = load_jsonl_subset(PROCESSED_VAL_PATH, 50)
        print(f"[OK] Extended to: train={len(train_seqs)}, val={len(val_seqs)}")

    # Build datasets
    train_dataset = VulnerabilityDataset(train_seqs, train_labels, train_lengths)
    val_dataset = VulnerabilityDataset(val_seqs, val_labels, val_lengths)

    train_loader = get_dataloader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = get_dataloader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # Initialize model (small for speed)
    print("\n[2/5] Initializing Bi-LSTM model...")
    model = BiLSTMVulnerabilityDetector(
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        padding_idx=0,
    )
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Model initialized. Trainable parameters: {param_count:,}")

    # Loss, optimizer
    pos_weight = calculate_pos_weight(train_labels)
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Training loop — 2 epochs only
    EPOCHS = 2
    print(f"\n[3/5] Running {EPOCHS}-epoch smoke training...")
    t_start = time.time()

    history = []
    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["length"]
            targets = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, lengths).squeeze(-1)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * input_ids.size(0)

        epoch_train_loss = train_loss / len(train_loader.dataset)

        # Validate
        import numpy as np
        model.eval()
        val_loss = 0.0
        all_targets, all_logits = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                lengths = batch["length"]
                targets = batch["label"].to(device)
                logits = model(input_ids, lengths).squeeze(-1)
                loss = criterion(logits, targets)
                val_loss += loss.item() * input_ids.size(0)
                all_targets.extend(targets.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        metrics = compute_classification_metrics(
            np.array(all_targets), np.array(all_logits), threshold=0.5
        )

        print(
            f"  Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val F1: {metrics['f1']:.4f}"
        )
        history.append({"epoch": epoch, **metrics})

    elapsed = time.time() - t_start
    print(f"[OK] Smoke training completed in {elapsed:.1f}s")

    # Save demo checkpoint
    print(f"\n[4/5] Saving demo checkpoint...")
    os.makedirs(os.path.dirname(DEMO_CHECKPOINT_PATH), exist_ok=True)

    import yaml
    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)

    final_f1 = history[-1].get("f1", 0.0)
    save_checkpoint(
        path=DEMO_CHECKPOINT_PATH,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=EPOCHS,
        best_f1=final_f1,
        metrics={
            "note": "DEMO-TRAINED CHECKPOINT — NOT A FULLY TRAINED FINAL MODEL",
            "smoke_epochs": EPOCHS,
            "smoke_train_size": len(train_seqs),
            **history[-1],
        },
        config=config,
        seed=42,
        vocab_size=vocab_size,
        pad_idx=0,
    )

    size_kb = os.path.getsize(DEMO_CHECKPOINT_PATH) / 1024
    print(f"[OK] Demo checkpoint saved: {DEMO_CHECKPOINT_PATH} ({size_kb:.1f} KB)")

    # Verify checkpoint can be loaded
    print("\n[5/5] Verifying checkpoint load...")
    loaded_ckpt = torch.load(DEMO_CHECKPOINT_PATH, map_location="cpu")
    assert "model_state_dict" in loaded_ckpt
    assert loaded_ckpt["epoch"] == EPOCHS
    print("[OK] Checkpoint verified successfully.")

    # Test single inference
    model.eval()
    test_ids = torch.randint(1, 100, (1, 32))
    test_len = torch.tensor([32])
    with torch.no_grad():
        logit = model(test_ids, test_len)
        prob = torch.sigmoid(logit).item()
    print(f"[OK] Single inference test: probability={prob:.4f}")

    print()
    print("=" * 60)
    print("SMOKE TEST PASSED [OK]")
    print("=" * 60)
    print(f"Demo checkpoint:     {DEMO_CHECKPOINT_PATH}")
    print(f"Training duration:   {elapsed:.1f}s")
    print(f"Smoke train samples: {len(train_seqs)}")
    print(f"Final val F1 score:  {final_f1:.4f}  (demo only, not production metric)")
    print()
    print("⚠️  IMPORTANT:")
    print("   This checkpoint was trained on a TINY subset for 2 epochs only.")
    print("   It is suitable for API fallback/demo only.")
    print("   Full model performance requires GPU training on complete Devign dataset.")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()
