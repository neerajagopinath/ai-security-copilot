"""
GraphCodeBERT Training Script (Phase 2.1B)
-------------------------------
Fine-tunes GraphCodeBERT (microsoft/graphcodebert-base) for binary
vulnerability detection on the Devign dataset.

Features:
- Loads raw source code from Parquet, JSONL, or CSV
- Automatic Mixed Precision (AMP) on CUDA
- Linear learning rate scheduler with warmup
- Comprehensive evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
- Checkpoint resumption capabilities
- Deterministic seeding

Smoke Test:
    The --smoke_test flag runs an entirely OFFLINE pipeline:
      - A tiny randomly initialised RobertaForSequenceClassification (2 layers,
        hidden_size=64) is built locally from RobertaConfig — no download.
      - A lightweight mock tokeniser built from the locally cached graphcodebert
        vocab is used; if the cache is absent, a BertTokenizerFast fallback is
        used instead.  Neither path downloads the 500 MB model weights.
    The smoke test exercises every component of the training loop (DataLoader,
    AMP toggle, scheduler, metrics, checkpoint save/load) without touching the
    Hugging Face Hub at inference time.

Production Training (Google Colab / GPU):
    python -m src.training.train_graphcodebert \
        --train_path data/raw/train-00000-of-00001.parquet \
        --val_path data/raw/validation-00000-of-00001.parquet \
        --output_dir models/checkpoints/graphcodebert_tuned
"""

import os
import sys
import json
import random
import argparse
import logging
import yaml
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Make execution as deterministic as possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphCodeBERT Fine-Tuning")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--train_path", type=str, default="data/raw/train-00000-of-00001.parquet")
    parser.add_argument("--val_path", type=str, default="data/raw/validation-00000-of-00001.parquet")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints/graphcodebert_tuned")
    
    # Overrides (if set)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume_from", type=str, default=None, help="Directory to resume from")
    
    parser.add_argument("--smoke_test", action="store_true", help="Run a fast 2-batch test")
    return parser.parse_args()


def load_raw_data(path: str) -> Tuple[List[str], List[int]]:
    """
    Load generic dataset containing 'func' (source code) and 'target' (label).
    Supports Parquet, JSONL, and CSV.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info("Loading dataset from %s", path)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported dataset format: {path}")

    # For Devign, columns are typically 'func' and 'target'
    if "func" not in df.columns or "target" not in df.columns:
        raise ValueError(f"Dataset at {path} must contain 'func' and 'target' columns. Found: {df.columns.tolist()}")

    sequences = df["func"].astype(str).tolist()
    labels = df["target"].astype(int).tolist()
    return sequences, labels


def _build_offline_smoke_assets():
    """
    Build a tiny randomly-initialised HuggingFace-compatible model + tokeniser
    for smoke testing WITHOUT any network downloads.

    Model: RobertaForSequenceClassification with a minimal config
      - hidden_size=64, intermediate_size=128, num_hidden_layers=2
      - num_labels=1  (binary classification logit, matches production)
      - vocab_size=512  (matches the mock tokeniser below)
      - Weights are randomly initialised — this is purely a pipeline check.

    Tokeniser: a BertTokenizerFast built from a tiny auto-generated vocab.
      This produces well-formed input_ids/attention_mask tensors of the
      correct shape so every DataLoader and collation path is exercised.

    Returns:
        (model, tokenizer)  — both are fully standard HF objects.
    """
    from transformers import (
        RobertaConfig,
        RobertaForSequenceClassification,
        BertTokenizerFast,
    )
    import tempfile, json as _json

    logger.info("[SMOKE] Building offline tiny model (no download).")

    # --- Tiny model ---------------------------------------------------------
    config = RobertaConfig(
        vocab_size=512,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=128,
        num_labels=1,     # single logit, matches BCEWithLogitsLoss
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
    )
    model = RobertaForSequenceClassification(config)

    # --- Tiny tokeniser from auto-generated vocab ---------------------------
    # Build a minimal BPE-compatible vocab and write it to a temp dir so that
    # BertTokenizerFast can load it without touching the network.
    vocab = {f"[tok{i}]": i for i in range(508)}
    vocab.update({"[UNK]": 508, "[PAD]": 509, "[CLS]": 510, "[SEP]": 511})

    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = os.path.join(tmpdir, "vocab.txt")
        with open(vocab_path, "w", encoding="utf-8") as f:
            for token in vocab:
                f.write(token + "\n")
        tokenizer = BertTokenizerFast(
            vocab_file=vocab_path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
        )

    logger.info("[SMOKE] Offline tiny model ready (hidden_size=64, layers=2, vocab=512).")
    return model, tokenizer


def _generate_smoke_sequences(n: int = 12) -> Tuple[List[str], List[int]]:
    """
    Generate synthetic code-like strings for the smoke test so that the
    DataLoader and tokenisation paths are exercised without any real data.
    Labels alternate 0/1 to ensure both classes appear in metrics.
    """
    templates = [
        "void f() {{ char buf[64]; strcpy(buf, input); }}",
        "int add(int a, int b) {{ return a + b; }}",
        "void cmd(char *s) {{ system(s); }}",
        "size_t n = strlen(src); memcpy(dst, src, n);",
    ]
    sequences = [templates[i % len(templates)] for i in range(n)]
    labels = [i % 2 for i in range(n)]
    return sequences, labels


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, clip_norm, use_amp):
    model.train()
    total_loss = 0.0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            loss = criterion(logits, labels)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
            
        total_loss += loss.item() * input_ids.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader.dataset)


def eval_epoch(model, dataloader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                loss = criterion(logits, labels)
                
            total_loss += loss.item() * input_ids.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5  # Happens if only one class is present in a tiny smoke test
        
    metrics = {
        "val_loss": avg_loss,
        "val_accuracy": acc,
        "val_precision": prec,
        "val_recall": rec,
        "val_f1": f1,
        "val_auc": auc
    }
    return metrics


def main():
    args = parse_args()
    
    # 1. Load configuration
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
    gcb_config = config.get("models", {}).get("graphcodebert", {})
    
    seed = gcb_config.get("seed", 42)
    set_seed(seed)
    logger.info("Random seed set to %d", seed)
    
    model_name = gcb_config.get("model_name", "microsoft/graphcodebert-base")
    max_length = gcb_config.get("max_sequence_length", 512)
    epochs = args.epochs or gcb_config.get("fine_tune_epochs", 3)
    batch_size = args.batch_size or 16
    lr = args.lr or gcb_config.get("learning_rate", 2e-5)
    num_workers = gcb_config.get("num_workers", 0)
    pin_memory = gcb_config.get("pin_memory", False)
    use_amp_config = gcb_config.get("mixed_precision", False)
    warmup_ratio = gcb_config.get("warmup_ratio", 0.1)
    resume_from = args.resume_from or gcb_config.get("resume_from", None)
    
    clip_grad_norm = 1.0
    dropout = 0.1
    
    # Device selection and AMP check
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
        use_amp = use_amp_config
    else:
        device = torch.device("cpu")
        use_amp = False
        pin_memory = False
        num_workers = 0  # Safe default for CPU
        logger.warning("CUDA not available or device is cpu. Disabled AMP and pin_memory.")
        if not args.smoke_test:
            logger.warning("Running full GraphCodeBERT fine-tuning on CPU is impractical.")

    logger.info("Device: %s | AMP: %s | Num Workers: %d", device.type.upper(), use_amp, num_workers)

    # 2. Import model components
    from src.models.graphcodebert import (
        get_graphcodebert_model,
        load_graphcodebert_tokenizer,
        prepare_graphcodebert_dataset,
        save_graphcodebert_checkpoint,
        load_graphcodebert_checkpoint
    )
    from transformers import get_linear_schedule_with_warmup

    # 3. Load dataset
    if args.smoke_test:
        # OFFLINE SMOKE PATH: synthetic data, no file I/O, no network calls
        train_sequences, train_labels = _generate_smoke_sequences(n=12)
        val_sequences, val_labels = _generate_smoke_sequences(n=6)
        epochs = 1
        batch_size = 4
        num_workers = 0
        logger.info("[SMOKE] Using synthetic offline dataset.")
    else:
        train_sequences, train_labels = load_raw_data(args.train_path)
        val_sequences, val_labels = load_raw_data(args.val_path)

    logger.info("Dataset sizes: train=%d, val=%d", len(train_sequences), len(val_sequences))

    # 4. Initialization (Tokenizer & DataLoaders)
    if args.smoke_test:
        # OFFLINE SMOKE PATH: tiny local model + mock tokeniser, zero downloads
        smoke_model, tokenizer = _build_offline_smoke_assets()
        max_length = 32   # keep tensors tiny for speed
    else:
        tokenizer = load_graphcodebert_tokenizer(model_name)

    train_dataset = prepare_graphcodebert_dataset(train_sequences, train_labels, tokenizer, max_length)
    val_dataset = prepare_graphcodebert_dataset(val_sequences, val_labels, tokenizer, max_length)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    # 5. Model Loading (with Resume capability)
    if args.smoke_test:
        # OFFLINE SMOKE PATH: use the pre-built tiny model
        model = smoke_model
        model.to(device)
        start_epoch = 1
        best_f1 = -1.0
        logger.info("[SMOKE] AMP enabled: %s", use_amp)
    elif resume_from and os.path.exists(resume_from):
        logger.info("Resuming training from checkpoint: %s", resume_from)
        model, previous_metrics = load_graphcodebert_checkpoint(resume_from, device=str(device))
        start_epoch = previous_metrics.get("epoch", 0) + 1
        best_f1 = previous_metrics.get("val_f1", -1.0)
    else:
        logger.info("Initializing new GraphCodeBERT model: %s", model_name)
        model = get_graphcodebert_model(model_name=model_name, dropout=dropout)
        model.to(device)
        start_epoch = 1
        best_f1 = -1.0

    # 6. Optimizer and Scheduler setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Note: If we were fully resuming state, we would load optimizer/scheduler state dicts here.
    # For now, HF model weights are resumed natively via AutoModel.
    
    # 7. Loss function with pos_weight
    neg_count = sum(1 for l in train_labels if l == 0)
    pos_count = sum(1 for l in train_labels if l == 1)
    pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    logger.info("Training for %d epoch(s) starting from epoch %d, batch_size=%d, lr=%s", epochs, start_epoch, batch_size, lr)

    training_history = []
    metrics_path = os.path.join(args.output_dir, "training_history.json")
    os.makedirs(args.output_dir, exist_ok=True)

    # 8. Training Loop
    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, clip_grad_norm, use_amp)
        val_metrics = eval_epoch(model, val_loader, criterion, device, use_amp)
        
        val_loss = val_metrics["val_loss"]
        val_f1 = val_metrics["val_f1"]
        
        logger.info(
            "Epoch %02d | Train Loss: %.4f | Val Loss: %.4f | Val F1: %.4f | Acc: %.4f | AUC: %.4f",
            epoch, train_loss, val_loss, val_f1, val_metrics["val_accuracy"], val_metrics["val_auc"]
        )
        
        history_entry = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        training_history.append(history_entry)
        
        # Save history incrementally
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(training_history, f, indent=4)

        if val_f1 > best_f1:
            best_f1 = val_f1
            if not args.smoke_test:
                save_graphcodebert_checkpoint(model, tokenizer, args.output_dir, epoch, history_entry)
                logger.info("New best checkpoint saved (F1=%.4f)", best_f1)

    if args.smoke_test:
        # Checkpoint round-trip verification
        import tempfile
        with tempfile.TemporaryDirectory() as ckpt_tmp:
            logger.info("[SMOKE] Verifying checkpoint save/load round-trip...")
            from src.models.graphcodebert import save_graphcodebert_checkpoint, load_graphcodebert_checkpoint
            save_graphcodebert_checkpoint(model, tokenizer, ckpt_tmp, epoch=1, metrics={"val_f1": best_f1})
            _, loaded_metrics = load_graphcodebert_checkpoint(ckpt_tmp, device=str(device))
            assert "val_f1" in loaded_metrics, "Checkpoint metrics missing after round-trip!"
            logger.info("[SMOKE] Checkpoint save/load OK. Loaded metrics: %s", loaded_metrics)
        logger.info("[SMOKE] ✓ Smoke test completed successfully. All components verified offline.")
    else:
        logger.info("Training complete. Best validation F1: %.4f", best_f1)

if __name__ == "__main__":
    main()
