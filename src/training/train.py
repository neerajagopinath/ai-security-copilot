import os
# ── OpenMP / MKL thread policy ────────────────────────────────────────────────
# MUST be set BEFORE torch imports any OMP/MKL library.
# KMP_BLOCKTIME=0  : OMP worker threads yield immediately when idle instead of
#                    busy-spinning. This helps prevent excessive CPU usage
#                    between batches and reduces thermal throttling on
#                    power-constrained systems.
# OMP_WAIT_POLICY  : PASSIVE is equivalent — threads yield to OS scheduler.
# MKL_DYNAMIC=FALSE: Disable MKL's own thread-count auto-tuning which can
#                    override torch.set_num_threads() at runtime.
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
# ─────────────────────────────────────────────────────────────────────────────
import json
import yaml
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib
# Use non-interactive backend for Matplotlib to avoid Tkinter display issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.seed import set_seed
from src.preprocessing.loader import load_data
from src.preprocessing.dataset import VulnerabilityDataset, get_dataloader
from src.models.bilstm import BiLSTMVulnerabilityDetector
from src.training.train_utils import (
    calculate_pos_weight, 
    compute_classification_metrics, 
    save_checkpoint, 
    load_checkpoint
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Security Copilot - Bi-LSTM Training Pipeline")
    parser.add_argument('--config', type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument('--epochs', type=int, default=None, help="Override epochs count")
    parser.add_argument('--batch_size', type=int, default=None, help="Override batch size")
    parser.add_argument('--lr', type=float, default=None, help="Override learning rate")
    parser.add_argument('--device', type=str, default=None, help="Override device (cpu or cuda)")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument('--smoke_test', action='store_true', help="Execute a fast, single-batch dry-run")
    parser.add_argument('--threads', type=int, default=None,
                        help="CPU thread count for PyTorch. Default: auto-detected. "
                             "Use this to limit thread usage in resource-constrained environments.")
    return parser.parse_args()

def load_jsonl_data(file_path: str) -> tuple[list[list[int]], list[int], list[int]]:
    """Load preprocessed sequences, labels, and lengths from JSONL file."""
    sequences = []
    labels = []
    lengths = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            sequences.append(record["input_ids"])
            labels.append(record["label"])
            lengths.append(record["length"])
            
    return sequences, labels, lengths

def plot_curves(history: list[dict], figures_dir: str) -> None:
    """Generate and save Loss, F1-score, and Learning Rate curves."""
    os.makedirs(figures_dir, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    
    # 1. Loss curves (Train vs Validation)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h["train_loss"] for h in history], label="Train Loss", color="#3b82f6", linewidth=2)
    plt.plot(epochs, [h["val_loss"] for h in history], label="Val Loss", color="#ef4444", linewidth=2)
    plt.title("Training and Validation Loss", fontsize=14, fontweight="bold", pad=10)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "loss_curves.png"), dpi=300)
    plt.close()
    
    # 2. Validation F1-score curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h["val_f1"] for h in history], label="Validation F1", color="#10b981", linewidth=2)
    plt.title("Validation F1-score Progress", fontsize=14, fontweight="bold", pad=10)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("F1-score", fontsize=12)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "f1_score_curve.png"), dpi=300)
    plt.close()
    
    # 3. Learning rate curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h["lr"] for h in history], label="Learning Rate", color="#8b5cf6", linewidth=2)
    plt.title("Learning Rate Decay schedule", fontsize=14, fontweight="bold", pad=10)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "lr_curve.png"), dpi=300)
    plt.close()
    
    print(f"[OK] Training plots generated and saved in: {os.path.abspath(figures_dir)}")

def main():
    args = parse_args()
    
    # Resolve absolute project root so all paths are correct regardless of CWD.
    # train.py lives at <project_root>/src/training/train.py  -> .parent.parent.parent
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    
    # 1. Load project configurations
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Resolve every config path to an absolute location anchored at PROJECT_ROOT
    for key in config.get("paths", {}):
        config["paths"][key] = str(PROJECT_ROOT / config["paths"][key])
        
    # Merge CLI Overrides
    epochs = args.epochs if args.epochs is not None else config["training"]["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else config["training"]["batch_size"]
    lr = args.lr if args.lr is not None else config["training"]["learning_rate"]
    
    # 2. Set Seed and Device Configuration
    set_seed(config["project"]["seed"])
    
    device_name = args.device if args.device is not None else config["training"]["device"]
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if device_name == "cuda":
            print("[Warning] CUDA was requested but is not available on this hardware. Defaulting to CPU.")
        device = torch.device("cpu")

        # ── CPU thread configuration ───────────────────────────────────────────
        # PyTorch defaults to using all logical cores. In some environments,
        # specifying fewer threads can prevent synchronization overhead and
        # improve overall throughput. We allow an explicit override via --threads.
        try:
            if args.threads is not None:
                chosen_threads = max(1, args.threads)
                torch.set_num_threads(chosen_threads)
            
            # Restrict inter-op parallelism during sequential training to save resources
            torch.set_num_interop_threads(1)
            
            print(f"  CPU threads: {torch.get_num_threads()}  (interop: {torch.get_num_interop_threads()})  "
                  f"[KMP_BLOCKTIME=0, OMP_WAIT_POLICY=PASSIVE]")
        except Exception as e:
            print(f"  [Warning] Could not configure CPU threads optimally: {e}")

    print(f"\nTraining Configured Device: {device.type.upper()}")
    if device.type == "cuda":
        print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
        
    # 3. Load Vocab size & Padding ID
    vocab_path = os.path.join(config["paths"]["processed_data_dir"], "vocabulary.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    pad_idx = 0  # Reused from configuration/tokenizer defaults
    
    # 4. Load Processed Datasets
    print("\n[1/5] Loading datasets...")
    train_path = os.path.join(config["paths"]["processed_data_dir"], "train.jsonl")
    val_path = os.path.join(config["paths"]["processed_data_dir"], "validation.jsonl")
    
    train_seqs, train_labels, train_lengths = load_jsonl_data(train_path)
    val_seqs, val_labels, val_lengths = load_jsonl_data(val_path)
    
    print(f"Loaded training samples:   {len(train_seqs)}")
    print(f"Loaded validation samples: {len(val_seqs)}")
    
    # Calculate pos_weight dynamically
    pos_weight = calculate_pos_weight(train_labels)
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float, device=device)
    print(f"Calculated training class pos_weight: {pos_weight:.4f}")
    
    # Check for smoke test override
    if args.smoke_test:
        print("\n!!! Smoke Test Activated: Slicing dataset to tiny subset !!!")
        train_seqs, train_labels, train_lengths = train_seqs[:8], train_labels[:8], train_lengths[:8]
        val_seqs, val_labels, val_lengths = val_seqs[:4], val_labels[:4], val_lengths[:4]
        epochs = 1
        batch_size = 4
        print(f"Smoke test dataset sizes: train={len(train_seqs)}, validation={len(val_seqs)}")
        
    # Construct PyTorch datasets
    train_dataset = VulnerabilityDataset(train_seqs, train_labels, train_lengths)
    val_dataset = VulnerabilityDataset(val_seqs, val_labels, val_lengths)
    
    train_loader = get_dataloader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=config["training"]["num_workers"]
    )
    val_loader = get_dataloader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=config["training"]["num_workers"]
    )
    
    # 5. Initialize Model, Optimizer, Loss, and Scheduler
    print("\n[2/5] Initializing Model Architecture...")
    lstm_params = config["models"]["bilstm"]
    model = BiLSTMVulnerabilityDetector(
        vocab_size=vocab_size,
        embedding_dim=lstm_params["embedding_dim"],
        hidden_dim=lstm_params["hidden_dim"],
        num_layers=lstm_params["num_layers"],
        dropout=lstm_params["dropout"],
        padding_idx=pad_idx
    )
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=config["training"]["weight_decay"]
    )
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    # Plateau learning rate scheduler (monitors validation F1 score, so mode is 'max')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=config["training"]["scheduler_factor"], 
        patience=config["training"]["scheduler_patience"]
    )
    
    # 6. Initialize Training Variables / Resume state
    start_epoch = 1
    best_f1 = -1.0
    best_val_loss = float('inf')
    history = []
    early_stopping_counter = 0
    early_stopping_patience = config["training"]["early_stopping_patience"]
    
    if args.resume:
        print("\n[3/5] Resuming training from checkpoint...")
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint["best_f1"]
        best_val_loss = checkpoint["metrics"].get("val_loss", float('inf'))
        print(f"Resuming at Epoch: {start_epoch} (Best validation F1: {best_f1:.4f})")
        
        # Reload history file if existing
        history_path = os.path.join(config["paths"]["metrics_dir"], "training_history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
            # Filter history to keep only epochs up to checkpoint epoch
            history = [h for h in history if h["epoch"] < start_epoch]
    else:
        print("\n[3/5] Starting fresh training run...")
        
    # 7. Training & Validation Loop
    print(f"\n[4/5] Training for {epochs} epochs on device: {device.type}...")
    
    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        
        # --- TRAINING CYCLE ---
        model.train()
        # Accumulate loss as a tensor to avoid 340 synchronisation barriers per epoch.
        # loss.item() forces a CPU sync on every call; instead we accumulate the
        # tensor and call .item() exactly once at the end of the epoch.
        train_loss_tensor = torch.tensor(0.0)
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["length"]  # CPU tensor
            targets = batch["label"].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward
            logits = model(input_ids, lengths).squeeze(-1)
            loss = criterion(logits, targets)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), config["training"]["clip_grad_norm"])
            
            optimizer.step()
            # Detach to avoid accumulating the entire computation graph
            train_loss_tensor += loss.detach() * input_ids.size(0)

        # Single CPU sync point for the entire epoch
        epoch_train_loss = train_loss_tensor.item() / len(train_loader.dataset)
        
        # --- VALIDATION CYCLE ---
        model.eval()
        val_loss_tensor = torch.tensor(0.0)
        all_targets = []
        all_logits = []

        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                lengths = batch["length"]
                targets = batch["label"].to(device)
                
                logits = model(input_ids, lengths).squeeze(-1)
                loss = criterion(logits, targets)
                
                # Accumulate as tensor — avoid per-batch .item() sync
                val_loss_tensor = val_loss_tensor + loss.detach() * input_ids.size(0)
                
                # Keep on device, just append to list
                all_targets.append(targets)
                all_logits.append(logits)
                
        epoch_val_loss = val_loss_tensor.item() / len(val_loader.dataset)
        
        # Calculate metrics efficiently: concatenate once, then move to CPU
        y_true = torch.cat(all_targets).cpu().numpy()
        y_pred_logits = torch.cat(all_logits).cpu().numpy()
        metrics = compute_classification_metrics(y_true, y_pred_logits, threshold=config["training"]["decision_threshold"])
        
        val_acc = metrics["accuracy"]
        val_prec = metrics["precision"]
        val_rec = metrics["recall"]
        val_f1 = metrics["f1"]
        val_auc = metrics["roc_auc"]
        
        # Retrieve current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_duration = time.time() - epoch_start_time
        
        # Log results
        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Acc: {val_acc:.4f} | "
              f"Prec: {val_prec:.4f} | "
              f"Rec: {val_rec:.4f} | "
              f"F1: {val_f1:.4f} | "
              f"AUC: {val_auc:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_duration:.1f}s")
              
        # Record history
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "val_accuracy": val_acc,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_f1": val_f1,
            "val_roc_auc": val_auc,
            "lr": current_lr,
            "duration_seconds": epoch_duration
        }
        history.append(epoch_metrics)
        
        # Update scheduler using F1 score
        if not args.smoke_test:
            scheduler.step(val_f1)
            
        # Checkpoint Saving and Early Stopping Check (Only for non-smoke test runs)
        if not args.smoke_test:
            # Update latest checkpoint for resume support
            latest_path = os.path.join(config["paths"]["checkpoints_dir"], "latest_checkpoint.pt")
            save_checkpoint(
                latest_path, model, optimizer, scheduler, epoch, best_f1, 
                {"val_loss": epoch_val_loss, **metrics}, config, 
                config["project"]["seed"], vocab_size, pad_idx
            )
            
            # Selection: Validation F1-score first, Val loss as tie-breaker
            is_better_f1 = val_f1 > best_f1
            is_tied_f1_better_loss = (val_f1 == best_f1) and (epoch_val_loss < best_val_loss)
            
            if is_better_f1 or is_tied_f1_better_loss:
                # Save best model checkpoint
                best_path = os.path.join(config["paths"]["checkpoints_dir"], "best_model.pt")
                save_checkpoint(
                    best_path, model, optimizer, scheduler, epoch, val_f1, 
                    {"val_loss": epoch_val_loss, **metrics}, config, 
                    config["project"]["seed"], vocab_size, pad_idx
                )
                
                best_f1 = val_f1
                best_val_loss = epoch_val_loss
                early_stopping_counter = 0
                print(f"==> Saved new best checkpoint at epoch {epoch}")
            else:
                early_stopping_counter += 1
                
            # Early stopping check
            if early_stopping_counter >= early_stopping_patience:
                print(f"\n[Early Stopping] Triggered after {early_stopping_patience} epochs without F1 improvement.")
                break
                
    # 8. Save Metrics & Plot Curves
    if not args.smoke_test:
        print("\n[5/5] Saving final run metrics and charts...")
        os.makedirs(config["paths"]["metrics_dir"], exist_ok=True)
        
        # Export history JSON
        history_path = os.path.join(config["paths"]["metrics_dir"], "training_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
        print(f"[OK] Training history saved to: {os.path.abspath(history_path)}")
        
        # Plot curves
        plot_curves(history, config["paths"]["figures_dir"])
        
    print("\n" + "=" * 60)
    print("Execution successfully completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
