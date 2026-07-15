import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import List, Dict, Any, Tuple

def calculate_pos_weight(labels: List[int]) -> float:
    """
    Calculate pos_weight for BCEWithLogitsLoss based strictly on training labels.
    pos_weight = neg_count / pos_count
    
    Args:
        labels (List[int]): List of training target labels (0 or 1).
        
    Returns:
        float: Calculated class weight.
    """
    neg_count = sum(1 for label in labels if label == 0)
    pos_count = sum(1 for label in labels if label == 1)
    
    if pos_count == 0:
        print("[Warning] No positive samples found in training labels. Setting pos_weight=1.0")
        return 1.0
        
    pos_weight = neg_count / pos_count
    return float(pos_weight)

def compute_classification_metrics(y_true: np.ndarray, 
                                 y_pred_logits: np.ndarray, 
                                 threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute binary classification metrics: Accuracy, Precision, Recall, F1, and ROC-AUC.
    
    Args:
        y_true (np.ndarray): 1D array of ground truth labels (0 or 1).
        y_pred_logits (np.ndarray): 1D array of raw output logits.
        threshold (float): Decision threshold to map probabilities to binary labels.
        
    Returns:
        Dict[str, float]: Computed metrics dictionary.
    """
    # Map logits to probabilities using sigmoid activation
    y_pred_probs = 1.0 / (1.0 + np.exp(-y_pred_logits))
    # Map probabilities to class decisions
    y_pred_classes = (y_pred_probs >= threshold).astype(int)
    
    acc = float(accuracy_score(y_true, y_pred_classes))
    prec = float(precision_score(y_true, y_pred_classes, zero_division=0))
    rec = float(recall_score(y_true, y_pred_classes, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_classes, zero_division=0))
    
    # Safely compute ROC-AUC (requires both classes to be present)
    auc = 0.0
    if len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_pred_probs))
        except ValueError:
            auc = 0.0
            
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc
    }

def save_checkpoint(path: str,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: Any,
                    epoch: int,
                    best_f1: float,
                    metrics: Dict[str, float],
                    config: Dict[str, Any],
                    seed: int,
                    vocab_size: int,
                    pad_idx: int) -> None:
    """
    Serialize and save a complete training checkpoint to disk.
    
    Args:
        path (str): Target file path.
        model (torch.nn.Module): Model instance.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        scheduler (Any): Scheduler instance.
        epoch (int): Current training epoch.
        best_f1 (float): Best validation F1-score achieved so far.
        metrics (Dict[str, float]): Validation metrics of the current epoch.
        config (Dict[str, Any]): Project configuration.
        seed (int): Reproducibility seed.
        vocab_size (int): Vocabulary size.
        pad_idx (int): Padding index.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_f1": best_f1,
        "metrics": metrics,
        "config": config,
        "seed": seed,
        "vocab_size": vocab_size,
        "pad_idx": pad_idx
    }
    
    torch.save(checkpoint, path)
    # Print absolute path cleanly
    print(f"[OK] Saved checkpoint to: {os.path.abspath(path)}")

def load_checkpoint(path: str,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer = None,
                    scheduler: Any = None) -> Dict[str, Any]:
    """
    Load a saved training checkpoint and restore model, optimizer, and scheduler states.
    
    Args:
        path (str): Path to checkpoint file.
        model (torch.nn.Module): Model instance to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (Any, optional): Scheduler to load state into.
        
    Returns:
        Dict[str, Any]: Loaded checkpoint dictionary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at: {path}")
        
    checkpoint = torch.load(path, map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
    print(f"[OK] Successfully loaded checkpoint from {os.path.abspath(path)} (Epoch: {checkpoint['epoch']})")
    return checkpoint
