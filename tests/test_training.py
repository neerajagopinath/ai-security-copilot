import os
import json
import pytest
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any

from src.training.train_utils import (
    calculate_pos_weight, 
    compute_classification_metrics, 
    save_checkpoint, 
    load_checkpoint
)
from src.models.bilstm import BiLSTMVulnerabilityDetector

@pytest.fixture
def mock_model():
    return nn.Linear(5, 1)

@pytest.fixture
def real_bilstm():
    return BiLSTMVulnerabilityDetector(
        vocab_size=10,
        embedding_dim=4,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        padding_idx=0
    )

def test_calculate_pos_weight():
    """Verify class weight calculation: pos_weight = neg / pos."""
    # Neg = 6, Pos = 2 -> weight = 6 / 2 = 3.0
    labels = [0, 0, 1, 0, 0, 1, 0, 0]
    weight = calculate_pos_weight(labels)
    assert weight == 3.0
    
    # All negatives (handled safely by check)
    assert calculate_pos_weight([0, 0]) == 1.0
    # if pos_count == 0: return 1.0 (as coded in train_utils)
    assert calculate_pos_weight([0, 0]) == 1.0

def test_compute_classification_metrics():
    """Verify metrics match mathematical expectations, including thresholding and ROC-AUC safety."""
    y_true = np.array([0, 1, 0, 1])
    # Sigmoid of logits:
    # 2.0 -> sigmoid(2) = 0.88 >= 0.5 (class 1)
    # -2.0 -> sigmoid(-2) = 0.11 < 0.5 (class 0)
    y_pred_logits = np.array([-2.0, 2.0, -2.0, 2.0]) # predicted classes: [0, 1, 0, 1] (100% correct)
    
    metrics = compute_classification_metrics(y_true, y_pred_logits, threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    
    # Test safe fallback for single class: should return 0.0
    single_class_true = np.array([0, 0, 0, 0])
    metrics_single = compute_classification_metrics(single_class_true, y_pred_logits)
    assert metrics_single["roc_auc"] == 0.0

def test_single_batch_training(real_bilstm):
    """Verify a forward, backward, and optimizer update step runs successfully on a batch."""
    real_bilstm.train()
    optimizer = torch.optim.AdamW(real_bilstm.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    # Batch size 2, length 4
    input_ids = torch.randint(low=1, high=10, size=(2, 4))
    lengths = torch.tensor([4, 3], dtype=torch.long)
    targets = torch.tensor([0.0, 1.0], dtype=torch.float)
    
    # Initial weights comparison
    initial_fc_weight = real_bilstm.fc.weight.clone()
    
    optimizer.zero_grad()
    logits = real_bilstm(input_ids, lengths).squeeze(-1)
    loss = criterion(logits, targets)
    loss.backward()
    
    optimizer.step()
    
    # Assert weight changed after step
    assert not torch.equal(real_bilstm.fc.weight, initial_fc_weight)

def test_single_batch_validation(real_bilstm):
    """Verify a validation step runs without tracking gradients."""
    real_bilstm.eval()
    input_ids = torch.randint(low=1, high=10, size=(2, 4))
    lengths = torch.tensor([4, 2], dtype=torch.long)
    
    with torch.no_grad():
        logits = real_bilstm(input_ids, lengths)
        
    assert logits.shape == (2, 1)
    assert logits.grad is None

def test_gradient_clipping():
    """Verify gradient clipping scales gradient norms correctly."""
    model = nn.Linear(2, 1)
    # Put large gradients manually
    model.weight.grad = torch.tensor([[100.0, 100.0]])
    model.bias.grad = torch.tensor([100.0])
    
    # Clip grad norm to 5.0
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    
    # Total norm before clipping was sqrt(100^2 + 100^2 + 100^2) = sqrt(30000) = 173.2
    assert norm > 170.0
    
    # New norm of weight + bias should be exactly 5.0
    new_norm = torch.norm(torch.cat([model.weight.grad.flatten(), model.bias.grad.flatten()]))
    assert torch.allclose(new_norm, torch.tensor(5.0), atol=1e-4)

def test_checkpoint_save_and_load(tmp_path, real_bilstm):
    """Verify serialization and state restoration from checkpoints."""
    checkpoint_file = os.path.join(tmp_path, "test_check.pt")
    optimizer = torch.optim.AdamW(real_bilstm.parameters(), lr=0.01)
    
    # Save checkpoint
    save_checkpoint(
        path=checkpoint_file,
        model=real_bilstm,
        optimizer=optimizer,
        scheduler=None,
        epoch=5,
        best_f1=0.88,
        metrics={"accuracy": 0.85, "val_loss": 0.3},
        config={"lr": 0.01},
        seed=42,
        vocab_size=10,
        pad_idx=0
    )
    
    assert os.path.exists(checkpoint_file)
    
    # Re-initialize another model and restore
    new_model = BiLSTMVulnerabilityDetector(
        vocab_size=10,
        embedding_dim=4,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        padding_idx=0
    )
    
    # Assert weights differ initially (due to random init)
    assert not torch.equal(new_model.fc.weight, real_bilstm.fc.weight)
    
    checkpoint = load_checkpoint(checkpoint_file, new_model)
    
    # Assert weights match after loading
    assert torch.equal(new_model.fc.weight, real_bilstm.fc.weight)
    assert checkpoint["epoch"] == 5
    assert checkpoint["best_f1"] == 0.88
    assert checkpoint["seed"] == 42

def test_early_stopping_behavior():
    """Verify early stopping patience logic."""
    early_stopping_patience = 3
    early_stopping_counter = 0
    best_f1 = 0.8
    
    # Simulating epoch runs
    # Epoch 1: F1 improves -> reset counter
    val_f1_epoch_1 = 0.85
    if val_f1_epoch_1 > best_f1:
        best_f1 = val_f1_epoch_1
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
    assert early_stopping_counter == 0
    assert best_f1 == 0.85
    
    # Epoch 2: F1 decreases -> increment counter
    val_f1_epoch_2 = 0.82
    if val_f1_epoch_2 > best_f1:
        best_f1 = val_f1_epoch_2
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
    assert early_stopping_counter == 1
    assert best_f1 == 0.85
    
    # Epoch 3: F1 decreases -> increment counter
    val_f1_epoch_3 = 0.84
    if val_f1_epoch_3 > best_f1:
        best_f1 = val_f1_epoch_3
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
    assert early_stopping_counter == 2
    
    # Epoch 4: F1 decreases -> early stopping counter hits limit
    val_f1_epoch_4 = 0.81
    if val_f1_epoch_4 > best_f1:
        best_f1 = val_f1_epoch_4
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
    assert early_stopping_counter == 3
    assert early_stopping_counter >= early_stopping_patience # Trigger stopping

def test_best_model_selection_tie_breaker():
    """Verify that model selection prioritizes validation F1 and uses validation loss as a tie-breaker."""
    # Scenario A: Val F1 is strictly higher -> selection success
    best_f1 = 0.85
    best_loss = 0.4
    
    val_f1_new = 0.87
    val_loss_new = 0.5 # loss is higher but F1 is better
    is_better = val_f1_new > best_f1
    assert is_better is True
    
    # Scenario B: Val F1 is tied, but new loss is lower -> tie-breaker success
    best_f1 = 0.85
    best_loss = 0.4
    
    val_f1_tie = 0.85
    val_loss_lower = 0.35 # Tied F1, lower loss
    
    is_tied_better_loss = (val_f1_tie == best_f1) and (val_loss_lower < best_loss)
    assert is_tied_better_loss is True
    
    # Scenario C: Val F1 is tied, but new loss is higher -> not better
    val_loss_higher = 0.45
    is_tied_better_loss_c = (val_f1_tie == best_f1) and (val_loss_higher < best_loss)
    assert is_tied_better_loss_c is False

def test_history_serialization(tmp_path):
    """Verify that training history logs serialize to JSON correctly."""
    history = [
        {"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "val_f1": 0.7},
        {"epoch": 2, "train_loss": 0.4, "val_loss": 0.35, "val_f1": 0.75}
    ]
    
    history_file = os.path.join(tmp_path, "history.json")
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)
        
    assert os.path.exists(history_file)
    with open(history_file, "r") as f:
        loaded_history = json.load(f)
        
    assert len(loaded_history) == 2
    assert loaded_history[0]["epoch"] == 1
    assert loaded_history[1]["val_f1"] == 0.75
