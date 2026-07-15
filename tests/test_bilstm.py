import pytest
import torch
import torch.nn as nn

from src.models.bilstm import BiLSTMVulnerabilityDetector

@pytest.fixture
def model_params():
    return {
        "vocab_size": 100,
        "embedding_dim": 16,
        "hidden_dim": 32,
        "num_layers": 2,
        "dropout": 0.3,
        "padding_idx": 0
    }

def test_model_initialization(model_params):
    """Verify that parameters are correctly configured and layers initialized."""
    model = BiLSTMVulnerabilityDetector(**model_params)
    
    assert model.vocab_size == model_params["vocab_size"]
    assert model.embedding_dim == model_params["embedding_dim"]
    assert model.hidden_dim == model_params["hidden_dim"]
    assert model.num_layers == model_params["num_layers"]
    assert model.padding_idx == model_params["padding_idx"]
    
    # Layer existence and sizes
    assert isinstance(model.embedding, nn.Embedding)
    assert model.embedding.num_embeddings == model_params["vocab_size"]
    assert model.embedding.embedding_dim == model_params["embedding_dim"]
    assert model.embedding.padding_idx == model_params["padding_idx"]
    
    assert isinstance(model.lstm, nn.LSTM)
    assert model.lstm.hidden_size == model_params["hidden_dim"]
    assert model.lstm.num_layers == model_params["num_layers"]
    assert model.lstm.bidirectional is True
    
    assert isinstance(model.dropout, nn.Dropout)
    assert model.dropout.p == model_params["dropout"]
    
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.in_features == model_params["hidden_dim"] * 2
    assert model.fc.out_features == 1

def test_forward_output_shape(model_params):
    """Verify output shape of forward pass is exactly (batch_size, 1)."""
    model = BiLSTMVulnerabilityDetector(**model_params)
    model.eval()
    
    batch_size = 5
    max_len = 10
    input_ids = torch.randint(low=1, high=model_params["vocab_size"], size=(batch_size, max_len))
    lengths = torch.tensor([10, 8, 7, 5, 3], dtype=torch.long)
    
    with torch.no_grad():
        output = model(input_ids, lengths)
        
    assert output.shape == (batch_size, 1)

def test_single_sample_batch(model_params):
    """Verify model execution for a batch of size 1 (single-sample batch)."""
    model = BiLSTMVulnerabilityDetector(**model_params)
    model.eval()
    
    input_ids = torch.randint(low=1, high=model_params["vocab_size"], size=(1, 8))
    lengths = torch.tensor([8], dtype=torch.long)
    
    with torch.no_grad():
        output = model(input_ids, lengths)
        
    assert output.shape == (1, 1)

def test_backward_gradient_flow(model_params):
    """Verify that loss backward pass computes gradients for embedding, LSTM, and fc layers."""
    model = BiLSTMVulnerabilityDetector(**model_params)
    model.train()
    
    batch_size = 3
    max_len = 12
    input_ids = torch.randint(low=1, high=model_params["vocab_size"], size=(batch_size, max_len))
    lengths = torch.tensor([12, 10, 6], dtype=torch.long)
    targets = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.float)
    
    # Forward
    logits = model(input_ids, lengths)
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, targets)
    
    # Backward
    loss.backward()
    
    # Verify gradients are populated and non-zero
    assert model.embedding.weight.grad is not None
    assert torch.any(model.embedding.weight.grad != 0)
    
    # Verify gradients for LSTM weights
    for name, param in model.lstm.named_parameters():
        assert param.grad is not None
        assert torch.any(param.grad != 0)
        
    # Verify gradients for Linear classifier weights
    assert model.fc.weight.grad is not None
    assert torch.any(model.fc.weight.grad != 0)
    assert model.fc.bias.grad is not None

def test_padding_independence(model_params):
    """
    Crucial Test: Verify that trailing padding tokens do not alter the sequence representation.
    We pass a sequence, then pass the same sequence with extra zero-padding at the end.
    With correct packing, the output logits for the active elements should be identical.
    """
    model = BiLSTMVulnerabilityDetector(**model_params)
    model.eval()
    
    # Active sequence sequence
    seq1 = [5, 12, 23, 44, 8, 3] # len 6
    pad_idx = model_params["padding_idx"]
    
    # Convert to input batches
    # Sample A: max length 6 (no padding)
    # Sample B: max length 10 (padded with 4 zeros)
    input_ids_a = torch.tensor([seq1], dtype=torch.long)
    lengths_a = torch.tensor([6], dtype=torch.long)
    
    input_ids_b = torch.tensor([seq1 + [pad_idx] * 4], dtype=torch.long)
    lengths_b = torch.tensor([6], dtype=torch.long)
    
    # Set weights to be fixed (or call forward)
    with torch.no_grad():
        output_a = model(input_ids_a, lengths_a)
        output_b = model(input_ids_b, lengths_b)
        
    # Check that outputs are identical, proving padding is ignored in computation
    assert torch.allclose(output_a, output_b, atol=1e-6)

def test_invalid_input_handling(model_params):
    """Verify that invalid shapes and mismatched batch sizes raise ValueError."""
    model = BiLSTMVulnerabilityDetector(**model_params)
    
    # input_ids is 3D instead of 2D
    bad_input_ids = torch.randint(low=1, high=10, size=(2, 2, 2))
    with pytest.raises(ValueError, match="input_ids must have 2 dimensions"):
        model(bad_input_ids, [2, 2])
        
    # lengths is 2D instead of 1D
    input_ids = torch.randint(low=1, high=10, size=(2, 5))
    bad_lengths = torch.tensor([[5, 5]], dtype=torch.long)
    with pytest.raises(ValueError, match="lengths must be a 1D tensor"):
        model(input_ids, bad_lengths)
        
    # Mismatched batch size (batch_size = 2, lengths size = 3)
    with pytest.raises(ValueError, match="Mismatched batch sizes"):
        model(input_ids, [5, 5, 5])

def test_cpu_execution(model_params):
    """Verify the model runs successfully on CPU."""
    model = BiLSTMVulnerabilityDetector(**model_params).cpu()
    model.eval()
    
    input_ids = torch.randint(low=1, high=model_params["vocab_size"], size=(2, 10))
    lengths = torch.tensor([10, 5], dtype=torch.long)
    
    with torch.no_grad():
        output = model(input_ids, lengths)
        
    assert output.device.type == "cpu"
    assert output.shape == (2, 1)
