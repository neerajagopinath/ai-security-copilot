import os
import json
import yaml
import torch

from src.models.bilstm import BiLSTMVulnerabilityDetector

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters of a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def main():
    print("=" * 60)
    print("AI Security Copilot - Model Verification")
    print("=" * 60)
    
    # 1. Load config
    config = load_config()
    print("[OK] Configuration loaded.")
    
    # 2. Get Vocabulary Size from processed vocab (or default config)
    vocab_path = "data/processed/vocabulary.json"
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)
        vocab_size = len(vocab_dict)
        print(f"[OK] Loaded processed vocabulary from {vocab_path} (Size: {vocab_size})")
    else:
        vocab_size = config["models"]["bilstm"]["vocab_size"]
        print(f"[Warning] Processed vocabulary not found. Using default config size: {vocab_size}")

    # 3. Read LSTM hyperparams
    lstm_config = config["models"]["bilstm"]
    embedding_dim = lstm_config["embedding_dim"]
    hidden_dim = lstm_config["hidden_dim"]
    num_layers = lstm_config["num_layers"]
    dropout = lstm_config["dropout"]
    padding_idx = 0 # Default pad idx
    
    print("\nModel Configuration:")
    print(f"  - Vocab Size:    {vocab_size}")
    print(f"  - Embedding Dim: {embedding_dim}")
    print(f"  - Hidden Dim:    {hidden_dim}")
    print(f"  - Num Layers:    {num_layers}")
    print(f"  - Dropout:       {dropout}")
    print(f"  - Padding Index: {padding_idx}")
    
    # 4. Initialize Model
    print("\nInitializing model...")
    model = BiLSTMVulnerabilityDetector(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        padding_idx=padding_idx
    )
    print("[OK] Model instantiated successfully.")
    
    # 5. Create simulated batch
    batch_size = 4
    max_len = 256
    
    # Generate random token IDs
    torch.manual_seed(42)
    mock_input_ids = torch.randint(low=2, high=vocab_size, size=(batch_size, max_len))
    
    # Define variable sequence lengths (first length = max_len, other varying)
    mock_lengths = torch.tensor([256, 128, 45, 180], dtype=torch.long)
    
    # Insert padding index in the padded areas
    for idx, length in enumerate(mock_lengths):
        mock_input_ids[idx, length.item():] = padding_idx
        
    print("\nSimulating Batch:")
    print(f"  - Input Tensor Shape:    {mock_input_ids.shape}")
    print(f"  - Lengths Tensor Shape:  {mock_lengths.shape}")
    print(f"  - Actual lengths list:   {mock_lengths.tolist()}")
    
    # 6. Perform forward pass
    print("\nExecuting forward pass...")
    try:
        model.eval() # Set eval mode to disable dropout behavior
        with torch.no_grad():
            output_logits = model(mock_input_ids, mock_lengths)
        print("[OK] Forward pass executed successfully!")
        print(f"  - Output Logits Shape:   {output_logits.shape}")
        print(f"  - Sample Logits values:\n{output_logits.squeeze(-1).tolist()}")
    except Exception as e:
        print(f"[FAIL] Forward pass failed with exception: {e}")
        return
        
    # 7. Print Model parameter count details
    total_params, trainable_params = count_parameters(model)
    print("\nParameter Summary:")
    print(f"  - Total Parameters:      {total_params:,}")
    print(f"  - Trainable Parameters:  {trainable_params:,}")
    
    print("\nModel Architecture:")
    print(model)
    print("=" * 60)

if __name__ == "__main__":
    main()
