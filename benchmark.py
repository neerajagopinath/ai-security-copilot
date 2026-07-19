import time
import json
import torch
import torch.nn as nn
from src.preprocessing.dataset import VulnerabilityDataset, get_dataloader
from src.models.bilstm import BiLSTMVulnerabilityDetector
from src.training.train_utils import calculate_pos_weight

def load_data_subset():
    sequences, labels, lengths = [], [], []
    count = 0
    with open("data/processed/train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            sequences.append(record["input_ids"])
            labels.append(record["label"])
            lengths.append(record["length"])
            count += 1
            if count >= 3000:
                break
    return sequences, labels, lengths

def run_benchmark():
    device = torch.device("cpu")
    print("Loading data subset...")
    sequences, labels, lengths = load_data_subset()
    
    print("Creating dataset...")
    t0 = time.time()
    dataset = VulnerabilityDataset(sequences, labels, lengths)
    t1 = time.time()
    print(f"Dataset creation took {t1-t0:.4f}s")
    
    dataloader = get_dataloader(dataset, batch_size=64, shuffle=True, num_workers=0)
    
    model = BiLSTMVulnerabilityDetector(
        vocab_size=10000,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.5,
        padding_idx=0
    )
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    pos_weight = torch.tensor([1.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    print("Starting benchmark (20 batches)...")
    model.train()
    
    total_time = 0
    data_time = 0
    forward_time = 0
    backward_time = 0
    
    t_start = time.time()
    batch_count = 0
    
    batch_start = time.time()
    for batch in dataloader:
        if batch_count >= 20:
            break
            
        data_end = time.time()
        data_time += (data_end - batch_start)
        
        input_ids = batch["input_ids"].to(device)
        batch_lengths = batch["length"]
        targets = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        fwd_start = time.time()
        logits = model(input_ids, batch_lengths).squeeze(-1)
        loss = criterion(logits, targets)
        fwd_end = time.time()
        forward_time += (fwd_end - fwd_start)
        
        bwd_start = time.time()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        bwd_end = time.time()
        backward_time += (bwd_end - bwd_start)
        
        batch_count += 1
        batch_start = time.time()
        
    t_end = time.time()
    total_time = t_end - t_start
    
    print(f"Total time for 20 batches: {total_time:.4f}s")
    print(f"Avg time per batch: {total_time / 20:.4f}s")
    print(f"  Data load: {data_time / 20:.4f}s")
    print(f"  Forward:   {forward_time / 20:.4f}s")
    print(f"  Backward:  {backward_time / 20:.4f}s")
    
    # Estimate for 1 epoch (~341 batches for 21854 samples)
    batches_per_epoch = 21854 / 64
    est_epoch = (total_time / 20) * batches_per_epoch
    print(f"Estimated time per epoch: {est_epoch:.2f}s ({est_epoch/60:.2f} min)")

if __name__ == '__main__':
    run_benchmark()
