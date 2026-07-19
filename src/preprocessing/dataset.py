import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random
from typing import List, Dict, Any, Iterator

class VulnerabilityDataset(Dataset):
    """
    PyTorch Dataset wrapper for processed source-code sequences.
    Returns input_ids, binary labels, and unpadded sequence lengths.
    """
    def __init__(self, 
                 sequences: List[List[int]], 
                 labels: List[int], 
                 lengths: List[int]) -> None:
        """
        Args:
            sequences (List[List[int]]): List of padded/truncated integer token IDs.
            labels (List[int]): List of binary vulnerability labels (0 or 1).
            lengths (List[int]): List of actual unpadded token lengths (including BOS/EOS).
        """
        if not (len(sequences) == len(labels) == len(lengths)):
            raise ValueError(
                f"Mismatch in sizes: sequences={len(sequences)}, labels={len(labels)}, lengths={len(lengths)}"
            )
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.lengths = torch.tensor(lengths, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.sequences[idx],
            "label": self.labels[idx],
            "length": self.lengths[idx]
        }

def sorted_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collates a list of dataset elements into a batch and sorts by length descending.
    This eliminates the sorting overhead in PyTorch's pack_padded_sequence (enforce_sorted=False).
    Additionally dynamically truncates the input_ids to the max length in the batch.
    """
    # Sort batch by length descending
    batch.sort(key=lambda x: x["length"].item(), reverse=True)
    
    input_ids = torch.stack([x["input_ids"] for x in batch])
    labels = torch.stack([x["label"] for x in batch])
    lengths = torch.stack([x["length"] for x in batch])
    
    # Dynamic padding: truncate to max length in this batch
    max_len = lengths[0].item()  # Since it's sorted, first element has max length
    input_ids = input_ids[:, :max_len]
    
    return {
        "input_ids": input_ids,
        "label": labels,
        "length": lengths
    }

class BucketBatchSampler(Sampler):
    """
    Groups sequences of similar lengths into batches to minimize padding waste.
    """
    def __init__(self, lengths: torch.Tensor, batch_size: int, shuffle: bool = True):
        self.lengths = lengths.tolist()
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[List[int]]:
        # Create list of indices
        indices = list(range(len(self.lengths)))
        
        # Sort indices by sequence length
        indices.sort(key=lambda i: self.lengths[i])
        
        # Group into batches
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        
        # Shuffle the batches if required
        if self.shuffle:
            random.shuffle(batches)
            
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

def get_dataloader(dataset: VulnerabilityDataset, 
                   batch_size: int = 64, 
                   shuffle: bool = True, 
                   num_workers: int = 0) -> DataLoader:
    """
    Create a PyTorch DataLoader for the VulnerabilityDataset.
    
    Args:
        dataset (VulnerabilityDataset): The dataset instance.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of subprocesses to use for data loading.
        
    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    batch_sampler = BucketBatchSampler(dataset.lengths, batch_size=batch_size, shuffle=shuffle)
    
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=sorted_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
