import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any

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
        self.sequences = sequences
        self.labels = labels
        self.lengths = lengths

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.sequences[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
            "length": torch.tensor(self.lengths[idx], dtype=torch.long)
        }

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
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
