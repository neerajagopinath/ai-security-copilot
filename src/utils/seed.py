import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """
    Set reproducible random seeds for Python, NumPy, and PyTorch.
    
    Args:
        seed (int): The random seed value to set.
    """
    # Python built-in random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch GPU (all devices if multi-GPU is used)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Configure cuDNN to be deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    print(f"Reproducibility seed set to: {seed}")
