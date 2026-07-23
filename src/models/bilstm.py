import torch
import torch.nn as nn
from typing import Union, List

class BiLSTMVulnerabilityDetector(nn.Module):
    """
    Bidirectional LSTM model for binary source code vulnerability detection.
    Processes token sequences in both directions, handles padded variable-length 
    sequences safely using packed sequences, and outputs a single classification logit.
    
    Output format: Tensor of shape (batch_size, 1) representing raw classification logits.
    """
    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int, 
                 hidden_dim: int, 
                 num_layers: int, 
                 dropout: float = 0.5, 
                 padding_idx: int = 0) -> None:
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimensionality of token embedding space.
            hidden_dim (int): Hidden dimension size of individual direction LSTM cell.
            num_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout probability applied to embedding and classifier.
            padding_idx (int): Vocabulary index representing padding tokens.
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        
        # Trainable embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
        # Bidirectional LSTM layer
        # If num_layers = 1, PyTorch's LSTM does not support dropout internally (dropout parameter is ignored)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Regularization layer
        self.dropout = nn.Dropout(p=dropout)
        
        # Fully connected projection layer mapping concatenated bidirectional states (hidden_dim * 2) to logit (1)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, input_ids: torch.Tensor, lengths: Union[torch.Tensor, List[int]]) -> torch.Tensor:
        """
        Forward pass of the BiLSTMVulnerabilityDetector model.
        
        Args:
            input_ids (torch.Tensor): Padded token index tensor of shape (batch_size, seq_len).
            lengths (torch.Tensor or List[int]): Actual unpadded token lengths of shape (batch_size,).
            
        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, 1).
            
        Raises:
            ValueError: If input dimensions are invalid or shapes do not align.
        """
        # 1. Validation Checks
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError(f"input_ids must be a torch.Tensor. Got: {type(input_ids)}")
            
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must have 2 dimensions (batch_size, seq_len). Got shape: {input_ids.shape}")
            
        batch_size = input_ids.size(0)
        
        # Standardize lengths to CPU LongTensor
        if isinstance(lengths, torch.Tensor):
            if lengths.dim() != 1:
                raise ValueError(f"lengths must be a 1D tensor of shape (batch_size,). Got shape: {lengths.shape}")
            lengths_cpu = lengths.cpu()
        else:
            if not isinstance(lengths, (list, tuple)):
                raise ValueError(f"lengths must be a Tensor or list/tuple of integers. Got: {type(lengths)}")
            lengths_cpu = torch.tensor(lengths, dtype=torch.long)
            
        if lengths_cpu.size(0) != batch_size:
            raise ValueError(
                f"Mismatched batch sizes: input_ids batch_size={batch_size}, lengths size={lengths_cpu.size(0)}"
            )
            
        # 2. Embedding Layer
        # Output shape: [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        # 3. Pack Padded Sequence for Variable Lengths
        # PyTorch requires lengths to be sorted descending if enforce_sorted=True. 
        # Using enforce_sorted=False allows PyTorch to sort and unsort internally.
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=True
        )
        
        # 4. Bidirectional LSTM Layer
        # Output packed representations, along with hidden and cell states.
        # hn shape: [num_layers * 2, batch_size, hidden_dim]
        # cn shape: [num_layers * 2, batch_size, hidden_dim]
        _, (hn, _) = self.lstm(packed_embedded)
        
        # 5. Extract Final Top-Layer Bidirectional Hidden States
        # The hidden states corresponding to the top-most layer of the LSTM are:
        # Forward hidden state: hn[-2] (or index num_layers*2 - 2)
        # Backward hidden state: hn[-1] (or index num_layers*2 - 1)
        # Concat shape: [batch_size, hidden_dim * 2]
        hn_forward = hn[-2]
        hn_backward = hn[-1]
        
        combined_representation = torch.cat((hn_forward, hn_backward), dim=-1)
        
        # 6. Dropout & Classification Projection
        # Output shape: [batch_size, 1]
        output_logits = self.fc(self.dropout(combined_representation))
        
        return output_logits
