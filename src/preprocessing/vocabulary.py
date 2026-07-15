import json
from collections import Counter
from typing import List, Dict, Iterable

class Vocabulary:
    """
    A class that maintains token-to-index mappings for code sequences.
    Ensures reproducibility and handles special tokens.
    """
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    
    PAD_IDX = 0
    UNK_IDX = 1
    BOS_IDX = 2
    EOS_IDX = 3

    def __init__(self) -> None:
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}
        self._reset()

    def _reset(self) -> None:
        """Reset vocab to only special tokens."""
        self.token_to_idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
            self.BOS_TOKEN: self.BOS_IDX,
            self.EOS_TOKEN: self.EOS_IDX
        }
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

    def __len__(self) -> int:
        return len(self.token_to_idx)

    def build_vocab(self, 
                    tokenized_texts: Iterable[List[str]], 
                    max_size: int = 10000, 
                    min_freq: int = 2) -> None:
        """
        Build the vocabulary from tokenized texts.
        This must be called ONLY on the training split to prevent leakage.
        
        Args:
            tokenized_texts (Iterable[List[str]]): Collection of token lists.
            max_size (int): Max size of vocabulary including special tokens.
            min_freq (int): Minimum count of a token to be registered.
        """
        self._reset()
        
        # Count all tokens in training corpus
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)
            
        # Filter tokens by minimum frequency and remove special tokens if present
        for spec_tok in [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]:
            if spec_tok in counter:
                del counter[spec_tok]
                
        # Get most common tokens up to capacity
        # Max capacity for new words is max_size - 4
        allowed_vocab_size = max(0, max_size - len(self.token_to_idx))
        most_common = counter.most_common(allowed_vocab_size)
        
        # Add to vocabulary
        idx = len(self.token_to_idx)
        for token, freq in most_common:
            if freq >= min_freq:
                self.token_to_idx[token] = idx
                self.idx_to_token[idx] = token
                idx += 1

    def numericalize(self, tokens: List[str]) -> List[int]:
        """
        Convert a list of string tokens into a list of integer indices.
        Uses the <UNK> token index for out-of-vocabulary words.
        
        Args:
            tokens (List[str]): List of string tokens.
            
        Returns:
            List[int]: List of corresponding token indices.
        """
        return [self.token_to_idx.get(tok, self.UNK_IDX) for tok in tokens]

    def save(self, file_path: str) -> None:
        """Save the vocabulary mappings as JSON."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.token_to_idx, f, indent=4, ensure_ascii=False)

    def load(self, file_path: str) -> None:
        """Load vocabulary mapping from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            token_to_idx = json.load(f)
            
        self.token_to_idx = token_to_idx
        self.idx_to_token = {int(idx): token for token, idx in self.token_to_idx.items()}
