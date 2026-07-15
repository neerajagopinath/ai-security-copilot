import os
import json
import pytest
import torch
import pandas as pd
from typing import List

from src.preprocessing.tokenizer import SourceCodeTokenizer
from src.preprocessing.vocabulary import Vocabulary
from src.preprocessing.dataset import VulnerabilityDataset, get_dataloader
from src.preprocessing.preprocess import clean_split, process_sample

def test_tokenizer_lexemes():
    """Verify that the tokenizer preserves operators, literals, variables and strips comments."""
    tokenizer = SourceCodeTokenizer()
    code = """
    // This is a line comment
    /* This is a 
       multi-line comment */
    void vuln_check(int *ptr, float val) {
        if (ptr != NULL && val >= 3.14) {
            char *msg = "buffer overflow";
            printf("%s\n", msg);
        }
    }
    """
    tokens = tokenizer.tokenize(code)
    
    # Verify comments are removed
    assert "line comment" not in tokens
    assert "multi-line comment" not in tokens
    
    # Verify syntax elements are preserved
    assert "void" in tokens
    assert "vuln_check" in tokens
    assert "ptr" in tokens
    assert "!=" in tokens
    assert "&&" in tokens
    assert ">=" in tokens
    assert '"buffer overflow"' in tokens
    assert "3.14" in tokens

def test_vocabulary_mapping():
    """Verify vocabulary compilation, special tokens, minimum frequency, and OOV unknown mapping."""
    vocab = Vocabulary()
    
    # Mock tokenized training sentences
    train_tokens = [
        ["void", "main", "(", ")", "{", "int", "x", "=", "0", ";", "}"],
        ["void", "test", "(", ")", "{", "int", "x", "=", "1", ";", "}"]
    ]
    
    # Build vocabulary with max size 12 and min frequency 2
    # Only "void", "main", "(", ")", "{", "int", "x", "=", "0", ";", "}", "test", "1"
    # Freq of "main" is 1, "test" is 1, "0" is 1, "1" is 1. Under min_freq=2 they should be filtered out.
    vocab.build_vocab(train_tokens, max_size=15, min_freq=2)
    
    # Verify special tokens are registered at correct indices
    assert vocab.token_to_idx[Vocabulary.PAD_TOKEN] == Vocabulary.PAD_IDX
    assert vocab.token_to_idx[Vocabulary.UNK_TOKEN] == Vocabulary.UNK_IDX
    assert vocab.token_to_idx[Vocabulary.BOS_TOKEN] == Vocabulary.BOS_IDX
    assert vocab.token_to_idx[Vocabulary.EOS_TOKEN] == Vocabulary.EOS_IDX
    
    # Common tokens should be present
    assert "void" in vocab.token_to_idx
    assert "int" in vocab.token_to_idx
    
    # Uncommon tokens (< min_freq) should be absent
    assert "main" not in vocab.token_to_idx
    assert "test" not in vocab.token_to_idx
    
    # OOV check: "main" and "test" should numericalize to UNK_IDX (1)
    numerical = vocab.numericalize(["void", "main"])
    assert numerical[0] == vocab.token_to_idx["void"]
    assert numerical[1] == Vocabulary.UNK_IDX

def test_padding_and_truncation():
    """Verify that sequences are correctly padded and truncated while keeping BOS and EOS tags."""
    tokenizer = SourceCodeTokenizer()
    vocab = Vocabulary()
    
    # Setup simple vocab
    train_tokens = [["int", "x", "=", "5", ";"]]
    vocab.build_vocab(train_tokens, min_freq=1)
    
    # Test sample processing
    code = "int x = 5;"  # Tokenized: ['int', 'x', '=', '5', ';'] -> len 5
    
    # Case 1: Padding needed (max_len = 10, BOS/EOS + 5 tokens = 7 values, needs 3 pads)
    seq, length, truncated, _ = process_sample(code, tokenizer, vocab, max_len=10)
    assert len(seq) == 10
    assert length == 7
    assert not truncated
    assert seq[0] == Vocabulary.BOS_IDX
    assert seq[6] == Vocabulary.EOS_IDX
    assert seq[7:] == [Vocabulary.PAD_IDX, Vocabulary.PAD_IDX, Vocabulary.PAD_IDX]
    
    # Case 2: Truncation needed (max_len = 5, BOS + first 3 tokens + EOS = 5 values)
    seq_trunc, length_trunc, truncated_flag, _ = process_sample(code, tokenizer, vocab, max_len=5)
    assert len(seq_trunc) == 5
    assert length_trunc == 5
    assert truncated_flag
    assert seq_trunc[0] == Vocabulary.BOS_IDX
    assert seq_trunc[-1] == Vocabulary.EOS_IDX

def test_dataset_and_dataloader():
    """Verify PyTorch Dataset conversions and DataLoader batched outputs."""
    seqs = [[2, 12, 13, 3, 0], [2, 14, 15, 3, 0]]
    labels = [0, 1]
    lengths = [4, 4]
    
    dataset = VulnerabilityDataset(seqs, labels, lengths)
    assert len(dataset) == 2
    
    # Test item access
    item = dataset[0]
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["label"], torch.Tensor)
    assert isinstance(item["length"], torch.Tensor)
    assert item["input_ids"].dtype == torch.long
    assert item["label"].dtype == torch.float
    
    # Test DataLoader
    loader = get_dataloader(dataset, batch_size=2, shuffle=False)
    for batch in loader:
        assert batch["input_ids"].shape == (2, 5)
        assert batch["label"].shape == (2,)
        assert batch["length"].tolist() == [4, 4]

def test_clean_split_duplicates():
    """Verify that clean_split properly deduplicates exact values and discards conflicting labels."""
    # Data:
    # Row 0, 1: Exact duplicates
    # Row 2, 3: Conflicting duplicates (same code, diff label)
    # Row 4: Clean valid
    # Row 5: Empty code string
    df = pd.DataFrame({
        "func": [
            "void f1() {}", "void f1() {}",
            "void f2() {}", "void f2() {}",
            "void f3() {}", "   "
        ],
        "target": [0, 0, 0, 1, 1, 0]
    })
    
    df_clean, stats = clean_split(df, "func", "target")
    
    # Expected results:
    # 'void f1() {}' -> exact duplicate, keep one (1 sample remains)
    # 'void f2() {}' -> conflicting labels, discard both (0 samples remain)
    # 'void f3() {}' -> clean (1 sample remains)
    # '   ' -> empty, discarded (0 samples remain)
    assert len(df_clean) == 2
    assert stats["exact_dupes_removed"] == 1
    assert stats["conflicting_samples_removed"] == 2
    assert stats["invalid_removed"] == 1  # the empty string
    
    # Validate final content
    assert set(df_clean["func"]) == {"void f1() {}", "void f3() {}"}

def test_cross_split_leakage_and_vocab_isolation():
    """
    Verify:
    1. Cross-split duplicates are purged to prevent leakage.
    2. Validation and Test datasets are isolated and never leak into Vocabulary.
    """
    train_df = pd.DataFrame({
        "func": ["void common() {}", "void train_only() {}", "void leak_check() {}"],
        "target": [1, 0, 1]
    })
    val_df = pd.DataFrame({
        "func": ["void common() {}", "void val_only() {}"],
        "target": [1, 0]
    })
    test_df = pd.DataFrame({
        "func": ["void leak_check() {}", "void test_only() {}"],
        "target": [1, 0]
    })
    
    # Preprocessing leakage checks logic:
    val_codes = set(val_df["func"])
    test_codes = set(test_df["func"])
    
    # Purge train overlap
    train_leak_mask = train_df["func"].isin(val_codes | test_codes)
    train_final = train_df[~train_leak_mask]
    
    # 'void common() {}' and 'void leak_check() {}' must be purged from training set
    assert len(train_final) == 1
    assert train_final["func"].values[0] == "void train_only() {}"
    
    # Build vocabulary strictly on train_final
    tokenizer = SourceCodeTokenizer()
    train_tokens = [tokenizer.tokenize(code) for code in train_final["func"]]
    
    vocab = Vocabulary()
    vocab.build_vocab(train_tokens, min_freq=1)
    
    # Verify vocabulary contains train terms
    assert "train_only" in vocab.token_to_idx
    
    # Verify validation and test terms ("val_only", "test_only") are ABSENT from vocab
    assert "val_only" not in vocab.token_to_idx
    assert "test_only" not in vocab.token_to_idx
    
    # Check that they map to UNK when numericalizing validation/test
    val_numerical = vocab.numericalize(tokenizer.tokenize("void val_only() {}"))
    assert vocab.token_to_idx[Vocabulary.UNK_TOKEN] in val_numerical
