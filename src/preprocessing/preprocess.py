import os
import json
import argparse
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any

from src.preprocessing.loader import load_data, detect_columns
from src.preprocessing.tokenizer import SourceCodeTokenizer
from src.preprocessing.vocabulary import Vocabulary

def clean_split(df: pd.DataFrame, src_col: str, lbl_col: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Individually clean a dataset split:
    - Remove nulls/empty entries
    - Validate binary labels
    - Remove exact duplicates (same code, same label)
    - Remove conflicting duplicates (same code, different labels)
    """
    initial_count = len(df)
    
    # 1. Remove rows with null code or label
    df = df.dropna(subset=[src_col, lbl_col])
    null_removed = initial_count - len(df)
    
    # 2. Remove empty or whitespace-only code strings
    df = df[df[src_col].astype(str).str.strip().str.len() > 0]
    empty_removed = initial_count - null_removed - len(df)
    
    # 3. Validate binary labels
    df = df[df[lbl_col].isin([0, 1, 0.0, 1.0, True, False])]
    df[lbl_col] = df[lbl_col].astype(int)
    invalid_lbl_removed = initial_count - null_removed - empty_removed - len(df)
    
    # 4. Remove exact duplicates (same code, same label) -> keep first occurrence
    df_dedup = df.drop_duplicates(subset=[src_col, lbl_col], keep='first')
    exact_dupes_removed = len(df) - len(df_dedup)
    
    # 5. Remove conflicting duplicates (same code, different labels) -> discard completely
    code_label_counts = df_dedup.groupby(src_col)[lbl_col].nunique()
    conflicting_codes = code_label_counts[code_label_counts > 1].index
    df_clean = df_dedup[~df_dedup[src_col].isin(conflicting_codes)]
    conflicting_removed = len(df_dedup) - len(df_clean)
    
    clean_count = len(df_clean)
    invalid_removed = null_removed + empty_removed + invalid_lbl_removed
    
    stats = {
        "initial_count": initial_count,
        "clean_count": clean_count,
        "invalid_removed": invalid_removed,
        "exact_dupes_removed": exact_dupes_removed,
        "conflicting_samples_removed": conflicting_removed,
        "unique_conflicting_codes": len(conflicting_codes)
    }
    
    return df_clean, stats

def process_sample(code: str, 
                   tokenizer: SourceCodeTokenizer, 
                   vocab: Vocabulary, 
                   max_len: int) -> Tuple[List[int], int, bool, int]:
    """
    Tokenize, numericalize, add BOS/EOS, and apply padding/truncation to a sample.
    
    Returns:
        Tuple: (sequence, unpadded_length_including_bos_eos, was_truncated, unk_token_count)
    """
    tokens = tokenizer.tokenize(code)
    raw_length = len(tokens)
    token_ids = vocab.numericalize(tokens)
    
    # Count UNK tokens
    unk_count = token_ids.count(Vocabulary.UNK_IDX)
    
    # Add BOS/EOS
    full_seq = [Vocabulary.BOS_IDX] + token_ids + [Vocabulary.EOS_IDX]
    unpadded_len = len(full_seq)
    
    was_truncated = False
    if unpadded_len > max_len:
        # Truncate: keep BOS, first max_len - 2 tokens, and EOS
        full_seq = [Vocabulary.BOS_IDX] + token_ids[:max_len - 2] + [Vocabulary.EOS_IDX]
        unpadded_len = max_len
        was_truncated = True
    elif unpadded_len < max_len:
        # Pad with PAD_IDX
        padding_needed = max_len - unpadded_len
        full_seq = full_seq + [Vocabulary.PAD_IDX] * padding_needed
        
    return full_seq, unpadded_len, was_truncated, unk_count

def run_preprocessing(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("AI Security Copilot - Preprocessing Pipeline")
    print("=" * 60)
    
    # 1. Load raw splits
    print("[1/6] Loading raw datasets...")
    train_df = load_data(args.train_raw)
    val_df = load_data(args.val_raw)
    test_df = load_data(args.test_raw)
    
    print(f"Loaded raw sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Detect columns
    train_src, train_lbl = detect_columns(train_df)
    val_src, val_lbl = detect_columns(val_df)
    test_src, test_lbl = detect_columns(test_df)
    
    # 2. Clean splits
    print("\n[2/6] Cleaning splits and removing duplicates...")
    train_clean, train_stats = clean_split(train_df, train_src, train_lbl)
    val_clean, val_stats = clean_split(val_df, val_src, val_lbl)
    test_clean, test_stats = clean_split(test_df, test_src, test_lbl)
    
    # 3. Cross-Split Leakage Prevention
    print("\n[3/6] Running cross-split leakage checks...")
    val_codes = set(val_clean[val_src].tolist())
    test_codes = set(test_clean[test_src].tolist())
    
    # Find train leakage: code strings appearing in val or test
    train_leakage_mask = train_clean[train_src].isin(val_codes | test_codes)
    train_leakage_samples = train_clean[train_leakage_mask]
    num_train_leakage = len(train_leakage_samples)
    train_final = train_clean[~train_leakage_mask].copy()
    
    # Find validation leakage: code strings appearing in test set
    val_leakage_mask = val_clean[val_src].isin(test_codes)
    val_leakage_samples = val_clean[val_leakage_mask]
    num_val_leakage = len(val_leakage_samples)
    val_final = val_clean[~val_leakage_mask].copy()
    
    test_final = test_clean.copy()
    
    print(f"  - Train leakage samples removed:      {num_train_leakage}")
    print(f"  - Validation leakage samples removed: {num_val_leakage}")
    print(f"Final clean split sizes: train={len(train_final)}, val={len(val_final)}, test={len(test_final)}")
    
    # 4. Initalize Tokenizer and compile Vocabulary
    print("\n[4/6] Initializing tokenizer and building vocabulary...")
    tokenizer = SourceCodeTokenizer()
    
    # Tokenize training set to fit vocab
    print("Tokenizing training set...")
    train_tokenized_codes = []
    raw_lengths = []
    
    for code in train_final[train_src].tolist():
        tokens = tokenizer.tokenize(code)
        train_tokenized_codes.append(tokens)
        # Store lengths of code BEFORE padding/truncation
        raw_lengths.append(len(tokens))
        
    vocab = Vocabulary()
    vocab.build_vocab(train_tokenized_codes, max_size=args.vocab_size, min_freq=args.min_freq)
    print(f"[OK] Vocabulary compiled. Size: {len(vocab)} (includes special tokens)")
    
    # 5. Process and vectorize all splits
    print("\n[5/6] Numericalizing, padding, and truncating sequences...")
    
    def process_dataset(df: pd.DataFrame, src_col: str, lbl_col: str, split_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        processed_records = []
        truncated_count = 0
        total_tokens = 0
        total_unks = 0
        lengths_pre = []
        
        for idx, row in df.iterrows():
            code = str(row[src_col])
            label = int(row[lbl_col])
            sample_id = row.get("id", idx)
            
            seq, unpadded_len, truncated, unks = process_sample(code, tokenizer, vocab, args.max_len)
            
            processed_records.append({
                "id": int(sample_id),
                "input_ids": seq,
                "label": label,
                "length": unpadded_len
            })
            
            if truncated:
                truncated_count += 1
                
            # Track lengths and unks
            tokens_count = len(tokenizer.tokenize(code))
            lengths_pre.append(tokens_count)
            total_tokens += tokens_count
            total_unks += unks
            
        pct_truncated = (truncated_count / len(df)) * 100 if len(df) > 0 else 0
        unk_rate = (total_unks / total_tokens) * 100 if total_tokens > 0 else 0
        
        # Class distribution
        labels = [rec["label"] for rec in processed_records]
        num_non_vuln = labels.count(0)
        num_vuln = labels.count(1)
        
        metrics = {
            "size": len(processed_records),
            "class_distribution": {
                "non_vulnerable": num_non_vuln,
                "vulnerable": num_vuln,
                "ratio_vulnerable": num_vuln / len(processed_records) if len(processed_records) > 0 else 0
            },
            "truncation": {
                "truncated_count": truncated_count,
                "percentage_truncated": pct_truncated
            },
            "unk_rate": unk_rate,
            "lengths_pre_stats": {
                "min": int(np.min(lengths_pre)) if lengths_pre else 0,
                "max": int(np.max(lengths_pre)) if lengths_pre else 0,
                "mean": float(np.mean(lengths_pre)) if lengths_pre else 0.0,
                "median": float(np.median(lengths_pre)) if lengths_pre else 0.0,
                "std": float(np.std(lengths_pre)) if lengths_pre else 0.0
            }
        }
        
        return processed_records, metrics

    train_processed, train_metrics = process_dataset(train_final, train_src, train_lbl, "train")
    val_processed, val_metrics = process_dataset(val_final, val_src, val_lbl, "validation")
    test_processed, test_metrics = process_dataset(test_final, test_src, test_lbl, "test")
    
    # 6. Save outputs
    print("\n[6/6] Saving preprocessed outputs and metadata...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save datasets in JSONL format
    def save_jsonl(records: List[Dict[str, Any]], filename: str) -> None:
        path = os.path.join(args.output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec) + '\n')
                
    save_jsonl(train_processed, "train.jsonl")
    save_jsonl(val_processed, "validation.jsonl")
    save_jsonl(test_processed, "test.jsonl")
    
    # Save vocab
    vocab_path = os.path.join(args.output_dir, "vocabulary.json")
    vocab.save(vocab_path)
    
    # Save tokenizer config
    tok_config_path = os.path.join(args.output_dir, "tokenizer_config.json")
    with open(tok_config_path, 'w', encoding='utf-8') as f:
        json.dump({
            "tokenizer_type": "SourceCodeTokenizer",
            "special_tokens": {
                "pad": Vocabulary.PAD_TOKEN,
                "unk": Vocabulary.UNK_TOKEN,
                "bos": Vocabulary.BOS_TOKEN,
                "eos": Vocabulary.EOS_TOKEN
            },
            "special_indices": {
                "pad": Vocabulary.PAD_IDX,
                "unk": Vocabulary.UNK_IDX,
                "bos": Vocabulary.BOS_IDX,
                "eos": Vocabulary.EOS_IDX
            },
            "max_length": args.max_len,
            "vocab_size": len(vocab),
            "min_freq": args.min_freq
        }, f, indent=4)
        
    # Save preprocessing metadata
    metadata = {
        "dataset_metadata": {
            "train_raw_file": os.path.abspath(args.train_raw),
            "val_raw_file": os.path.abspath(args.val_raw),
            "test_raw_file": os.path.abspath(args.test_raw),
            "max_sequence_length": args.max_len
        },
        "cleaning_statistics": {
            "train": train_stats,
            "validation": val_stats,
            "test": test_stats
        },
        "leakage_statistics": {
            "train_leakage_removed": num_train_leakage,
            "val_leakage_removed": num_val_leakage
        },
        "splits_report": {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics
        },
        "vocabulary": {
            "vocab_size": len(vocab),
            "min_frequency": args.min_freq
        }
    }
    
    metadata_path = os.path.join(args.output_dir, "preprocessing_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
        
    print("\n" + "=" * 60)
    print("Preprocessing completed successfully!")
    print(f"  Processed datasets saved to: {os.path.abspath(args.output_dir)}")
    print(f"  Vocabulary saved to:         {vocab_path}")
    print(f"  Metadata saved to:           {metadata_path}")
    print("=" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AI Security Copilot Preprocessing pipeline")
    parser.add_argument('--train_raw', type=str, required=True, help="Path to raw train dataset")
    parser.add_argument('--val_raw', type=str, required=True, help="Path to raw validation dataset")
    parser.add_argument('--test_raw', type=str, required=True, help="Path to raw test dataset")
    parser.add_argument('--output_dir', type=str, default="data/processed", help="Output directory")
    parser.add_argument('--max_len', type=str, default=256, help="Configurable maximum sequence length")
    parser.add_argument('--vocab_size', type=str, default=10000, help="Configurable vocabulary size")
    parser.add_argument('--min_freq', type=str, default=2, help="Minimum token frequency for vocabulary")
    
    args = parser.parse_args()
    
    # Cast integers safely from parser
    args.max_len = int(args.max_len)
    args.vocab_size = int(args.vocab_size)
    args.min_freq = int(args.min_freq)
    
    run_preprocessing(args)
