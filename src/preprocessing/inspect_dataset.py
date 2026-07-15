import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing.loader import load_data, detect_columns

def sanitize_code(code_str: str, max_lines: int = 8, max_chars: int = 350) -> str:
    """
    Sanitize source code snippet for safe console display.
    """
    if not isinstance(code_str, str):
        return str(code_str)
    
    # Strip leading/trailing whitespace
    code_str = code_str.strip()
    
    # Limit by lines
    lines = code_str.splitlines()
    if len(lines) > max_lines:
        truncated = "\n".join(lines[:max_lines]) + f"\n... [Truncated: {len(lines) - max_lines} more lines] ..."
    else:
        truncated = code_str
        
    # Limit by characters
    if len(truncated) > max_chars:
        truncated = truncated[:max_chars] + "\n... [Truncated due to character limit] ..."
        
    return truncated

def run_inspection(file_path: str, figures_dir: str, metrics_dir: str) -> dict:
    """
    Load and inspect the dataset, print summary findings, and save results.
    """
    print(f"\n[1/5] Loading dataset: {file_path}")
    df = load_data(file_path)
    total_samples = len(df)
    print(f"[OK] Loaded {total_samples} samples.")
    
    print("\n[2/5] Inspecting schema...")
    source_col, label_col = detect_columns(df)
    print(f"[OK] Detected source-code column: '{source_col}'")
    print(f"[OK] Detected label column:       '{label_col}'")
    print(f"Available columns: {list(df.columns)}")
    
    # Extract columns and count missing
    missing_source = df[source_col].isnull().sum()
    missing_label = df[label_col].isnull().sum()
    
    # Detect empty strings or whitespace-only code strings
    # Handle non-string types safely
    empty_source = df[source_col].apply(lambda x: len(str(x).strip()) == 0 or pd.isnull(x)).sum()
    
    # Safety Check: Labels validation
    unique_labels = df[label_col].dropna().unique()
    is_binary = set(unique_labels).issubset({0, 1, 0.0, 1.0, True, False})
    
    # Count classes
    label_series = df[label_col].fillna(-1).astype(int)
    counts = label_series.value_counts().to_dict()
    
    # Vulnerable vs Non-Vulnerable
    num_non_vuln = counts.get(0, 0)
    num_vuln = counts.get(1, 0)
    num_invalid = counts.get(-1, 0)
    
    pct_non_vuln = (num_non_vuln / total_samples) * 100 if total_samples > 0 else 0
    pct_vuln = (num_vuln / total_samples) * 100 if total_samples > 0 else 0
    
    print("\n[3/5] Performing dataset quality checks...")
    
    # Duplicate code string analysis
    # Drop NAs for duplicate checks
    clean_code_df = df[[source_col, label_col]].dropna()
    
    # Total duplicates
    total_code_dupes = clean_code_df.duplicated(subset=[source_col]).sum()
    
    # Exact duplicates (same code, same label)
    exact_dupes = clean_code_df.duplicated(subset=[source_col, label_col]).sum()
    
    # Conflicting duplicates (same code, different labels)
    code_groups = clean_code_df.groupby(source_col)[label_col].nunique()
    conflicting_codes = code_groups[code_groups > 1].index
    num_conflicting_unique_codes = len(conflicting_codes)
    num_conflicting_samples = clean_code_df[clean_code_df[source_col].isin(conflicting_codes)].shape[0]
    
    print(f"  - Missing source code values:  {missing_source}")
    print(f"  - Missing label values:        {missing_label}")
    print(f"  - Empty code strings:          {empty_source}")
    print(f"  - Label binary validity:       {'PASSED (strictly binary)' if is_binary else 'FAILED (non-binary labels found)'}")
    print(f"  - Total identical code dupes:  {total_code_dupes}")
    print(f"  - Exact duplicates (same Lbl): {exact_dupes}")
    print(f"  - Conflicting duplicates:      {num_conflicting_samples} samples ({num_conflicting_unique_codes} unique codes)")
    
    print("\n[4/5] Computing code-length statistics...")
    lengths = df[source_col].fillna("").astype(str).str.len()
    len_min = int(lengths.min())
    len_max = int(lengths.max())
    len_mean = float(lengths.mean())
    len_median = float(lengths.median())
    len_std = float(lengths.std())
    
    print(f"  - Min characters:    {len_min}")
    print(f"  - Max characters:    {len_max}")
    print(f"  - Mean characters:   {len_mean:.2f}")
    print(f"  - Median characters: {len_median:.2f}")
    print(f"  - Std dev:           {len_std:.2f}")
    
    # Print Sanitized Samples
    print("\n[Displaying Sanitized Samples]")
    vulnerable_samples = df[df[label_col] == 1].head(1)
    non_vulnerable_samples = df[df[label_col] == 0].head(1)
    
    if not vulnerable_samples.empty:
        print("\n--- Example: VULNERABLE (Label = 1) ---")
        print(sanitize_code(vulnerable_samples[source_col].values[0]))
    if not non_vulnerable_samples.empty:
        print("\n--- Example: NON-VULNERABLE (Label = 0) ---")
        print(sanitize_code(non_vulnerable_samples[source_col].values[0]))
    print("-" * 40)
    
    print("\n[5/5] Generating output figures and metrics...")
    
    # Create output directories if missing
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # 1. Generate class-distribution chart
    # Use a modern color palette: Slate-ish dark theme
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(6, 5))
    
    categories = ['Non-Vulnerable (0)', 'Vulnerable (1)']
    counts_plot = [num_non_vuln, num_vuln]
    colors = ['#34d399', '#f87171'] # Tailwind emerald-400 (green) and red-400 (red)
    
    bars = ax.bar(categories, counts_plot, color=colors, edgecolor='none', width=0.6)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}\n({height/total_samples*100:.1f}%)' if total_samples > 0 else f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
    ax.set_title('Devign Class Distribution', fontsize=14, pad=15, fontweight='bold')
    ax.set_ylabel('Sample Count', fontsize=12)
    ax.set_ylim(0, max(counts_plot) * 1.15 if counts_plot else 10)
    
    # Style tweaks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    fig_path = os.path.join(figures_dir, 'class_distribution.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[OK] Saved class distribution chart to: {fig_path}")
    
    # 2. Save summary metrics
    summary = {
        "dataset_file": os.path.abspath(file_path),
        "total_samples": total_samples,
        "schema": {
            "columns": list(df.columns),
            "detected_source_column": source_col,
            "detected_label_column": label_col
        },
        "quality_checks": {
            "is_binary_valid": bool(is_binary),
            "missing_values": {
                "source_code": int(missing_source),
                "label": int(missing_label)
            },
            "empty_source_code_samples": int(empty_source),
            "total_duplicate_code_samples": int(total_code_dupes),
            "exact_duplicates": int(exact_dupes),
            "conflicting_duplicates": {
                "unique_conflicting_codes": int(num_conflicting_unique_codes),
                "conflicting_samples": int(num_conflicting_samples)
            }
        },
        "class_distribution": {
            "non_vulnerable": {
                "count": int(num_non_vuln),
                "percentage": float(pct_non_vuln)
            },
            "vulnerable": {
                "count": int(num_vuln),
                "percentage": float(pct_vuln)
            },
            "invalid_or_missing_label": int(num_invalid)
        },
        "code_length_statistics": {
            "min_chars": len_min,
            "max_chars": len_max,
            "mean_chars": len_mean,
            "median_chars": len_median,
            "std_dev_chars": len_std
        }
    }
    
    metrics_path = os.path.join(metrics_dir, 'dataset_summary.json')
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"[OK] Saved dataset summary JSON to: {metrics_path}")
    
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Devign Dataset Inspection & Validation")
    parser.add_argument('--input', type=str, required=True, help="Path to raw dataset file")
    parser.add_argument('--figures_dir', type=str, default="results/figures", help="Directory to save figures")
    parser.add_argument('--metrics_dir', type=str, default="results/metrics", help="Directory to save metric summaries")
    
    args = parser.parse_args()
    
    run_inspection(args.input, args.figures_dir, args.metrics_dir)
