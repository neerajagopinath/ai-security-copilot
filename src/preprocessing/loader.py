import os
import pandas as pd
from typing import Tuple

def load_data(file_path: str) -> pd.DataFrame:
    """
    Safely load a dataset from a file. Supports Parquet, JSON, and JSONL formats.
    
    Args:
        file_path (str): Path to the dataset file.
        
    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported or loading fails.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
        
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == '.parquet':
            return pd.read_parquet(file_path)
        elif ext == '.jsonl':
            return pd.read_json(file_path, lines=True)
        elif ext == '.json':
            # Peek at the first character to differentiate between JSON list and JSON lines
            first_char = ''
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        first_char = stripped[0]
                        break
            
            if first_char == '[':
                return pd.read_json(file_path, lines=False)
            else:
                try:
                    return pd.read_json(file_path, lines=True)
                except Exception:
                    return pd.read_json(file_path, lines=False)
        else:
            raise ValueError(f"Unsupported file extension '{ext}'. Only .parquet, .json, and .jsonl are supported.")
    except Exception as e:
        raise ValueError(f"Failed to load dataset file at {file_path}: {e}")

def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Automatically detect the source code and label columns in a DataFrame.
    
    Args:
        df (pd.DataFrame): The loaded dataset.
        
    Returns:
        Tuple[str, str]: A tuple containing (source_code_column, label_column).
        
    Raises:
        ValueError: If source code or label columns cannot be determined.
    """
    if df.empty:
        raise ValueError("Cannot detect columns of an empty DataFrame.")
        
    columns = [col.strip() for col in df.columns]
    columns_lower = [col.lower() for col in columns]
    
    # 1. Detect Source Code Column
    source_candidates = ["func", "code", "source", "source_code", "text", "function_body", "input_code"]
    source_col = None
    
    # Check for direct keyword matches
    for candidate in source_candidates:
        if candidate in columns_lower:
            idx = columns_lower.index(candidate)
            source_col = columns[idx]
            break
            
    # Fallback: find the string/object column with the longest average length
    if source_col is None:
        string_cols = []
        for col in df.columns:
            # Check if column is object or string type
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                # Take sample and compute average length
                avg_len = df[col].astype(str).str.len().mean()
                string_cols.append((col, avg_len))
        if string_cols:
            # Sort by average length descending
            string_cols.sort(key=lambda x: x[1], reverse=True)
            source_col = string_cols[0][0]
            
    if source_col is None:
        raise ValueError("Could not automatically detect the source code column.")

    # 2. Detect Label Column
    label_candidates = ["target", "label", "vuln", "vulnerability", "class", "defect"]
    label_col = None
    
    # Check for direct keyword matches
    for candidate in label_candidates:
        if candidate in columns_lower:
            idx = columns_lower.index(candidate)
            label_col = columns[idx]
            break
            
    # Fallback: check for integer/boolean columns representing binary values {0, 1}
    if label_col is None:
        binary_candidates = []
        for col in df.columns:
            if col == source_col:
                continue
            # Check if numeric or boolean
            if (pd.api.types.is_integer_dtype(df[col]) or 
                pd.api.types.is_bool_dtype(df[col]) or 
                pd.api.types.is_numeric_dtype(df[col])):
                unique_vals = set(df[col].dropna().unique())
                # Must be binary {0, 1} or booleans
                if unique_vals.issubset({0, 1, 0.0, 1.0, True, False}) and len(unique_vals) <= 2:
                    binary_candidates.append(col)
                    
        if binary_candidates:
            # If multiple, prefer columns containing 'target' or 'label' substrings, else the first one
            label_col = binary_candidates[0]
            
    if label_col is None:
        raise ValueError("Could not automatically detect the binary label column.")
        
    return source_col, label_col
