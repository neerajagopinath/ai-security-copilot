import os
import json
import pytest
import pandas as pd
import numpy as np

from src.preprocessing.loader import load_data, detect_columns
from src.preprocessing.inspect_dataset import run_inspection

@pytest.fixture
def temp_files(tmp_path):
    """Fixture to create temporary files for format verification."""
    data = [
        {"id": 1, "func": "void check() { int x = 0; }", "target": 0},
        {"id": 2, "func": "void main() { int y = 1; }", "target": 1}
    ]
    df = pd.DataFrame(data)
    
    # 1. Parquet
    parquet_path = tmp_path / "test_data.parquet"
    df.to_parquet(parquet_path)
    
    # 2. JSONL
    jsonl_path = tmp_path / "test_data.jsonl"
    df.to_json(jsonl_path, orient="records", lines=True)
    
    # 3. JSON standard list
    json_path = tmp_path / "test_data.json"
    df.to_json(json_path, orient="records", lines=False)
    
    # 4. Invalid extension
    invalid_path = tmp_path / "test_data.txt"
    with open(invalid_path, "w") as f:
        f.write("text data")
        
    return {
        "parquet": str(parquet_path),
        "jsonl": str(jsonl_path),
        "json": str(json_path),
        "invalid": str(invalid_path)
    }

def test_load_data_file_not_found():
    """Verify that FileNotFoundError is raised for non-existent files."""
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.parquet")

def test_load_data_invalid_extension(temp_files):
    """Verify that unsupported extensions raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_data(temp_files["invalid"])

def test_load_data_formats(temp_files):
    """Verify that Parquet, JSON, and JSONL formats load successfully into DataFrames."""
    df_parquet = load_data(temp_files["parquet"])
    assert len(df_parquet) == 2
    assert "func" in df_parquet.columns
    
    df_jsonl = load_data(temp_files["jsonl"])
    assert len(df_jsonl) == 2
    
    df_json = load_data(temp_files["json"])
    assert len(df_json) == 2

def test_detect_columns_keywords():
    """Verify direct column keyword matching for source code and label columns."""
    df1 = pd.DataFrame({"func": ["code"], "target": [1]})
    src, val = detect_columns(df1)
    assert src == "func"
    assert val == "target"
    
    df2 = pd.DataFrame({"code": ["code"], "label": [0]})
    src, val = detect_columns(df2)
    assert src == "code"
    assert val == "label"

def test_detect_columns_fallback():
    """Verify fallback detection logic (longest average text string for code, binary column for label)."""
    # Source code is the column with long average strings ('some_long_column')
    # Label is the binary integer column ('binary_indicator')
    df = pd.DataFrame({
        "metadata_id": [101, 102],
        "some_long_column": ["int main() { return 0; }", "void test() { char *p = NULL; }"],
        "binary_indicator": [0, 1]
    })
    src, val = detect_columns(df)
    assert src == "some_long_column"
    assert val == "binary_indicator"

def test_duplicate_and_conflict_detection(tmp_path):
    """Verify that duplicate, exact duplicate, and conflicting duplicate counts are calculated correctly."""
    # Data summary:
    # Row 0: Code A, Label 0 (unique)
    # Row 1: Code B, Label 1
    # Row 2: Code B, Label 1 (Exact duplicate of Row 1)
    # Row 3: Code C, Label 0
    # Row 4: Code C, Label 1 (Conflicting duplicate of Row 3)
    # Row 5: Empty code string
    # Row 6: Null code string
    data = [
        {"func": "void code_a() {}", "target": 0},
        {"func": "void code_b() {}", "target": 1},
        {"func": "void code_b() {}", "target": 1},  # Exact Duplicate
        {"func": "void code_c() {}", "target": 0},
        {"func": "void code_c() {}", "target": 1},  # Conflicting Duplicate (label differs)
        {"func": "   ", "target": 0},                # Empty string
        {"func": None, "target": 0}                  # Null value
    ]
    df = pd.DataFrame(data)
    parquet_path = tmp_path / "mock_dupes.parquet"
    df.to_parquet(parquet_path)
    
    fig_dir = tmp_path / "figs"
    met_dir = tmp_path / "metrics"
    
    summary = run_inspection(str(parquet_path), str(fig_dir), str(met_dir))
    
    # 7 rows total
    assert summary["total_samples"] == 7
    
    # Empty code count should be 2 (null + whitespace only)
    assert summary["quality_checks"]["empty_source_code_samples"] == 2
    
    # Missing source values is 1 (Null)
    assert summary["quality_checks"]["missing_values"]["source_code"] == 1
    
    # Clean duplicates check ignores NAs/Nulls:
    # Valid non-null code strings in clean:
    # A, B, B, C, C, '   ' (since whitespace-only is non-null)
    # Total duplicates of code strings: 'B' is duplicated once, 'C' is duplicated once.
    # Total duplicate code samples = 2
    assert summary["quality_checks"]["total_duplicate_code_samples"] == 2
    
    # Exact duplicate code and label:
    # ('code_b', 1) is duplicated once.
    # ('code_c', 0) vs ('code_c', 1) has different labels, so not an exact duplicate.
    assert summary["quality_checks"]["exact_duplicates"] == 1
    
    # Conflicting duplicates:
    # 'code_c' maps to {0, 1} which has cardinality 2 > 1.
    # Conflicting unique codes = 1 ('code_c')
    # Conflicting samples count = 2
    assert summary["quality_checks"]["conflicting_duplicates"]["unique_conflicting_codes"] == 1
    assert summary["quality_checks"]["conflicting_duplicates"]["conflicting_samples"] == 2
