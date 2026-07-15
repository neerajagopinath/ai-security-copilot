# Data Directory

This directory contains the datasets used for training and evaluating the vulnerability detection models.

## Structure

- `raw/`: Raw, unaltered source code datasets. Place raw Devign dataset files here.
- `processed/`: Processed, cleaned, tokenized, or vectorized representations of the source code suitable for model ingestion.

---

## Devign Dataset Acquisition and Placement

The Devign dataset is a benchmark dataset containing source-code functions labeled as vulnerable (`1`) or non-vulnerable (`0`).

### 1. Where to Obtain the Dataset
You can acquire the Devign dataset from:
*   **Hugging Face Hub**: The dataset is hosted on Hugging Face (e.g., as part of the CodeXGLUE defect detection task under `google/code_x_glue_cc_defect_detection` or community copies like `DetectVul/devign`).
*   **Official Repository**: The original paper source files can be retrieved from the authors' GitHub repository: [epicosy/devign](https://github.com/epicosy/devign).

### 2. Supported Formats
Our ingestion pipeline is format-agnostic and supports:
*   **Parquet**: e.g., `train-00000-of-00001.parquet` (default from Hugging Face Datasets cache downloads).
*   **JSONL / JSON**: e.g., `train.jsonl` or `train.json`.

### 3. Folder Placement
To run inspection and analysis scripts, place your raw files directly into `data/raw/`. For example:
```text
data/
└── raw/
    ├── train-00000-of-00001.parquet
    ├── validation-00000-of-00001.parquet
    └── test-00000-of-00001.parquet
```

> [!WARNING]
> Large dataset files under `raw/` and `processed/` are ignored by git to keep the repository lightweight. Do not force-commit them to the repository.
