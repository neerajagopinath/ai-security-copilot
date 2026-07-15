# AI Security Copilot
## Source Code Vulnerability Detection & Secure Fix Recommendation

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 1. Problem Statement

Software vulnerabilities are a leading cause of security breaches worldwide. Manual code review is slow, expensive, and error-prone. Automated static analysis tools based on syntactic patterns produce high false-positive rates and miss semantic vulnerabilities.

This project applies **deep learning** to binary vulnerability detection — classifying whether a given code function is potentially vulnerable — using the real-world Devign dataset from open-source C/C++ projects (QEMU, FFmpeg, Linux kernel, OpenSSL).

---

## 2. Why This Project Matters

- **Scale**: Devign contains ~27,000 real CVE-tagged functions from production codebases.
- **Research relevance**: Combines classical sequence modeling (Bi-LSTM) with state-of-the-art transformer models (GraphCodeBERT).
- **Practical output**: The security analyzer provides actionable recommendations, not just a binary label.
- **Reproducibility**: Complete pipeline from raw data to inference with official dataset splits.

---

## 3. Features

| Feature | Status |
|---|---|
| Source-code-aware tokenizer (C/C++) | ✅ Complete |
| Vocabulary builder (train-split only, no leakage) | ✅ Complete |
| Official Devign train/val/test splits | ✅ Complete |
| PyTorch Bi-LSTM vulnerability classifier | ✅ Complete |
| Bidirectional LSTM with packed sequence handling | ✅ Complete |
| BCEWithLogitsLoss + AdamW + gradient clipping | ✅ Complete |
| Validation F1 model selection + early stopping | ✅ Complete |
| GraphCodeBERT integration infrastructure | ✅ Complete |
| Deterministic rule-based security analysis | ✅ Complete |
| FastAPI backend (4 endpoints) | ✅ Complete |
| Streamlit demo UI | ✅ Complete |
| Demo checkpoint generation | ✅ Complete |
| GPU training support | ✅ Ready |
| Full test suite (39+ tests) | ✅ Complete |

---

## 4. Architecture

```
Source Code Input
       │
       ├─────────────────────────┐
       │                         │
  [Bi-LSTM Path]          [GraphCodeBERT Path]
       │                         │
  SourceCodeTokenizer     HuggingFace Tokenizer
       │                         │
  Vocabulary Lookup       Subword Tokenization
       │                         │
  Embedding Layer         Transformer Encoder
       │                         │
  Bidirectional LSTM      [CLS] Representation
       │                         │
  Dropout + Linear        Dropout + Linear
       │                         │
  Vulnerability Logit     Vulnerability Logit
       │                         │
       └────────┬────────────────┘
                │
    Security Analysis Service
                │
       ├── ML Model Prediction
       ├── Rule-Based Pattern Detection
       ├── Category Classification
       ├── Explanation Generation
       ├── Security Recommendations
       └── Secure Code Suggestions
                │
         FastAPI Backend
                │
         Streamlit UI
```

---

## 5. Technology Stack

| Layer | Technology |
|---|---|
| ML Framework | PyTorch 2.1+ |
| Transformer | HuggingFace Transformers |
| Data Processing | pandas, pyarrow, NumPy |
| Metrics | scikit-learn |
| Backend API | FastAPI + Uvicorn |
| Frontend UI | Streamlit |
| Testing | pytest |
| Config | PyYAML |
| Visualization | Matplotlib |

---

## 6. Dataset

**Devign** — A manually annotated dataset of real-world C/C++ vulnerabilities from open-source projects:

| Split | Size | Source |
|---|---|---|
| Train | ~21,854 samples | HuggingFace Hub |
| Validation | ~2,732 samples | HuggingFace Hub |
| Test | ~2,732 samples | HuggingFace Hub |
| Total | ~27,318 samples | Zhou et al., NeurIPS 2019 |

- **Features**: Function-level C/C++ source code
- **Label**: Binary (0 = non-vulnerable, 1 = vulnerable)
- **Sources**: QEMU, FFmpeg, Linux kernel, OpenSSL
- **Format**: Parquet files stored in `data/raw/`

---

## 7. Bi-LSTM Model

```
Pipeline:
  Source Code
    → SourceCodeTokenizer (regex-based, C/C++-aware)
    → Token IDs (via Vocabulary lookup)
    → Embedding Layer (trainable, vocab_size × embedding_dim)
    → Bidirectional LSTM (num_layers, hidden_dim per direction)
    → Dropout (applied to embedding + final representation)
    → Linear Classification Head (hidden_dim×2 → 1)
    → Sigmoid → Vulnerability Probability

Key Design Choices:
  - Packed sequences: padding-aware, variable-length safe
  - BCEWithLogitsLoss with pos_weight for class imbalance
  - AdamW optimizer with weight decay
  - Gradient clipping (max_norm=5.0)
  - ReduceLROnPlateau scheduler (monitors validation F1)
  - Early stopping (patience=3)
  - Model selection: best validation F1 (tie-breaker: val loss)
```

**Model hyperparameters** (see `configs/config.yaml`):

| Parameter | Value |
|---|---|
| Embedding dim | 128 |
| Hidden dim (per direction) | 256 |
| LSTM layers | 2 |
| Dropout | 0.5 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Max sequence length | 256 tokens |
| Vocabulary size | 10,000 |

---

## 8. GraphCodeBERT Extension

GraphCodeBERT (`microsoft/graphcodebert-base`) is a transformer pre-trained on code that understands both natural language and programming language structure.

```
Pipeline:
  Source Code
    → AutoTokenizer (subword tokenization)
    → RobertaModel (12-layer transformer, 768-dim hidden)
    → [CLS] token representation
    → Dropout + Linear head
    → Vulnerability logit

Training setup:
  - Fine-tune all transformer layers + classification head
  - AdamW with lr=2e-5, warmup schedule
  - BCEWithLogitsLoss with pos_weight
  - Batch size: 16 (requires ≥16GB VRAM for full fine-tuning)
```

> ⚠️ **GPU Requirement**: GraphCodeBERT requires a GPU for practical fine-tuning.
> On CPU, one epoch over the full Devign dataset takes several hours.
> Full fine-tuning performance is **NOT YET MEASURED — FULL GPU TRAINING REQUIRED**.

---

## 9. Vulnerability Analysis Workflow

```
1. Input: source code + language + selected model
2. Run ML model inference → vulnerability probability
3. Run deterministic rule-based security pattern detection:
   - Unsafe memory operations (memcpy, malloc, free)
   - Unsafe string functions (strcpy, strcat, sprintf, gets)
   - Command execution (system, popen, exec family)
   - SQL query string construction
   - Hard-coded credentials
   - Weak cryptographic algorithms (MD5, SHA-1, DES)
   - Missing input validation (atoi, atol, atof)
   - Dangerous file operations
   - Format string vulnerabilities
   - Integer boundary risks
4. Combine ML + rule findings → final probability
5. Generate explanation, recommendations, suggested secure code
6. Add disclaimer: findings require developer review
```

**Key principles:**
- Rule-based findings are NEVER attributed to ML predictions
- No exact CWE numbers claimed without strong evidence
- Hedged language: "potential", "possible", "requires review"
- Submitted code is NEVER executed, compiled, or eval'd

---

## 10. Project Structure

```
ai-security-copilot/
├── app.py                        # Streamlit demo UI
├── smoke_test_bilstm.py          # Smoke test + demo checkpoint generator
├── main.py                       # Entry point (info)
├── configs/
│   └── config.yaml               # All model and training parameters
├── api/
│   ├── __init__.py
│   └── main.py                   # FastAPI backend (4 endpoints)
├── src/
│   ├── models/
│   │   ├── bilstm.py             # Bi-LSTM classifier
│   │   ├── graphcodebert.py      # GraphCodeBERT classifier + utilities
│   │   └── verify_bilstm.py      # Model verification script
│   ├── preprocessing/
│   │   ├── loader.py             # Multi-format data loader
│   │   ├── preprocess.py         # Full preprocessing pipeline
│   │   ├── tokenizer.py          # C/C++ source-code tokenizer
│   │   ├── vocabulary.py         # Vocabulary builder (train-only)
│   │   ├── dataset.py            # PyTorch Dataset + DataLoader
│   │   └── inspect_dataset.py    # EDA and validation
│   ├── training/
│   │   ├── train.py              # Bi-LSTM training pipeline
│   │   ├── train_utils.py        # Metrics, checkpoints, utilities
│   │   └── train_graphcodebert.py # GraphCodeBERT training
│   ├── inference/
│   │   ├── model_manager.py      # Model loading + inference
│   │   └── security_analyzer.py  # Security analysis pipeline
│   ├── evaluation/
│   │   └── evaluate_graphcodebert.py  # GraphCodeBERT evaluation
│   └── utils/
│       └── seed.py               # Reproducibility seed
├── data/
│   ├── raw/                      # Original Devign parquet files (do not modify)
│   └── processed/                # Preprocessed JSONL, vocabulary, metadata
├── models/
│   ├── checkpoints/              # Training + demo checkpoints
│   └── saved/                    # Production model directory
├── tests/
│   ├── test_bilstm.py            # Bi-LSTM model tests
│   ├── test_training.py          # Training pipeline tests
│   ├── test_preprocessing.py     # Preprocessing tests
│   ├── test_loader.py            # Data loader tests
│   └── test_api_and_analyzer.py  # API + security analyzer tests
├── results/
│   ├── figures/                  # Training curves
│   └── metrics/                  # Training history JSON
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 11. Installation

### Prerequisites
- Python 3.11+
- (Optional but recommended) CUDA-capable GPU for GraphCodeBERT training

### Steps

```bash
# 1. Clone or navigate to project directory
cd ai-security-copilot

# 2. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 12. How to Run Preprocessing

The raw Devign parquet files are already in `data/raw/`. Processed data is already in `data/processed/`. To re-run preprocessing:

```bash
python -m src.preprocessing.preprocess \
    --train_raw data/raw/train-00000-of-00001.parquet \
    --val_raw data/raw/validation-00000-of-00001.parquet \
    --test_raw data/raw/test-00000-of-00001.parquet \
    --output_dir data/processed \
    --max_len 256 \
    --vocab_size 10000 \
    --min_freq 2
```

---

## 13. How to Run Bi-LSTM Training (CPU)

```bash
# Standard training (reads from config.yaml)
python -m src.training.train --config configs/config.yaml

# Quick smoke test (CPU-safe, bounded, generates demo checkpoint)
python smoke_test_bilstm.py

# Override specific parameters
python -m src.training.train \
    --config configs/config.yaml \
    --epochs 5 \
    --batch_size 32 \
    --lr 0.001
```

---

## 14. How to Run Full GPU Training

```bash
# Full GPU training with all data
python -m src.training.train \
    --config configs/config.yaml \
    --device cuda \
    --epochs 10 \
    --batch_size 64

# Resume from checkpoint
python -m src.training.train \
    --config configs/config.yaml \
    --device cuda \
    --resume models/checkpoints/latest_checkpoint.pt
```

### GraphCodeBERT GPU Training

```bash
# Full GraphCodeBERT fine-tuning (GPU required)
python -m src.training.train_graphcodebert \
    --train_path data/processed/train.jsonl \
    --val_path data/processed/validation.jsonl \
    --output_dir models/checkpoints \
    --epochs 3 \
    --batch_size 16 \
    --lr 2e-5 \
    --device cuda

# GraphCodeBERT smoke test (CPU-safe, 2-batch verification)
python -m src.training.train_graphcodebert --smoke_test
```

---

## 15. How to Run the API

```bash
# Start the FastAPI server
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

# API will be available at:
#   http://127.0.0.1:8000
#   http://127.0.0.1:8000/docs  (Swagger UI)
#   http://127.0.0.1:8000/redoc (ReDoc)
```

---

## 16. How to Run Streamlit

> Requires the FastAPI server to be running first.

```bash
# Start the demo UI
streamlit run app.py
```

The UI will open at `http://localhost:8501`

---

## 17. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check and model status |
| GET | `/models` | List available models |
| POST | `/predict` | Quick prediction (probability only) |
| POST | `/analyze` | Full vulnerability analysis |

### GET /health — Response

```json
{
  "status": "ok",
  "version": "1.0.0",
  "bilstm_status": "demo",
  "device": "cpu",
  "timestamp": 1720000000.0
}
```

### POST /analyze — Request

```json
{
  "code": "void copy_data(char *src) {\n  char buf[64];\n  strcpy(buf, src);\n}",
  "model": "bilstm",
  "language": "C/C++"
}
```

### POST /analyze — Response

```json
{
  "model": "bilstm",
  "language": "C/C++",
  "prediction": "Potentially Vulnerable",
  "vulnerability_probability": 0.82,
  "confidence": 0.82,
  "model_mode": "demo",
  "potential_category": "Possible Buffer Overflow Risk",
  "suspicious_patterns": [
    "Unsafe string function without bounds checking (strcpy/strcat/sprintf/gets)"
  ],
  "rule_matches_count": 1,
  "explanation": "...",
  "recommendations": [
    "Replace unsafe string functions (strcpy, strcat, sprintf, gets) with strncpy, strncat, snprintf, fgets.",
    "Run static analysis tools (e.g., Coverity, CodeQL, Semgrep) for deeper inspection."
  ],
  "suggested_code": "/* TODO: replace with strncpy */ strcpy(buf, src);",
  "disclaimer": "DISCLAIMER: This analysis is provided for educational..."
}
```

---

## 18. Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_bilstm.py -v
python -m pytest tests/test_training.py -v
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_loader.py -v
python -m pytest tests/test_api_and_analyzer.py -v

# Run with coverage (if pytest-cov installed)
python -m pytest tests/ -v --cov=src --cov=api
```

---

## 19. Current Model Status

| Model | Status | Notes |
|---|---|---|
| Bi-LSTM (demo) | ✅ Demo checkpoint available | 2-epoch, 32-sample smoke run |
| Bi-LSTM (full) | ⏳ Not yet trained | Requires GPU training on full dataset |
| GraphCodeBERT | ⏳ Not yet fine-tuned | Requires GPU |

### Genuine Metrics Available

> **NOT YET MEASURED — FULL GPU TRAINING REQUIRED**

The demo checkpoint was trained for only 2 epochs on 32 samples for pipeline verification. Its metrics are NOT representative of model capability.

Full Bi-LSTM training on the complete Devign dataset (GPU, 10 epochs) is expected to yield results competitive with published baselines (~60-70% F1).

GraphCodeBERT fine-tuning (GPU, 3 epochs) is expected to yield results approaching state-of-the-art (~70-75% F1 based on published literature).

---

## 20. Limitations

1. **Demo mode**: Without full GPU training, ML predictions are approximate.
2. **Language scope**: Bi-LSTM tokenizer is optimized for C/C++. Other languages use the same tokenizer with reduced effectiveness.
3. **Context window**: Maximum 256 tokens (Bi-LSTM) or 512 tokens (GraphCodeBERT) — very long functions are truncated.
4. **False positives**: Rule-based detection will flag safe uses of flagged functions.
5. **False negatives**: The model cannot detect logic errors, authentication flaws, or business logic vulnerabilities.
6. **No interprocedural analysis**: Each function is analyzed independently; cross-function vulnerabilities are not detected.
7. **Not a replacement**: This is a research tool, not a production security scanner.

---

## 21. Future Enhancements

- [ ] Full GPU training on Devign with proper evaluation
- [ ] Multi-class vulnerability categorization (CWE-level)
- [ ] Graph Neural Network (GNN) model for code structure
- [ ] Interprocedural/cross-file analysis
- [ ] Support for Python, Java, Go vulnerability patterns
- [ ] Integration with IDE plugins (VS Code extension)
- [ ] CI/CD pipeline integration
- [ ] SARIF output format for GitHub Security tab

---

## 22. Resume Description

**AI Security Copilot — Source Code Vulnerability Detection System**
*Personal Project | Python, PyTorch, FastAPI, Streamlit | 2024*

- Built an end-to-end ML pipeline for binary vulnerability detection on the Devign dataset (~27K C/C++ functions from QEMU, FFmpeg, Linux kernel, OpenSSL).
- Implemented a Bidirectional LSTM classifier with padding-aware packed sequence handling, BCEWithLogitsLoss with class-imbalance weighting, AdamW optimizer, and early stopping on validation F1.
- Integrated GraphCodeBERT (microsoft/graphcodebert-base) infrastructure for transformer-based classification with full GPU fine-tuning support.
- Developed a deterministic rule-based security analyzer detecting 13 vulnerability categories (buffer overflows, command injection, SQL injection, weak crypto, etc.).
- Built a production-ready FastAPI backend with Pydantic validation, input size limits, CORS, logging, and Swagger documentation.
- Created a professional Streamlit demo UI with dark theme, code analysis display, and actionable security recommendations.
- Wrote 39+ unit and integration tests covering model inference, API endpoints, security patterns, and fallback behavior.

---

## 23. Interview Explanation

> "How does your AI Security Copilot work?"

The system has three layers:

**Layer 1 — ML Model**: I train a Bidirectional LSTM on the Devign dataset. The model takes C/C++ source code, tokenizes it with a regex-based tokenizer that handles operators, literals, comments, and identifiers, looks up token IDs in a trained vocabulary, embeds them, processes them through a 2-layer BiLSTM with packed sequences (to handle variable lengths correctly), and outputs a single binary classification logit. I use BCEWithLogitsLoss with pos_weight for class imbalance, AdamW for optimization, and select the best model by validation F1.

**Layer 2 — Rules Engine**: Independently from the ML model, I apply 13 regex-based security patterns to detect known dangerous patterns like strcpy, system(), hard-coded passwords, MD5, etc. These findings are always clearly labeled as rule-based, never as ML predictions.

**Layer 3 — API + UI**: The FastAPI backend loads models once at startup, accepts code strings via POST /analyze, runs both layers, and returns a structured JSON response with probability, category, explanation, recommendations, and suggested fixes. The Streamlit UI provides a clean demo interface.

Key security property: submitted code is NEVER executed, compiled, or eval'd — it's only analyzed as text.
