# AI Security Copilot

## Overview
AI Security Copilot is a deep-learning-based source code vulnerability detection system designed to identify security risks in C/C++ code. It combines machine learning models trained on the Devign dataset with deterministic rule-based security pattern analysis to provide actionable recommendations. The system features a FastAPI backend for inference and a Streamlit frontend for user interaction.

## Features
- **Bi-LSTM Vulnerability Classifier**: A PyTorch Bidirectional LSTM model trained to classify source code vulnerabilities.
- **GraphCodeBERT Integration**: HuggingFace transformer support for state-of-the-art vulnerability detection.
- **Rule-Based Security Analysis**: Deterministic detection of 13 vulnerability categories (e.g., buffer overflows, command injection, weak crypto) using regex patterns.
- **Unified Security Assessment**: Safe merging of ML predictions and rule-based findings into a cohesive, prioritized security rating.
- **FastAPI Backend**: Production-ready API with robust error handling and model management.
- **Streamlit Demo UI**: Interactive frontend for code submission and result visualization.
- **Dynamic Model Selection**: Both Bi-LSTM and GraphCodeBERT models are loaded into memory concurrently, allowing seamless switching.

## System Architecture
- **Streamlit Frontend**: A lightweight, interactive web application (`app.py`) for users to submit code snippets and view analysis results across multiple tabs (Explanation, Recommendations, Suggested Fixes).
- **FastAPI Backend**: A RESTful API (`api/main.py`) that manages inference requests, model loading, and routes inputs to the analysis engine.
- **Bi-LSTM Model**: A recurrent neural network utilizing packed sequences and a custom source code tokenizer, optimized for learning syntactic vulnerability patterns.
- **GraphCodeBERT Model**: A pre-trained transformer model (`microsoft/graphcodebert-base`) fine-tuned for sequence classification on source code.
- **Rule-Based Security Analyzer**: A deterministic engine that scans code against known dangerous patterns, providing strict, understandable findings.
- **Unified Security Assessment**: An overarching logic layer that combines the probabilistic ML output with deterministic rule severities to ensure safe, non-contradictory vulnerability reporting.

## Supported Languages
- **C/C++** (Fully Supported)
- Python, Java, JavaScript, Go, Rust (Available in the UI dropdown, but analysis is currently C/C++ focused)

*Note: The ML models (Bi-LSTM and GraphCodeBERT) were trained exclusively on the Devign C/C++ dataset. The deterministic rule-based analysis is also heavily optimized for C/C++ memory operations and standard libraries.*

## Project Structure
```text
ai-security-copilot/
├── api/
│   └── main.py                   # FastAPI backend endpoints
├── configs/
│   └── config.yaml               # Model configuration parameters
├── data/
│   ├── processed/                # Tokenized JSONL and vocabulary
│   └── raw/                      # Original Devign dataset (parquet)
├── models/
│   └── checkpoints/              # Model weights and configs
├── src/
│   ├── evaluation/               # Model evaluation scripts
│   ├── inference/                # Model manager and security analyzer
│   ├── models/                   # PyTorch architectures (Bi-LSTM, GraphCodeBERT)
│   ├── preprocessing/            # Tokenization and data loading
│   └── training/                 # Training loops
├── tests/                        # Pytest suite
├── app.py                        # Streamlit frontend
└── requirements.txt              # Project dependencies
```

## Installation
```bash
git clone https://github.com/your-username/ai-security-copilot.git
cd ai-security-copilot
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Running the Project

**Backend:**
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

**Frontend:**
```bash
streamlit run app.py
```

## Models

### Bi-LSTM
A custom PyTorch Bidirectional LSTM classifier utilizing a regex-based C/C++ tokenizer. It learns syntactic vulnerability patterns from the Devign dataset.

### GraphCodeBERT
A fine-tuned `microsoft/graphcodebert-base` transformer that leverages both natural language and programming language context for robust vulnerability detection. 
- The model configuration, tokenizer, and training metrics are included in the repository.
- **Important**: The `model.safetensors` weight file is excluded from Git due to GitHub's file size limit.
- Users must retrain the model locally or manually place their own `model.safetensors` in the `models/checkpoints/graphcodebert_tuned/` directory.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | API health check and model loading status |
| GET | `/models` | List all available loaded models |
| POST | `/predict` | Quick binary vulnerability prediction (probability only) |
| POST | `/analyze` | Comprehensive ML and rule-based vulnerability analysis |

## Screenshots
*(Add relevant screenshots of the Streamlit interface here)*

## Current Status
- Bi-LSTM integrated
- GraphCodeBERT integrated
- Streamlit connected to FastAPI
- Rule-based vulnerability detection implemented (13 categories)
- Unified security assessment operational
- Dynamic model selection via API fetching
- Test suite passing

## Future Improvements
- Multi-language AST parsing (e.g., Tree-sitter) for accurate syntax analysis
- Genuine multi-language ML and rule-based support beyond C/C++
- Improved automated code fixes beyond simple string substitution
- Full GPU training runs for Bi-LSTM to establish definitive baseline metrics
