"""
Model Manager for Inference
-----------------------------
Loads and manages Bi-LSTM (and optionally GraphCodeBERT) models
for inference. Handles missing checkpoints gracefully by falling
back to rule-based analysis only.

KEY FIX: Architecture parameters (embedding_dim, hidden_dim, num_layers)
are read DIRECTLY from the checkpoint's state-dict tensor shapes, NOT
from configs/config.yaml. This ensures the reconstructed model always
matches the saved weights exactly — regardless of config file values.

Models are loaded ONCE at startup and reused for all requests.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

import torch

logger = logging.getLogger(__name__)

# Paths relative to project root
DEMO_CHECKPOINT_PATH = "models/checkpoints/demo_bilstm_model.pt"
BEST_CHECKPOINT_PATH = "models/checkpoints/best_model.pt"
VOCAB_PATH = "data/processed/vocabulary.json"


def _infer_bilstm_arch_from_state_dict(sd: dict) -> dict:
    """
    Infer Bi-LSTM architecture parameters solely from state-dict tensor shapes.

    Verified mappings (PyTorch BiLSTM with batch_first=True):
      embedding.weight          → (vocab_size, embedding_dim)
      lstm.weight_ih_l0         → (4*hidden_dim, embedding_dim)
      lstm.weight_hh_l0         → (4*hidden_dim, hidden_dim)
      fc.weight                 → (1, hidden_dim * 2)    [bidirectional concat]

    Returns dict with keys:
      vocab_size, embedding_dim, hidden_dim, num_layers, fc_in

    Raises:
      KeyError   – required tensor not found in state dict
      ValueError – inferred dimensions are internally inconsistent
    """
    # vocab_size and embedding_dim from embedding layer
    emb_shape = sd["embedding.weight"].shape          # (vocab_size, embedding_dim)
    vocab_size = int(emb_shape[0])
    embedding_dim = int(emb_shape[1])

    # hidden_dim from the weight_hh tensor of layer 0 (columns = hidden_dim)
    hh_l0_shape = sd["lstm.weight_hh_l0"].shape       # (4*hidden_dim, hidden_dim)
    hidden_dim = int(hh_l0_shape[1])

    # num_layers: count distinct non-reverse LSTM weight_hh keys
    num_layers = sum(
        1 for k in sd
        if k.startswith("lstm.weight_hh_l") and "_reverse" not in k
    )

    # fc.weight input dim must equal hidden_dim * 2 (bidirectional concat)
    fc_in = int(sd["fc.weight"].shape[1])
    expected_fc_in = hidden_dim * 2
    if fc_in != expected_fc_in:
        raise ValueError(
            f"Checkpoint architecture inconsistency: "
            f"fc.weight input dim={fc_in}, but hidden_dim*2={expected_fc_in}. "
            f"This checkpoint may be corrupted or from an unsupported architecture variant."
        )

    return {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "fc_in": fc_in,
    }


class ModelManager:
    """
    Manages loaded model instances for inference.

    Model status labels:
      - "trained"  : A fully trained model checkpoint is available.
      - "demo"     : A demo/smoke-test checkpoint (clearly labelled, not for metrics).
      - "fallback" : No checkpoint available; rule-based analysis only.
    """

    def __init__(self):
        self.bilstm_model = None
        self.bilstm_vocab = None
        self.bilstm_arch: Optional[dict] = None   # inferred arch params for logging
        self.bilstm_status = "fallback"
        self.bilstm_checkpoint_path: Optional[str] = None

        self.graphcodebert_model = None
        self.graphcodebert_tokenizer = None
        self.graphcodebert_status = "not_loaded"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("ModelManager device: %s", self.device)

    # ------------------------------------------------------------------
    # Bi-LSTM loading
    # ------------------------------------------------------------------

    def load_bilstm(self, checkpoint_path: Optional[str] = None) -> None:
        """
        Attempt to load the Bi-LSTM model.
        Priority: best_model.pt > demo_bilstm_model.pt > fallback

        Args:
            checkpoint_path: Optional explicit path to a checkpoint file.
                             When provided, skips auto-discovery and loads
                             this file directly. Status is inferred from the
                             filename ('demo' if the name contains 'demo',
                             otherwise 'trained').

        Architecture parameters are read DIRECTLY from the checkpoint's
        state-dict tensor shapes. The config.yaml values are NOT used
        to reconstruct the model (they may differ from the actual saved weights).
        """
        from src.models.bilstm import BiLSTMVulnerabilityDetector

        # ── 1. Load vocabulary ────────────────────────────────────────────
        if not os.path.exists(VOCAB_PATH):
            logger.warning("Vocabulary not found at %s. Bi-LSTM unavailable.", VOCAB_PATH)
            self.bilstm_status = "fallback"
            return

        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            self.bilstm_vocab = json.load(f)

        # ── 2. Locate checkpoint ──────────────────────────────────────────
        status = "fallback"

        if checkpoint_path is not None:
            # Explicit path provided — use it directly
            if not os.path.exists(checkpoint_path):
                logger.warning("Explicit checkpoint not found at %s. Falling back.", checkpoint_path)
                self.bilstm_status = "fallback"
                return
            # Infer status from filename: 'demo' if 'demo' in the filename
            fname = os.path.basename(checkpoint_path).lower()
            status = "demo" if "demo" in fname else "trained"
            logger.info("Using explicit checkpoint: %s  (status=%s)", checkpoint_path, status)
        elif os.path.exists(BEST_CHECKPOINT_PATH):
            checkpoint_path = BEST_CHECKPOINT_PATH
            status = "trained"
            logger.info("Found best model checkpoint: %s", BEST_CHECKPOINT_PATH)
        elif os.path.exists(DEMO_CHECKPOINT_PATH):
            checkpoint_path = DEMO_CHECKPOINT_PATH
            status = "demo"
            logger.info(
                "Found demo checkpoint: %s  "
                "(Bi-LSTM Demo Model — 2-Epoch Demo Checkpoint)", DEMO_CHECKPOINT_PATH
            )
        else:
            logger.warning(
                "No Bi-LSTM checkpoint found. Operating in rule-based fallback mode.\n"
                "  Checked: %s, %s", BEST_CHECKPOINT_PATH, DEMO_CHECKPOINT_PATH
            )
            self.bilstm_status = "fallback"
            return


        # ── 3. Load raw checkpoint from disk ─────────────────────────────
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except Exception as exc:
            logger.error(
                "Cannot read checkpoint file '%s': %s", checkpoint_path, exc
            )
            self.bilstm_status = "fallback"
            return

        if "model_state_dict" not in checkpoint:
            logger.error(
                "Checkpoint '%s' has no 'model_state_dict' key. Cannot load.",
                checkpoint_path,
            )
            self.bilstm_status = "fallback"
            return

        sd = checkpoint["model_state_dict"]

        # ── 4. Infer exact architecture from state-dict tensor shapes ─────
        try:
            arch = _infer_bilstm_arch_from_state_dict(sd)
        except KeyError as exc:
            logger.error(
                "Missing expected tensor key in state dict for '%s': %s",
                checkpoint_path, exc,
            )
            self.bilstm_status = "fallback"
            return
        except ValueError as exc:
            logger.error(
                "Architecture validation failed for checkpoint '%s': %s",
                checkpoint_path, exc,
            )
            self.bilstm_status = "fallback"
            return

        pad_idx = int(checkpoint.get("pad_idx", 0))

        logger.info(
            "Architecture inferred from checkpoint state dict:\n"
            "  vocab_size   = %d\n"
            "  embedding_dim= %d\n"
            "  hidden_dim   = %d  (per direction)\n"
            "  num_layers   = %d\n"
            "  fc_in        = %d  (= hidden_dim * 2, bidirectional)\n"
            "  pad_idx      = %d",
            arch["vocab_size"], arch["embedding_dim"], arch["hidden_dim"],
            arch["num_layers"], arch["fc_in"], pad_idx,
        )

        # ── 5. Reconstruct model with exact checkpoint dimensions ─────────
        try:
            model = BiLSTMVulnerabilityDetector(
                vocab_size=arch["vocab_size"],
                embedding_dim=arch["embedding_dim"],
                hidden_dim=arch["hidden_dim"],
                num_layers=arch["num_layers"],
                dropout=0.0,        # dropout=0 at inference (no effect on weights)
                padding_idx=pad_idx,
            )
            model.load_state_dict(sd, strict=True)
            model.to(self.device)
            model.eval()

        except RuntimeError as exc:
            logger.error(
                "load_state_dict(strict=True) failed for '%s':\n%s",
                checkpoint_path, exc,
            )
            self.bilstm_status = "fallback"
            return
        except Exception as exc:
            logger.error(
                "Unexpected error reconstructing Bi-LSTM from '%s': %s",
                checkpoint_path, exc,
            )
            self.bilstm_status = "fallback"
            return

        # ── 6. Commit successfully loaded model ───────────────────────────
        self.bilstm_model = model
        self.bilstm_status = status
        self.bilstm_checkpoint_path = checkpoint_path
        self.bilstm_arch = arch

        label = (
            "Bi-LSTM Demo Model — 2-Epoch Demo Checkpoint"
            if status == "demo"
            else "Bi-LSTM Fully Trained Model"
        )
        logger.info(
            "[OK] %s loaded.\n"
            "  Status     : %s\n"
            "  Checkpoint : %s",
            label, status, checkpoint_path,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, code: str, model_name: str = "bilstm") -> Dict[str, Any]:
        """
        Run inference on a code snippet.

        Returns:
            Dict with keys: probability, mode
        """
        if model_name == "bilstm":
            return self._predict_bilstm(code)
        elif model_name == "graphcodebert":
            return self._predict_graphcodebert(code)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _predict_bilstm(self, code: str) -> Dict[str, Any]:
        """Run Bi-LSTM inference."""
        if self.bilstm_model is None:
            return {"probability": 0.0, "mode": "fallback"}

        from src.preprocessing.tokenizer import SourceCodeTokenizer

        tokenizer = SourceCodeTokenizer()
        tokens = tokenizer.tokenize(code)

        if not tokens:
            return {"probability": 0.15, "mode": self.bilstm_status}

        # Numericalize using loaded vocabulary
        max_len = 256
        unk_idx = 1
        token_ids = [self.bilstm_vocab.get(tok, unk_idx) for tok in tokens]

        # Add BOS/EOS tokens
        bos_idx, eos_idx, pad_idx = 2, 3, 0
        full_seq = [bos_idx] + token_ids + [eos_idx]
        unpadded_len = len(full_seq)

        if unpadded_len > max_len:
            full_seq = [bos_idx] + token_ids[: max_len - 2] + [eos_idx]
            unpadded_len = max_len
        elif unpadded_len < max_len:
            full_seq = full_seq + [pad_idx] * (max_len - unpadded_len)

        input_ids = torch.tensor([full_seq], dtype=torch.long, device=self.device)
        lengths = torch.tensor([min(unpadded_len, max_len)], dtype=torch.long)

        try:
            with torch.no_grad():
                logit = self.bilstm_model(input_ids, lengths)
            prob = torch.sigmoid(logit).item()
            return {"probability": float(prob), "mode": self.bilstm_status}
        except Exception as exc:
            logger.error("Bi-LSTM inference error: %s", exc)
            return {"probability": 0.0, "mode": "fallback"}

    def _predict_graphcodebert(self, code: str) -> Dict[str, Any]:
        """Run GraphCodeBERT inference (requires loaded model)."""
        if self.graphcodebert_model is None:
            logger.warning("GraphCodeBERT not loaded. Returning fallback.")
            return {"probability": 0.0, "mode": "fallback"}

        try:
            tokenizer = self.graphcodebert_tokenizer
            inputs = tokenizer(
                code,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logit = self.graphcodebert_model(
                    inputs["input_ids"], inputs["attention_mask"]
                )
            prob = torch.sigmoid(logit).item()
            return {"probability": float(prob), "mode": self.graphcodebert_status}
        except Exception as exc:
            logger.error("GraphCodeBERT inference error: %s", exc)
            return {"probability": 0.0, "mode": "fallback"}

    def get_status(self) -> Dict[str, Any]:
        """Return current model loading status."""
        return {
            "bilstm": {
                "status": self.bilstm_status,
                "checkpoint": self.bilstm_checkpoint_path,
                "loaded": self.bilstm_model is not None,
                "arch": self.bilstm_arch,
            },
            "graphcodebert": {
                "status": self.graphcodebert_status,
                "loaded": self.graphcodebert_model is not None,
            },
            "device": str(self.device),
        }

    def list_available_models(self):
        """Return list of available model identifiers."""
        bilstm_desc = (
            "Bi-LSTM Demo Model — 2-Epoch Demo Checkpoint. "
            "Trained on a 32-sample subset for pipeline verification only. "
            "NOT a fully trained production model."
            if self.bilstm_status == "demo"
            else (
                "Bidirectional LSTM trained on the Devign C/C++ dataset."
                if self.bilstm_status == "trained"
                else "No checkpoint loaded. Rule-based analysis only."
            )
        )
        models = [
            {
                "id": "bilstm",
                "name": "Bi-LSTM Vulnerability Detector",
                "status": self.bilstm_status,
                "description": bilstm_desc,
            },
            {
                "id": "graphcodebert",
                "name": "GraphCodeBERT (microsoft/graphcodebert-base)",
                "status": self.graphcodebert_status,
                "description": (
                    "Transformer-based model pre-trained on code. "
                    "Requires GPU fine-tuning for full performance. "
                    "NOT YET FINE-TUNED — GPU training required."
                ),
            },
        ]
        return models
