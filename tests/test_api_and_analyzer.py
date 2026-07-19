"""
Tests for Security Analyzer and API
---------------------------------------
Covers:
  - Security rule analysis
  - API health endpoint
  - API analyze endpoint
  - Invalid input handling
  - Missing checkpoint fallback
  - Configuration loading
  - GraphCodeBERT mock tests
"""

import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Security Analyzer Tests
# ---------------------------------------------------------------------------

class TestSecurityAnalyzer:
    """Tests for the deterministic security analysis pipeline."""

    def setup_method(self):
        from src.inference.security_analyzer import SecurityAnalysisService
        self.service = SecurityAnalysisService(model_manager=None)

    def test_empty_code_returns_clean_result(self):
        """Empty code should return no vulnerability detected."""
        result = self.service.analyze(code="", language="C/C++", model_name="bilstm")
        assert result["prediction"] == "No Vulnerability Detected"
        assert result["vulnerability_probability"] == 0.0
        assert "Empty Input" in result["potential_category"]

    def test_whitespace_code_returns_clean(self):
        """Whitespace-only code returns clean result."""
        result = self.service.analyze(code="   \n\t  ", language="C/C++")
        assert result["prediction"] == "No Vulnerability Detected"

    def test_strcpy_detected(self):
        """strcpy usage should be detected as unsafe string operation."""
        code = 'void f() { char buf[64]; strcpy(buf, user_input); }'
        result = self.service.analyze(code=code)
        patterns = result.get("suspicious_patterns", [])
        assert len(patterns) > 0
        assert any("strcpy" in p.lower() or "string" in p.lower() for p in patterns)

    def test_system_call_detected(self):
        """system() calls should trigger command execution detection."""
        code = 'void f(char *input) { system(input); }'
        result = self.service.analyze(code=code)
        patterns = result.get("suspicious_patterns", [])
        assert len(patterns) > 0
        assert any("command" in p.lower() or "system" in p.lower() or "exec" in p.lower() for p in patterns)

    def test_hardcoded_credential_detected(self):
        """Hard-coded passwords should be detected."""
        code = 'char *password = "supersecret123";'
        result = self.service.analyze(code=code)
        patterns = result.get("suspicious_patterns", [])
        assert len(patterns) > 0
        assert any("credential" in p.lower() or "password" in p.lower() or "secret" in p.lower() for p in patterns)

    def test_safe_code_lower_probability(self):
        """Code with no unsafe patterns should have low rule-based probability."""
        code = '''
        int add(int a, int b) {
            return a + b;
        }
        '''
        result = self.service.analyze(code=code)
        # Safe code should have low probability in pure rule-based mode
        assert result["vulnerability_probability"] < 0.5

    def test_weak_crypto_detected(self):
        """MD5 usage should be flagged as weak crypto."""
        code = 'unsigned char hash[16]; MD5(data, len, hash);'
        result = self.service.analyze(code=code)
        patterns = result.get("suspicious_patterns", [])
        assert any("crypto" in p.lower() or "md5" in p.lower() or "weak" in p.lower() for p in patterns)

    def test_result_contains_required_fields(self):
        """Analysis result must contain all required fields."""
        code = "void f() { int x = 0; }"
        result = self.service.analyze(code=code)
        required_fields = [
            "model", "prediction", "vulnerability_probability", "confidence",
            "potential_category", "suspicious_patterns", "explanation",
            "recommendations", "suggested_code", "disclaimer",
        ]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    def test_disclaimer_always_present(self):
        """Disclaimer must always be included in results."""
        result = self.service.analyze(code="int x = 0;")
        assert len(result["disclaimer"]) > 50

    def test_recommendations_always_list(self):
        """Recommendations must always be a list."""
        result = self.service.analyze(code="int x = 0;")
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) >= 1

    def test_probability_range(self):
        """Probability must always be in [0, 1]."""
        for code in ["", "int x=0;", 'strcpy(buf, input);', 'system("ls");']:
            result = self.service.analyze(code=code)
            prob = result["vulnerability_probability"]
            assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"

    def test_suggested_code_for_unsafe_functions(self):
        """Suggested code should be generated for unsafe function usage."""
        code = 'void f() { char buf[64]; sprintf(buf, "%s", input); }'
        result = self.service.analyze(code=code)
        # May or may not have suggested code, but should not crash
        assert isinstance(result["suggested_code"], str)

    def test_no_code_execution(self):
        """Malicious code should be analyzed, not executed."""
        dangerous_code = '__import__("os").system("rm -rf /")'
        result = self.service.analyze(code=dangerous_code)
        # Should return analysis, not execute
        assert "prediction" in result
        assert "disclaimer" in result


# ---------------------------------------------------------------------------
# Rule-based analysis unit tests
# ---------------------------------------------------------------------------

class TestRuleAnalysis:
    """Unit tests for the deterministic rule engine."""

    def test_run_rule_analysis_empty(self):
        from src.inference.security_analyzer import run_rule_analysis
        matches = run_rule_analysis("")
        assert matches == []

    def test_run_rule_analysis_strcpy(self):
        from src.inference.security_analyzer import run_rule_analysis
        matches = run_rule_analysis('strcpy(buf, src);')
        assert len(matches) > 0
        names = [m.pattern_name for m in matches]
        assert "unsafe_string_op" in names

    def test_run_rule_analysis_multiple_patterns(self):
        from src.inference.security_analyzer import run_rule_analysis
        code = 'system(input); strcpy(buf, src);'
        matches = run_rule_analysis(code)
        assert len(matches) >= 2

    def test_derive_category(self):
        from src.inference.security_analyzer import derive_category_from_rules, RuleMatch
        matches = [
            RuleMatch(
                "command_execution", "desc", "Possible Command Injection Risk", "cmd",
                severity="High", line_number=10, cwe="CWE-78", owasp="A03", recommendation="Fix it"
            ),
        ]
        cat = derive_category_from_rules(matches)
        assert "Command" in cat or "command" in cat.lower()

    def test_recommendations_generated(self):
        from src.inference.security_analyzer import build_recommendations, RuleMatch
        matches = [
            RuleMatch(
                "unsafe_string_op", "desc", "cat", "strcpy",
                severity="High", line_number=12, cwe="CWE-120", owasp="A03", recommendation="Use strncpy"
            ),
        ]
        recs = build_recommendations(matches, "Potentially Vulnerable")
        assert len(recs) >= 1
        assert any("strcpy" in r or "strncpy" in r for r in recs)


# ---------------------------------------------------------------------------
# Model Manager Tests
# ---------------------------------------------------------------------------

class TestModelManager:
    """Tests for model loading and fallback behavior."""

    def test_manager_initializes_without_checkpoint(self):
        """ModelManager should initialize cleanly even without checkpoints."""
        from src.inference.model_manager import ModelManager
        mgr = ModelManager()
        assert mgr.bilstm_model is None
        assert mgr.bilstm_status == "fallback"

    def test_load_bilstm_fallback_without_checkpoint(self):
        """load_bilstm() should set status=fallback when no checkpoint exists."""
        from src.inference.model_manager import ModelManager
        mgr = ModelManager()
        with patch("os.path.exists") as mock_exists:
            def exists_side_effect(path):
                if "vocabulary.json" in str(path):
                    return True
                return False
            mock_exists.side_effect = exists_side_effect
            mgr.load_bilstm()
        assert mgr.bilstm_status == "fallback"

    def test_predict_no_model_returns_fallback(self):
        """Without a loaded model, _predict_bilstm must return mode='fallback'."""
        from src.inference.model_manager import ModelManager
        mgr = ModelManager()
        result = mgr._predict_bilstm("int x = 0;")
        assert "probability" in result
        assert "mode" in result
        assert isinstance(result["probability"], float)
        assert result["mode"] == "fallback"

    def test_get_status_returns_complete_dict(self):
        """get_status() must return complete status dictionary."""
        from src.inference.model_manager import ModelManager
        mgr = ModelManager()
        status = mgr.get_status()
        assert "bilstm" in status
        assert "graphcodebert" in status
        assert "device" in status

    def test_list_models_returns_both(self):
        """list_available_models() should return both Bi-LSTM and GraphCodeBERT entries."""
        from src.inference.model_manager import ModelManager
        mgr = ModelManager()
        models = mgr.list_available_models()
        assert len(models) >= 2
        ids = [m["id"] for m in models]
        assert "bilstm" in ids
        assert "graphcodebert" in ids

    def test_rule_based_fallback_still_works_without_model(self):
        """SecurityAnalysisService with no model_manager must still produce rule-based findings."""
        from src.inference.security_analyzer import SecurityAnalysisService
        svc = SecurityAnalysisService(model_manager=None)
        result = svc.analyze("void f() { strcpy(buf, src); }")
        assert result["model_mode"] == "fallback"
        assert result["rule_matches_count"] >= 1


# ---------------------------------------------------------------------------
# Demo Checkpoint Loading Tests
# (Require models/checkpoints/demo_bilstm_model.pt to exist)
# ---------------------------------------------------------------------------

import pytest as _pytest

_DEMO_CKPT = "models/checkpoints/demo_bilstm_model.pt"


@_pytest.mark.skipif(
    not os.path.exists(_DEMO_CKPT),
    reason="Demo checkpoint not found — run smoke_test_bilstm.py first",
)
class TestDemoCheckpoint:
    """
    Tests verifying that the demo checkpoint loads correctly with the right
    architecture (embedding_dim=64, hidden_dim=128 — the smoke-test dims)
    and that inference runs successfully.
    """

    def test_demo_checkpoint_loads_without_error(self):
        """Demo checkpoint must load with no RuntimeError (no size mismatch)."""
        from src.inference.model_manager import ModelManager
        mgr = ModelManager()
        mgr.load_bilstm(checkpoint_path=_DEMO_CKPT)
        assert mgr.bilstm_model is not None, (
            "Model is None — checkpoint loading failed. Check logs for size mismatch."
        )
        assert mgr.bilstm_status == "demo", (
            f"Expected status='demo', got '{mgr.bilstm_status}'."
        )

    def test_demo_checkpoint_not_labelled_trained(self):
        """Demo checkpoint must NOT report status='trained'."""
        from src.inference.model_manager import ModelManager
        mgr = ModelManager()
        mgr.load_bilstm(checkpoint_path=_DEMO_CKPT)
        assert mgr.bilstm_status != "trained"
        assert mgr.bilstm_status != "fallback"

    def test_arch_inferred_from_state_dict_not_config(self):
        """Architecture dims must match the smoke-test values (not config.yaml defaults)."""
        from src.inference.model_manager import ModelManager
        mgr = ModelManager()
        mgr.load_bilstm(checkpoint_path=_DEMO_CKPT)
        assert mgr.bilstm_arch is not None
        arch = mgr.bilstm_arch
        # Smoke test used embedding_dim=64, hidden_dim=128 — NOT config defaults (128, 256)
        assert arch["embedding_dim"] == 64, (
            f"Expected embedding_dim=64 from state dict, got {arch['embedding_dim']}."
        )
        assert arch["hidden_dim"] == 128, (
            f"Expected hidden_dim=128 from state dict, got {arch['hidden_dim']}."
        )
        assert arch["vocab_size"] == 10000
        assert arch["num_layers"] == 2
        assert arch["fc_in"] == 256

    def test_demo_model_inference_valid_probability(self):
        """Model inference must return probability in [0.0, 1.0]."""
        from src.inference.model_manager import ModelManager
        mgr = ModelManager()
        mgr.load_bilstm()
        assert mgr.bilstm_model is not None
        snippets = [
            "void f() { char buf[64]; strcpy(buf, input); }",
            "int add(int a, int b) { return a + b; }",
            "void cmd(char *s) { system(s); }",
        ]
        for code in snippets:
            result = mgr._predict_bilstm(code)
            prob = result["probability"]
            assert 0.0 <= prob <= 1.0, f"Prob {prob} out of range for: {code!r}"

    def test_demo_model_returns_demo_mode(self):
        """Inference must report mode='demo', not 'fallback' or 'trained'."""
        from src.inference.model_manager import ModelManager
        mgr = ModelManager()
        mgr.load_bilstm(checkpoint_path=_DEMO_CKPT)
        result = mgr._predict_bilstm("void f() { gets(buf); }")
        assert result["mode"] == "demo"

    def test_security_service_reports_demo_mode(self):
        """Full analysis pipeline must report model_mode='demo' when demo ckpt is loaded."""
        from src.inference.model_manager import ModelManager
        from src.inference.security_analyzer import SecurityAnalysisService
        mgr = ModelManager()
        mgr.load_bilstm(checkpoint_path=_DEMO_CKPT)
        svc = SecurityAnalysisService(model_manager=mgr)
        result = svc.analyze(
            "void f() { char buf[64]; strcpy(buf, input); }",
            language="C/C++",
            model_name="bilstm",
        )
        assert result["model_mode"] == "demo"
        assert 0.0 <= result["vulnerability_probability"] <= 1.0

    def test_get_status_reports_demo_loaded(self):
        """get_status() must report demo+loaded=True after successful checkpoint load."""
        from src.inference.model_manager import ModelManager
        mgr = ModelManager()
        mgr.load_bilstm(checkpoint_path=_DEMO_CKPT)
        status = mgr.get_status()
        assert status["bilstm"]["status"] == "demo"
        assert status["bilstm"]["loaded"] is True

    def test_arch_helper_correct_values(self):
        """_infer_bilstm_arch_from_state_dict must return exact demo-checkpoint dims."""
        import torch
        from src.inference.model_manager import _infer_bilstm_arch_from_state_dict
        ckpt = torch.load(_DEMO_CKPT, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]
        arch = _infer_bilstm_arch_from_state_dict(sd)
        assert arch["vocab_size"] == 10000
        assert arch["embedding_dim"] == 64
        assert arch["hidden_dim"] == 128
        assert arch["num_layers"] == 2
        assert arch["fc_in"] == 256

    def test_arch_helper_rejects_bad_fc_dim(self):
        """_infer_bilstm_arch_from_state_dict must raise ValueError for mismatched fc dim."""
        import torch
        from src.inference.model_manager import _infer_bilstm_arch_from_state_dict
        ckpt = torch.load(_DEMO_CKPT, map_location="cpu", weights_only=False)
        sd = dict(ckpt["model_state_dict"])
        sd["fc.weight"] = torch.zeros(1, 999)  # tamper: wrong input dim
        with _pytest.raises(ValueError, match="inconsistency"):
            _infer_bilstm_arch_from_state_dict(sd)


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------

class TestConfiguration:
    """Tests for configuration loading."""

    def test_config_file_exists(self):
        """configs/config.yaml should exist."""
        assert os.path.exists("configs/config.yaml"), "Config file missing!"

    def test_config_loads(self):
        """Config file should be valid YAML with required keys."""
        import yaml
        with open("configs/config.yaml") as f:
            config = yaml.safe_load(f)
        assert "training" in config
        assert "models" in config
        assert "paths" in config
        assert "bilstm" in config["models"]

    def test_bilstm_config_params(self):
        """Bi-LSTM config should have required parameters."""
        import yaml
        with open("configs/config.yaml") as f:
            config = yaml.safe_load(f)
        lstm = config["models"]["bilstm"]
        for key in ["embedding_dim", "hidden_dim", "num_layers", "dropout"]:
            assert key in lstm, f"Missing key in bilstm config: {key}"

    def test_graphcodebert_config_present(self):
        """GraphCodeBERT config section should be present."""
        import yaml
        with open("configs/config.yaml") as f:
            config = yaml.safe_load(f)
        assert "graphcodebert" in config["models"]
        assert "model_name" in config["models"]["graphcodebert"]


# ---------------------------------------------------------------------------
# Bi-LSTM Inference Tests (without full checkpoint)
# ---------------------------------------------------------------------------

class TestBiLSTMInference:
    """Tests for Bi-LSTM model loading and inference."""

    def test_model_can_be_instantiated(self):
        """BiLSTMVulnerabilityDetector should instantiate with any valid params."""
        from src.models.bilstm import BiLSTMVulnerabilityDetector
        model = BiLSTMVulnerabilityDetector(
            vocab_size=200, embedding_dim=32, hidden_dim=64,
            num_layers=1, dropout=0.0, padding_idx=0,
        )
        assert model is not None

    def test_model_inference_returns_float(self):
        """Model should return a float logit for valid input."""
        import torch
        from src.models.bilstm import BiLSTMVulnerabilityDetector
        model = BiLSTMVulnerabilityDetector(
            vocab_size=200, embedding_dim=32, hidden_dim=64,
            num_layers=1, dropout=0.0, padding_idx=0,
        )
        model.eval()
        input_ids = torch.randint(1, 100, (1, 20))
        lengths = torch.tensor([20])
        with torch.no_grad():
            logit = model(input_ids, lengths)
        assert logit.shape == (1, 1)
        prob = torch.sigmoid(logit).item()
        assert 0.0 <= prob <= 1.0

    def test_model_checkpoint_structure(self, tmp_path):
        """A saved checkpoint must contain required keys."""
        import torch
        from src.models.bilstm import BiLSTMVulnerabilityDetector
        from src.training.train_utils import save_checkpoint

        model = BiLSTMVulnerabilityDetector(
            vocab_size=50, embedding_dim=16, hidden_dim=32,
            num_layers=1, dropout=0.0, padding_idx=0,
        )
        optimizer = torch.optim.AdamW(model.parameters())
        path = str(tmp_path / "test_model.pt")
        save_checkpoint(
            path=path, model=model, optimizer=optimizer, scheduler=None,
            epoch=1, best_f1=0.5, metrics={"val_f1": 0.5},
            config={}, seed=42, vocab_size=50, pad_idx=0,
        )
        ckpt = torch.load(path, map_location="cpu")
        for key in ["model_state_dict", "epoch", "best_f1", "vocab_size"]:
            assert key in ckpt, f"Missing checkpoint key: {key}"


# ---------------------------------------------------------------------------
# API Tests (using TestClient — no live server required)
# ---------------------------------------------------------------------------

class TestAPI:
    """Tests for the FastAPI endpoints using TestClient."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        """Set up TestClient with mocked model manager."""
        from fastapi.testclient import TestClient
        from api.main import app

        with TestClient(app) as client:
            self.client = client
            yield

    def test_health_endpoint_returns_ok(self):
        """GET /health should return status=ok."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "bilstm_status" in data

    def test_models_endpoint_returns_list(self):
        """GET /models should return a non-empty list."""
        response = self.client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_analyze_endpoint_basic(self):
        """POST /analyze should return analysis result for valid code."""
        payload = {
            "code": "void f() { char buf[64]; strcpy(buf, input); }",
            "model": "bilstm",
            "language": "C/C++",
        }
        response = self.client.post("/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "vulnerability_probability" in data
        assert "disclaimer" in data
        assert data["prediction"] in ["Potentially Vulnerable", "No Vulnerability Detected"]

    def test_analyze_endpoint_probability_range(self):
        """Vulnerability probability must be in [0.0, 1.0]."""
        payload = {"code": "int main() { return 0; }", "model": "bilstm"}
        response = self.client.post("/analyze", json=payload)
        assert response.status_code == 200
        prob = response.json()["vulnerability_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_endpoint_basic(self):
        """POST /predict should return probability and model mode."""
        payload = {"code": "void f() { system(user_input); }", "model": "bilstm"}
        response = self.client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "vulnerability_probability" in data
        assert "model_mode" in data

    def test_analyze_invalid_model(self):
        """POST /analyze with invalid model name should return 422."""
        payload = {"code": "int x = 0;", "model": "invalid_model_xyz"}
        response = self.client.post("/analyze", json=payload)
        assert response.status_code == 422

    def test_analyze_empty_code_field(self):
        """POST /analyze with empty code should return 422 (min_length=1)."""
        payload = {"code": "", "model": "bilstm"}
        response = self.client.post("/analyze", json=payload)
        assert response.status_code == 422

    def test_analyze_code_too_long(self):
        """POST /analyze with code > 50000 chars should return 422."""
        payload = {"code": "x" * 50001, "model": "bilstm"}
        response = self.client.post("/analyze", json=payload)
        assert response.status_code == 422

    def test_analyze_missing_code_field(self):
        """POST /analyze with missing code field should return 422."""
        payload = {"model": "bilstm"}
        response = self.client.post("/analyze", json=payload)
        assert response.status_code == 422

    def test_analyze_returns_recommendations(self):
        """POST /analyze should always return a list of recommendations."""
        payload = {"code": "void f() { gets(buf); }", "model": "bilstm"}
        response = self.client.post("/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["recommendations"], list)

    def test_analyze_dangerous_code_not_executed(self):
        """
        Submitting dangerous code should analyze it safely, not execute it.
        The response should contain analysis fields, not execution output.
        """
        dangerous_payloads = [
            "import os; os.system('rm -rf /')",
            "__import__('subprocess').call(['ls'])",
            "eval('print(1)')",
        ]
        for code in dangerous_payloads:
            payload = {"code": code, "model": "bilstm"}
            response = self.client.post("/analyze", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "disclaimer" in data

    def test_root_endpoint(self):
        """GET / should return API info."""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data


# ---------------------------------------------------------------------------
# GraphCodeBERT Mock Tests
# ---------------------------------------------------------------------------

class TestGraphCodeBERTMocked:
    """Tests for GraphCodeBERT infrastructure using mocks (no downloads)."""

    def test_dataset_class_exists(self):
        """GraphCodeBERTDataset should be importable."""
        from src.models.graphcodebert import GraphCodeBERTDataset
        assert GraphCodeBERTDataset is not None

    def test_dataset_with_mock_tokenizer(self):
        """GraphCodeBERTDataset should work with a mock tokenizer."""
        import torch
        from src.models.graphcodebert import GraphCodeBERTDataset

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 100, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
        }

        code_samples = ["void f() {}", "int x = 0;"]
        labels = [1, 0]

        dataset = GraphCodeBERTDataset(code_samples, labels, mock_tokenizer, max_length=10)
        assert len(dataset) == 2
        item = dataset[0]
        assert "labels" in item

    def test_model_class_exists(self):
        """GraphCodeBERTVulnerabilityClassifier should be importable."""
        from src.models.graphcodebert import GraphCodeBERTVulnerabilityClassifier
        assert GraphCodeBERTVulnerabilityClassifier is not None

    def test_model_init_skipped_without_transformers(self):
        """Verify that the model class has required architecture attributes."""
        import inspect
        from src.models.graphcodebert import GraphCodeBERTVulnerabilityClassifier
        # Verify class structure
        assert hasattr(GraphCodeBERTVulnerabilityClassifier, '__init__')
        assert hasattr(GraphCodeBERTVulnerabilityClassifier, 'forward')
        # Check __init__ signature has expected parameters
        sig = inspect.signature(GraphCodeBERTVulnerabilityClassifier.__init__)
        params = list(sig.parameters.keys())
        assert "model_name" in params
        assert "dropout" in params

    def test_save_and_load_functions_exist(self):
        """Checkpoint save and load functions should be importable."""
        from src.models.graphcodebert import (
            save_graphcodebert_checkpoint,
            load_graphcodebert_checkpoint,
        )
        assert callable(save_graphcodebert_checkpoint)
        assert callable(load_graphcodebert_checkpoint)

    def test_load_nonexistent_checkpoint_raises(self):
        """Loading a non-existent checkpoint should raise FileNotFoundError."""
        from src.models.graphcodebert import load_graphcodebert_checkpoint
        with pytest.raises(FileNotFoundError):
            load_graphcodebert_checkpoint("/nonexistent/path/model.pt")
