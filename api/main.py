"""
FastAPI Backend — AI Security Copilot
---------------------------------------
Provides REST API endpoints for vulnerability analysis.

Endpoints:
  GET  /health          — Health check
  GET  /models          — List available models
  POST /predict         — Quick prediction (probability only)
  POST /analyze         — Full vulnerability analysis

Security invariants:
  - Does NOT execute submitted code
  - Does NOT compile submitted code
  - Does NOT use eval() or exec()
  - Does NOT expose secrets or internal stack traces
  - Input size is capped at 50,000 characters

Usage:
  uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("api.main")

# Application globals — models loaded once at startup
model_manager = None
analysis_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models and services at startup."""
    global model_manager, analysis_service

    logger.info("=" * 60)
    logger.info("AI Security Copilot API — Starting up")
    logger.info("=" * 60)

    try:
        from src.inference.model_manager import ModelManager
        from src.inference.security_analyzer import SecurityAnalysisService

        model_manager = ModelManager()
        model_manager.load_bilstm()

        analysis_service = SecurityAnalysisService(model_manager=model_manager)

        status = model_manager.get_status()
        logger.info("Bi-LSTM status: %s", status["bilstm"]["status"])
        logger.info("Device: %s", status["device"])

        if status["bilstm"]["status"] == "fallback":
            logger.warning(
                "No trained checkpoint found. Operating in rule-based fallback mode. "
                "Run training to generate a checkpoint."
            )

    except Exception as exc:
        logger.error("Startup error: %s", exc, exc_info=True)
        # Continue — API remains usable in fallback mode
        from src.inference.security_analyzer import SecurityAnalysisService
        analysis_service = SecurityAnalysisService(model_manager=None)

    logger.info("API ready.")
    yield

    # Shutdown
    logger.info("AI Security Copilot API — Shutting down")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Security Copilot",
    description=(
        "Source Code Vulnerability Detection and Secure Fix Recommendation API. "
        "Uses a Bi-LSTM model trained on the Devign dataset for binary vulnerability "
        "classification, combined with deterministic security-pattern analysis."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

MAX_CODE_LENGTH = 50_000  # characters


class PredictRequest(BaseModel):
    code: str = Field(..., description="Source code to analyze", min_length=1)
    model: str = Field(default="bilstm", description="Model to use: 'bilstm' or 'graphcodebert'")
    language: str = Field(default="C/C++", description="Programming language (for display)")

    @field_validator("code")
    @classmethod
    def validate_code_length(cls, v):
        if len(v) > MAX_CODE_LENGTH:
            raise ValueError(
                f"Code exceeds maximum allowed length of {MAX_CODE_LENGTH} characters."
            )
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        allowed = {"bilstm", "graphcodebert"}
        if v.lower() not in allowed:
            raise ValueError(f"Model must be one of: {allowed}")
        return v.lower()


class AnalyzeRequest(BaseModel):
    code: str = Field(..., description="Source code to analyze", min_length=1)
    model: str = Field(default="bilstm", description="Model to use: 'bilstm' or 'graphcodebert'")
    language: str = Field(default="C/C++", description="Programming language (for display)")

    @field_validator("code")
    @classmethod
    def validate_code_length(cls, v):
        if len(v) > MAX_CODE_LENGTH:
            raise ValueError(
                f"Code exceeds maximum allowed length of {MAX_CODE_LENGTH} characters."
            )
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        allowed = {"bilstm", "graphcodebert"}
        if v.lower() not in allowed:
            raise ValueError(f"Model must be one of: {allowed}")
        return v.lower()


class PredictResponse(BaseModel):
    model: str
    prediction: str
    vulnerability_probability: float
    confidence: float
    model_mode: str


class AnalyzeResponse(BaseModel):
    model: str
    language: str
    prediction: str
    vulnerability_probability: float
    confidence: float
    model_mode: str
    potential_category: str
    suspicious_patterns: List[str]
    rule_matches_count: int
    explanation: str
    recommendations: List[str]
    suggested_code: str
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    version: str
    bilstm_status: str
    device: str
    timestamp: float


class ModelInfo(BaseModel):
    id: str
    name: str
    status: str
    description: str


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all handler — never exposes internal stack traces."""
    logger.error("Unhandled exception on %s: %s", request.url.path, str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please check your input and try again.",
        },
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns API health status and model loading information.",
    tags=["System"],
)
async def health():
    bilstm_status = "fallback"
    device_str = "cpu"
    if model_manager is not None:
        status = model_manager.get_status()
        bilstm_status = status["bilstm"]["status"]
        device_str = status["device"]

    return HealthResponse(
        status="ok",
        version="1.0.0",
        bilstm_status=bilstm_status,
        device=device_str,
        timestamp=time.time(),
    )


@app.get(
    "/models",
    response_model=List[ModelInfo],
    summary="List available models",
    description="Returns information about all supported models and their current loading status.",
    tags=["Models"],
)
async def list_models():
    if model_manager is None:
        return [
            ModelInfo(
                id="bilstm",
                name="Bi-LSTM Vulnerability Detector",
                status="fallback",
                description="Model not loaded. Rule-based analysis only.",
            )
        ]
    models = model_manager.list_available_models()
    return [ModelInfo(**m) for m in models]


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Quick vulnerability prediction",
    description=(
        "Runs the selected model on the submitted code and returns a binary prediction "
        "with probability. Does not include detailed analysis. "
        "NOTE: This endpoint does NOT execute or compile the submitted code."
    ),
    tags=["Analysis"],
)
async def predict(request: PredictRequest):
    if analysis_service is None:
        raise HTTPException(status_code=503, detail="Analysis service not initialized.")

    try:
        result = analysis_service.analyze(
            code=request.code,
            language=request.language,
            model_name=request.model,
        )
        return PredictResponse(
            model=result["model"],
            prediction=result["prediction"],
            vulnerability_probability=result["vulnerability_probability"],
            confidence=result["confidence"],
            model_mode=result["model_mode"],
        )
    except Exception as exc:
        logger.error("Predict error: %s", exc, exc_info=True)
        raise HTTPException(status_code=422, detail="Analysis failed. Check code input.")


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Full vulnerability analysis",
    description=(
        "Performs comprehensive vulnerability analysis including:\n"
        "- ML model prediction (Bi-LSTM or GraphCodeBERT)\n"
        "- Rule-based security pattern detection\n"
        "- Explanation and recommendations\n"
        "- Suggested secure code improvements\n\n"
        "NOTE: This endpoint does NOT execute or compile the submitted code. "
        "Output clearly separates model predictions from rule-based findings."
    ),
    tags=["Analysis"],
)
async def analyze(request: AnalyzeRequest):
    if analysis_service is None:
        raise HTTPException(status_code=503, detail="Analysis service not initialized.")

    try:
        result = analysis_service.analyze(
            code=request.code,
            language=request.language,
            model_name=request.model,
        )
        return AnalyzeResponse(**result)
    except Exception as exc:
        logger.error("Analyze error: %s", exc, exc_info=True)
        raise HTTPException(status_code=422, detail="Analysis failed. Check code input.")


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "AI Security Copilot API",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0",
    }
