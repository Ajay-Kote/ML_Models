"""
api/app.py

FastAPI orchestrator for Module 5 (Email Detection) only, per design doc Section 9's
`module5_email/` folder layout (this module does not expose the other four modules or
the top-level Adaptive Risk Fusion Engine — those are out of scope for this deliverable).

Run:
    uvicorn api.app:app --reload --port 8005
    # then POST raw email text to http://localhost:8005/predict
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException

from api.schemas import EmailPredictRequest, EmailPredictResponse, HealthResponse
from predict import predict, METADATA_MODEL_PATH, FUSION_MODEL_PATH
from text_branch.distilbert_branch import get_text_branch

app = FastAPI(
    title="Adaptive Risk Fusion — Email Detection Module",
    description="Module 5 of the Adaptive Risk Fusion system (design doc Section 5.5). "
                "DistilBERT text branch + LightGBM metadata branch, fused via a small MLP, "
                "with SHAP/LIME explainability.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health():
    text_branch = get_text_branch(prefer_distilbert=True)
    return HealthResponse(
        status="ok",
        metadata_model_loaded=METADATA_MODEL_PATH.exists(),
        fusion_model_loaded=FUSION_MODEL_PATH.exists(),
        text_branch=text_branch.__class__.__name__,
    )


@app.post("/predict", response_model=EmailPredictResponse)
def predict_email(request: EmailPredictRequest):
    if not request.raw_email or not request.raw_email.strip():
        raise HTTPException(status_code=400, detail="raw_email must be non-empty.")
    try:
        result = predict(request.raw_email)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return result
