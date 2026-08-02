"""
api/schemas.py

Request/response models for the Email Detection Module's FastAPI orchestrator
(design doc, Section 8: "Backend / API -> FastAPI or Flask").
"""

from __future__ import annotations

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class EmailPredictRequest(BaseModel):
    raw_email: str = Field(..., description="Full raw RFC822 email text, e.g. the contents of an .eml file.")


class MetadataFeatureContribution(BaseModel):
    feature: str
    value: float
    shap_contribution: float


class BranchConfidence(BaseModel):
    text: float
    metadata: float


class EmailPredictResponse(BaseModel):
    phishing_probability: float
    branch_confidence: BranchConfidence
    top_metadata_features: List[MetadataFeatureContribution]
    top_text_tokens: List[List]  # [[token, weight], ...]
    explanation: str
    raw_metadata_features: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    metadata_model_loaded: bool
    fusion_model_loaded: bool
    text_branch: str
