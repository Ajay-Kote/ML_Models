"""
Module 4 - Payment Image Detection
Standalone FastAPI service so this module can be trained/tested/deployed
independently of the other four modules (Design Principle 4.1: Modularity),
before being wired into the top-level fusion backend (Section 9: backend/app.py).
"""

import os
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    fraud_probability: float
    structured_fields: dict
    ocr_quality: dict
    top_contributing_features: list
    tampering_heatmap_path: str | None

from models.predict import PaymentImageDetector

app = FastAPI(title="Payment Image (Screenshot) Detection Module")
detector = PaymentImageDetector(artifacts_dir="models/artifacts")


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # heatmap file is intentionally kept on disk -- its path is returned
        # in the response so the caller can fetch/download it afterward.
        heatmap_path = tmp_path.replace(".jpg", "_heatmap.jpg")
        result = detector.predict(tmp_path, heatmap_out_path=heatmap_path)
        return result
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.get("/health")
async def health():
    return {"status": "ok"}
