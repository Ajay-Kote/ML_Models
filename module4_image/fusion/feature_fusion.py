"""
Module 4 - Payment Image Detection
Feature Fusion Layer: merges
  (a) OCR structured fields + quality signals   (from ocr/paddleocr_pipeline.py)
  (b) EfficientNet-B0 visual embedding (1280-d) (from vision/efficientnet_extractor.py)
into a single flat feature vector consumed by models/train.py (LightGBM / MLP).

Design choice: the 1280-d visual embedding is reduced via PCA before
concatenation with the ~10 structured/quality features, so that LightGBM
(which handles a handful of hundred features comfortably but degrades
with heavy sparse/high-dim input) isn't dominated by the visual branch.
"""

from dataclasses import asdict
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ocr.paddleocr_pipeline import OCRResult
from vision.efficientnet_extractor import VisualEmbeddingService
from vision.ela_features import compute_ela_features, ela_features_to_row

KNOWN_APP_VOCAB = [
    "google pay", "gpay", "phonepe", "paytm", "amazon pay",
    "bhim", "whatsapp pay", "cred", "mobikwik", None,
]
STATUS_VOCAB = ["success", "successful", "completed", "failed", "pending", "declined", None]


class VisualPCAReducer:
    """Fit once on training embeddings, reused at inference time."""

    def __init__(self, n_components: int = 32):
        self.n_components = n_components
        self.pca: Optional[PCA] = None

    def fit(self, embeddings: np.ndarray):
        self.pca = PCA(n_components=self.n_components, random_state=42)
        self.pca.fit(embeddings)
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("VisualPCAReducer must be fit() before transform().")
        return self.pca.transform(embeddings)

    def save(self, path: str):
        joblib.dump(self.pca, path)

    def load(self, path: str):
        self.pca = joblib.load(path)
        return self


def _one_hot(value: Optional[str], vocab: list, prefix: str) -> dict:
    value = value.lower() if isinstance(value, str) else None
    return {f"{prefix}_{v}": int(value == v) for v in vocab if v is not None} | {
        f"{prefix}_missing": int(value is None)
    }


def structured_features_to_row(ocr_result: OCRResult) -> dict:
    """Flatten OCRStructuredFields + OCRQualitySignals into a numeric dict."""
    s = asdict(ocr_result.structured)
    q = asdict(ocr_result.quality)

    row = {
        "amount": s["amount"] if s["amount"] is not None else -1.0,
        "amount_present": int(s["amount"] is not None),
        "upi_id_present": int(s["upi_id"] is not None),
        "transaction_id_present": int(s["transaction_id"] is not None),
        "timestamp_present": int(s["timestamp"] is not None),
        **q,  # avg_confidence, min_confidence, num_text_boxes, font_size_std,
              # line_spacing_std, low_confidence_ratio
    }
    row.update(_one_hot(s["bank_or_app_name"], KNOWN_APP_VOCAB, "app"))
    row.update(_one_hot(s["status_text"], STATUS_VOCAB, "status"))
    return row


def build_feature_vector(
    ocr_result: OCRResult,
    visual_embedding: np.ndarray,
    pca_reducer: Optional[VisualPCAReducer] = None,
    image_path: Optional[str] = None,
) -> pd.Series:
    """
    Returns a single pandas Series (one row) ready to append to a training
    DataFrame or feed directly into the trained fusion classifier.

    image_path, if given, adds Error Level Analysis (ELA) forensic features
    (see vision/ela_features.py) -- pass it whenever the source image path
    is available (it is at every current call site: models/train.py,
    models/predict.py, FusionFeatureBuilder.build below). If omitted, ELA
    columns are simply absent from this row; downstream code already
    reindexes + fillna(-1) against the saved feature_columns list, so this
    stays backward compatible with old cached feature rows.
    """
    row = structured_features_to_row(ocr_result)

    if image_path is not None:
        row.update(ela_features_to_row(compute_ela_features(image_path)))

    if pca_reducer is not None and pca_reducer.pca is not None:
        reduced = pca_reducer.transform(visual_embedding.reshape(1, -1))[0]
    else:
        reduced = visual_embedding  # unreduced fallback (e.g. before PCA is fit)

    for i, val in enumerate(reduced):
        row[f"visual_pc_{i}"] = float(val)

    return pd.Series(row)


class FusionFeatureBuilder:
    """End-to-end convenience: image path -> fused feature row."""

    def __init__(self, ocr_pipeline, visual_service: VisualEmbeddingService, pca_reducer: VisualPCAReducer):
        self.ocr_pipeline = ocr_pipeline
        self.visual_service = visual_service
        self.pca_reducer = pca_reducer

    def build(self, image_path: str) -> pd.Series:
        ocr_result = self.ocr_pipeline.extract(image_path)
        embedding, _ = self.visual_service.embed(image_path)
        return build_feature_vector(ocr_result, embedding, self.pca_reducer, image_path=image_path)
