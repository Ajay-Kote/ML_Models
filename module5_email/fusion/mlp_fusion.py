"""
fusion/mlp_fusion.py

Small MLP that fuses the text branch (DistilBERT / TF-IDF fallback) output with the
metadata branch (LightGBM) output into a single calibrated email phishing probability,
per design doc Section 5.5: "combined through a small MLP fusion layer."

Fusion input vector (kept intentionally small/interpretable):
    [text_prob, metadata_prob, urgency_keyword_count, link_display_mismatch_count,
     spf_pass, dkim_pass, dmarc_pass, reply_to_mismatch]

Rationale: feeding the two branch probabilities plus a handful of the most decision-relevant
raw metadata signals lets the MLP learn calibrated, non-linear interactions (e.g. "text branch
alone is only mildly suspicious, but combined with a DKIM failure the risk compounds")
without needing the full 768-dim DistilBERT embedding, which would make this fusion layer
much larger/harder to train from a small dataset. This mirrors the top-level Adaptive Risk
Fusion Engine design rationale in Section 6.1 ("a meta-classifier learns the weighting
function directly from data").
"""

from __future__ import annotations

import joblib
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPClassifier

FUSION_FEATURE_NAMES = [
    "text_prob",
    "metadata_prob",
    "urgency_keyword_count",
    "link_display_mismatch_count",
    "spf_pass",
    "dkim_pass",
    "dmarc_pass",
    "reply_to_mismatch",
]


class EmailFusionMLP:
    def __init__(self, model: MLPClassifier | None = None):
        self.model = model or MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            max_iter=2000,
            random_state=42,
        )
        self.feature_names = FUSION_FEATURE_NAMES

    @staticmethod
    def build_feature_vector(text_prob: float, metadata_prob: float, meta_features: "EmailMetadataFeatures") -> np.ndarray:
        return np.array([
            text_prob,
            metadata_prob,
            meta_features.urgency_keyword_count,
            meta_features.link_display_mismatch_count,
            meta_features.spf_pass,
            meta_features.dkim_pass,
            meta_features.dmarc_pass,
            meta_features.reply_to_mismatch,
        ], dtype=float)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EmailFusionMLP":
        self.model.fit(X, y)
        return self

    def predict_proba(self, feature_vector: np.ndarray) -> float:
        return float(self.model.predict_proba(feature_vector.reshape(1, -1))[0, 1])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, path)

    @classmethod
    def load(cls, path: str | Path) -> "EmailFusionMLP":
        payload = joblib.load(path)
        obj = cls(model=payload["model"])
        obj.feature_names = payload["feature_names"]
        return obj
