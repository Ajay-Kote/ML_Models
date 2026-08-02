"""
metadata_branch/lightgbm_model.py

Thin, testable wrapper around a LightGBM classifier trained on the structured metadata
feature vector produced by `feature_extraction.py`. Matches design doc Section 5.5:
"Model: ... LightGBM (metadata features)".
"""

from __future__ import annotations

import joblib
import numpy as np
import lightgbm as lgb
from pathlib import Path
from typing import Optional

from metadata_branch.feature_extraction import FEATURE_ORDER, EmailMetadataFeatures


class EmailMetadataClassifier:
    """LightGBM binary classifier: phishing (1) vs legitimate (0), from metadata features."""

    def __init__(self, model: Optional[lgb.LGBMClassifier] = None):
        self.model = model or lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=15,
            min_child_samples=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary",
            random_state=42,
            verbosity=-1,
        )
        self.feature_names = FEATURE_ORDER

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EmailMetadataClassifier":
        self.model.fit(X, y, feature_name=self.feature_names)
        return self

    def predict_proba(self, features: EmailMetadataFeatures | np.ndarray) -> float:
        vec = self._as_vector(features)
        return float(self.model.predict_proba(self._as_frame(vec.reshape(1, -1)))[0, 1])

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._as_frame(X))[:, 1]

    def _as_frame(self, X: np.ndarray):
        import pandas as pd
        return pd.DataFrame(X, columns=self.feature_names)

    def _as_vector(self, features: EmailMetadataFeatures | np.ndarray) -> np.ndarray:
        if isinstance(features, EmailMetadataFeatures):
            return np.array(features.to_vector(), dtype=float)
        return np.asarray(features, dtype=float)

    def feature_importances(self) -> dict:
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            return {}
        return dict(sorted(zip(self.feature_names, importances.tolist()), key=lambda kv: -kv[1]))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, path)

    @classmethod
    def load(cls, path: str | Path) -> "EmailMetadataClassifier":
        payload = joblib.load(path)
        obj = cls(model=payload["model"])
        obj.feature_names = payload["feature_names"]
        return obj
