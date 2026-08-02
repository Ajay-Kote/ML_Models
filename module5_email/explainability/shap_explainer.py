"""
explainability/shap_explainer.py

Design doc Section 7: "LightGBM (URL, QR, Email metadata) -> SHAP (TreeExplainer) ->
Per-feature contribution to the phishing/fraud score."
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict

import shap

from metadata_branch.lightgbm_model import EmailMetadataClassifier
from metadata_branch.feature_extraction import EmailMetadataFeatures


def explain_metadata(
    classifier: EmailMetadataClassifier,
    features: EmailMetadataFeatures,
    top_k: int = 5,
) -> List[Dict]:
    """
    Returns the top_k metadata features ranked by |SHAP value|, e.g.:
        [{"feature": "dkim_pass", "value": 0, "shap_contribution": 0.21}, ...]
    Positive shap_contribution pushes the prediction toward "phishing".
    """
    explainer = shap.TreeExplainer(classifier.model)
    vec = np.array(features.to_vector(), dtype=float).reshape(1, -1)
    shap_values = explainer.shap_values(vec)

    # LightGBM binary classifiers: shap_values may be a single array (recent SHAP) or a
    # 2-element list [class0, class1] (older SHAP) — normalize to the "phishing" class view.
    if isinstance(shap_values, list):
        values = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
    else:
        values = shap_values[0]
        if values.ndim > 1:
            values = values[:, 1] if values.shape[-1] > 1 else values[:, 0]

    contributions = list(zip(classifier.feature_names, vec[0].tolist(), np.asarray(values).ravel().tolist()))
    contributions.sort(key=lambda item: -abs(item[2]))

    return [
        {"feature": name, "value": value, "shap_contribution": round(shap_val, 4)}
        for name, value, shap_val in contributions[:top_k]
    ]
