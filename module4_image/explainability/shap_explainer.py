"""
Module 4 - Payment Image Detection
Explainability: SHAP (TreeExplainer) over the fused LightGBM classifier.
Reveals which structured/quality/visual-PC features pushed the score
toward fraud vs genuine (Section 7 of the design doc).
"""

from typing import List

import pandas as pd
import shap


class PaymentSHAPExplainer:
    def __init__(self, lgbm_model, feature_columns: List[str]):
        self.explainer = shap.TreeExplainer(lgbm_model)
        self.feature_columns = feature_columns

    def shap_values(self, X: pd.DataFrame):
        return self.explainer.shap_values(X)

    def top_contributors(self, X: pd.DataFrame, k: int = 5) -> List[dict]:
        """Returns the top-k features by |SHAP value| for a single-row X."""
        raw = self.explainer.shap_values(X)
        # For binary LGBMClassifier, shap_values may return a list [class0, class1]
        vals = raw[1][0] if isinstance(raw, list) else raw[0]

        contributions = sorted(
            zip(self.feature_columns, vals),
            key=lambda t: abs(t[1]),
            reverse=True,
        )[:k]

        return [
            {"feature": name, "shap_value": round(float(val), 4)}
            for name, val in contributions
        ]

    def human_readable_summary(self, X: pd.DataFrame, k: int = 3) -> str:
        top = self.top_contributors(X, k=k)
        parts = [
            f"{c['feature']} ({'+' if c['shap_value'] > 0 else ''}{c['shap_value']})"
            for c in top
        ]
        return "Primary drivers of this score: " + ", ".join(parts)
