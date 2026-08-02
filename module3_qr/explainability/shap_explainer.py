"""
shap_explainer.py
------------------
SHAP TreeExplainer wrapper for the QR module's LightGBM classifier
(Section 7: "LightGBM -> SHAP (TreeExplainer) -> Per-feature contribution").

Pixel features (px_0000 ... px_1023) are individually meaningless to a human,
so contributions from that block are aggregated into a single "visual pixel
pattern" line item, while metadata and texture features (which map to
concrete, interpretable QR properties) are surfaced individually by name --
this is exactly the human-interpretability improvement over "anonymous pixel
index" explanations called out in the design document's Section 5.3 note.
"""
import numpy as np
import shap

READABLE_NAMES = {
    "qr_version": "QR version (payload capacity)",
    "error_correction_level": "error-correction level",
    "module_density": "module density (grid complexity)",
    "quiet_zone_size": "quiet-zone (border) size",
    "metadata_confidence": "metadata extraction confidence",
    "edge_density": "edge sharpness / density",
    "contrast": "image contrast",
    "entropy": "pixel-intensity entropy",
}


def _readable(name):
    if name.startswith("px_"):
        return "visual pixel pattern"
    if name.startswith("lbp_bin_"):
        return "local texture pattern (LBP)"
    return READABLE_NAMES.get(name, name)


class QRExplainer:
    def __init__(self, model, feature_names, background_path):
        self.model = model
        self.feature_names = feature_names
        background = np.load(background_path)
        self.explainer = shap.TreeExplainer(model, background)

    def explain(self, feature_vector, top_k=4):
        shap_values = self.explainer.shap_values(feature_vector.reshape(1, -1))
        # LightGBM binary classifier via TreeExplainer returns a single array
        # for the positive class in recent SHAP versions, or a 2-list [neg,pos].
        values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        # Aggregate raw pixel/LBP contributions into single human-readable buckets,
        # keep metadata/texture-summary features individual.
        buckets = {}
        for name, val in zip(self.feature_names, values):
            key = _readable(name)
            buckets[key] = buckets.get(key, 0.0) + float(val)

        ranked = sorted(buckets.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top = ranked[:top_k]

        top_features = [
            {"feature": name, "shap_contribution": round(val, 4),
             "direction": "increases risk" if val > 0 else "decreases risk"}
            for name, val in top
        ]

        risk_phrases = [f"{f['feature']}" for f in top_features if f["shap_contribution"] > 0]
        if risk_phrases:
            explanation = (
                "Flagged as high risk primarily due to " + ", ".join(risk_phrases[:3]) + "."
            )
        else:
            explanation = "Scored as low risk; no strongly suspicious features detected."

        return explanation, top_features
