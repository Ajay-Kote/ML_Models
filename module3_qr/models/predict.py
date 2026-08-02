"""
predict.py
----------
Inference entry point for the QR Code Analysis Module.

Given a QR image, returns:
  - malicious_probability (float, 0-1)
  - decoded_url (str or None) -- optionally forwarded to the URL Detection
    Module for a secondary check, per Section 5.3's Output spec.
  - explanation -- human-readable SHAP-based justification (Section 7)
"""
import os
import sys

import cv2
import joblib
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from feature_extraction.build_features import build_feature_vector
from explainability.shap_explainer import QRExplainer

THIS_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(THIS_DIR, "saved_model.pkl")
BG_PATH = os.path.join(THIS_DIR, "shap_background.npy")

_model_bundle = None
_explainer = None


def _load():
    global _model_bundle, _explainer
    if _model_bundle is None:
        _model_bundle = joblib.load(MODEL_PATH)
        _explainer = QRExplainer(_model_bundle["model"], _model_bundle["feature_names"], BG_PATH)
    return _model_bundle, _explainer


def decode_url(image_path):
    """Attempt to decode the QR payload without visiting it (Section 5.3 design intent)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(img)
    return data if data else None


def predict(image_path, top_k=4):
    bundle, explainer = _load()
    model = bundle["model"]

    features = build_feature_vector(image_path, manifest_row=None)  # inference: estimate metadata from image
    proba = float(model.predict_proba(features.reshape(1, -1))[0, 1])
    url = decode_url(image_path)
    explanation, top_features = explainer.explain(features, top_k=top_k)

    return {
        "malicious_qr_probability": round(proba, 4),
        "label": "malicious" if proba >= 0.5 else "legitimate",
        "decoded_url": url,
        "top_contributing_features": top_features,
        "explanation": explanation,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_qr_image.png>")
        sys.exit(1)
    result = predict(sys.argv[1])
    import json
    print(json.dumps(result, indent=2, default=str))
