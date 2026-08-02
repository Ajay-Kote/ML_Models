"""
Module 4 - Payment Image Detection
Inference: single screenshot -> fraud probability + structured fields
           + tampering-localization heatmap (Grad-CAM) + SHAP top drivers.

This is the entry point the FastAPI backend orchestrator (Section 9,
backend/app.py) calls for this module, and matches the "Output" contract
defined in Section 5.4 of the design doc:
    Fraud probability, extracted structured transaction fields,
    and a tampering-localization heatmap.
"""

import argparse
import os
from dataclasses import asdict

import joblib
import numpy as np

from explainability.gradcam_explainer import PaymentGradCAM
from explainability.shap_explainer import PaymentSHAPExplainer
from fusion.feature_fusion import VisualPCAReducer, build_feature_vector
from ocr.paddleocr_pipeline import PaymentOCRPipeline
from vision.efficientnet_extractor import VisualEmbeddingService


def _safe_load_artifact(artifacts_dir: str, filename: str):
    path = os.path.join(artifacts_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing artifact: {path}")
    return joblib.load(path)


class PaymentImageDetector:
    def __init__(self, artifacts_dir: str = "models/artifacts"):
        self.clf = _safe_load_artifact(artifacts_dir, "lgbm_fraud_classifier.pkl")
        # calibrated_clf drives the actual fraud_probability output (see
        # models/train.py for why: raw LightGBM probabilities under
        # class_weight="balanced" tend to cluster near 0.5 even when
        # correctly ranked). self.clf (raw, uncalibrated) is kept
        # separately because shap.TreeExplainer needs direct tree access
        # and can't work through the calibration wrapper.
        # Falls back to the raw model if artifacts were produced by an
        # older version of train.py that didn't save a calibrated model.
        calibrated_path = os.path.join(artifacts_dir, "lgbm_fraud_classifier_calibrated.pkl")
        self.calibrated_clf = (
            joblib.load(calibrated_path) if os.path.exists(calibrated_path) else self.clf
        )
        self.feature_columns = _safe_load_artifact(artifacts_dir, "feature_columns.pkl")
        self.pca_reducer = VisualPCAReducer().load(os.path.join(artifacts_dir, "visual_pca.pkl"))

        # Tuned decision threshold (see models/train.py) -- the default 0.5
        # cutoff combined with class_weight="balanced" on this imbalanced
        # dataset produced a very high false-positive rate on genuine
        # images. Falls back to 0.5 if artifacts predate this fix.
        threshold_path = os.path.join(artifacts_dir, "decision_threshold.pkl")
        self.decision_threshold = joblib.load(threshold_path) if os.path.exists(threshold_path) else 0.5

        self.ocr_pipeline = PaymentOCRPipeline()
        self.visual_service = VisualEmbeddingService()
        self.gradcam = PaymentGradCAM(self.visual_service.model)
        self.shap_explainer = PaymentSHAPExplainer(self.clf, self.feature_columns)

    def predict(self, image_path: str, heatmap_out_path: str = None) -> dict:
        ocr_result = self.ocr_pipeline.extract(image_path)
        embedding, pil_img = self.visual_service.embed(image_path)

        feature_row = build_feature_vector(ocr_result, embedding, self.pca_reducer, image_path=image_path)
        feature_row = feature_row.reindex(self.feature_columns).fillna(-1)
        X = feature_row.to_frame().T

        fraud_prob = float(self.calibrated_clf.predict_proba(X)[0, 1])
        is_fraud = fraud_prob >= self.decision_threshold
        top_drivers = self.shap_explainer.top_contributors(X, k=5)

        if heatmap_out_path:
            os.makedirs(os.path.dirname(heatmap_out_path) or ".", exist_ok=True)

        heatmap_path = None
        if heatmap_out_path:
            heatmap_path = self.gradcam.generate(image_path, heatmap_out_path)

        return {
            "fraud_probability": round(fraud_prob, 4),
            "is_fraud": is_fraud,
            "decision_threshold": self.decision_threshold,
            "structured_fields": asdict(ocr_result.structured),
            "ocr_quality": asdict(ocr_result.quality),
            "top_contributing_features": top_drivers,
            "tampering_heatmap_path": heatmap_path,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--artifacts_dir", default="models/artifacts")
    parser.add_argument("--heatmap_out", default="heatmap_output.jpg")
    args = parser.parse_args()

    detector = PaymentImageDetector(args.artifacts_dir)
    result = detector.predict(args.image_path, heatmap_out_path=args.heatmap_out)

    print("Fraud probability:", result["fraud_probability"])
    print(f"Is fraud (threshold={result['decision_threshold']:.3f}):", result["is_fraud"])
    print("Structured fields:", result["structured_fields"])
    print("Top contributing features:", result["top_contributing_features"])
    print("Heatmap saved to:", result["tampering_heatmap_path"])