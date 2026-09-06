"""
evaluate_saved.py (Payment Screenshot module)
-----------------------------------------------
Does NOT retrain anything. Reuses:
  - models/artifacts/feature_cache.joblib  (OCR + visual embeddings, already computed)
  - models/artifacts/visual_pca.pkl        (already-fitted PCA reducer)
  - models/artifacts/lgbm_fraud_classifier.pkl              (raw model, for feature_importances_)
  - models/artifacts/lgbm_fraud_classifier_calibrated.pkl   (calibrated model, for predictions)
  - models/artifacts/decision_threshold.pkl                 (tuned threshold)
  - models/artifacts/feature_columns.pkl                    (exact column order used at train time)

Reproduces the EXACT same train/val/test split as train.py (same two
train_test_split calls, same random_state=42) so this evaluates on the
same held-out test rows the saved metrics (eval_metrics.json) came from.

Generates confusion_matrix.png and feature_importance.png for the paper.
"""
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.train import normalize_manifest, build_feature_vector
from fusion.feature_fusion import VisualPCAReducer

THIS_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(THIS_DIR, "..", "data")
ARTIFACTS_DIR = os.path.join(THIS_DIR, "artifacts")
RESULTS_DIR = os.path.join(THIS_DIR, "..", "results")
MANIFEST_PATH = os.path.join(DATA_DIR, "labels_all.csv")


def main():
    print("Loading manifest...")
    manifest = normalize_manifest(pd.read_csv(MANIFEST_PATH), MANIFEST_PATH)

    print("Loading cached OCR + visual-embedding features (no re-extraction)...")
    cache = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_cache.joblib"))
    pca_reducer = VisualPCAReducer().load(os.path.join(ARTIFACTS_DIR, "visual_pca.pkl"))
    feature_columns = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_columns.pkl"))
    threshold = joblib.load(os.path.join(ARTIFACTS_DIR, "decision_threshold.pkl"))
    raw_clf = joblib.load(os.path.join(ARTIFACTS_DIR, "lgbm_fraud_classifier.pkl"))
    calibrated_clf = joblib.load(os.path.join(ARTIFACTS_DIR, "lgbm_fraud_classifier_calibrated.pkl"))

    from ocr.paddleocr_pipeline import OCRResult, OCRStructuredFields, OCRQualitySignals

    # The cache keys might not exactly match manifest["image_path"] strings
    # (e.g. different path separators, or absolute vs relative paths from
    # a different working directory when train.py was originally run).
    # Build a basename-based lookup as a fallback.
    cache_by_basename = {}
    for k in cache.keys():
        cache_by_basename.setdefault(os.path.basename(k), k)

    print(f"Cache has {len(cache)} entries. Example keys: {list(cache.keys())[:3]}")
    print(f"Manifest example paths: {manifest['image_path'].tolist()[:3]}")

    rows, labels, paths = [], [], []
    misses = 0
    for _, row in manifest.iterrows():
        path = row["image_path"]
        entry = cache.get(path)
        if entry is None:
            # fallback: match by basename
            alt_key = cache_by_basename.get(os.path.basename(path))
            if alt_key is not None:
                entry = cache.get(alt_key)
        if entry is None or "embedding" not in entry:
            misses += 1
            continue
        structured = OCRStructuredFields(**entry["structured"]) if entry.get("structured") else OCRStructuredFields()
        quality = OCRQualitySignals(**entry["quality"]) if entry.get("quality") else OCRQualitySignals()
        ocr_res = OCRResult(raw_text=entry.get("raw_text", ""), structured=structured, quality=quality)
        emb = np.asarray(entry["embedding"])
        rows.append(build_feature_vector(ocr_res, emb, pca_reducer, image_path=path))
        labels.append(row["label"])
        paths.append(path)

    print(f"Matched {len(rows)} rows from cache, {misses} misses.")

    X = pd.DataFrame(rows).fillna(-1)
    X = X.reindex(columns=feature_columns, fill_value=-1)  # exact same column order as training
    y = np.array(labels)

    print(f"Reconstructed {X.shape[0]} feature rows from cache, {X.shape[1]} features.")

    # Same two-step split as train.py
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
    )

    print(f"Reproduced test split: {X_test.shape[0]} samples (should match eval_metrics.json)")

    y_test_proba = calibrated_clf.predict_proba(X_test)[:, 1]
    preds = (y_test_proba >= threshold).astype(int)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()
    print(f"Saved -> {os.path.join(RESULTS_DIR, 'confusion_matrix.png')}")
    print("Confusion matrix:\n", cm)

    # ---- Feature Importance (from the RAW LightGBM model) ----
    importance = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": raw_clf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 8))
    plt.barh(importance["Feature"][:20], importance["Importance"][:20])
    plt.gca().invert_yaxis()
    plt.title("Top 20 Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=300)
    plt.close()
    print(f"Saved -> {os.path.join(RESULTS_DIR, 'feature_importance.png')}")


if __name__ == "__main__":
    main()