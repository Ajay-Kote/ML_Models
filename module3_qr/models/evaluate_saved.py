"""
evaluate_saved.py (QR module)
------------------------------
Loads the ALREADY-TRAINED model (saved_model.pkl) and evaluates it -
does NOT call clf.fit() again, so this skips the slow training step
(300-round boosting with early stopping) entirely. It still has to
re-extract features from the images (that part isn't cached anywhere),
but training itself, usually the slower part, is skipped.

Uses the exact same train_test_split (test_size=0.2, random_state=42)
as train.py, so this evaluates on the same held-out rows the model
was originally tested on.

Generates confusion_matrix.png and feature_importance.png for the paper.
"""
import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from feature_extraction.build_features import build_feature_vector, feature_names

THIS_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(THIS_DIR, "..", "data", "raw")
MODEL_PATH = os.path.join(THIS_DIR, "saved_model.pkl")
RESULTS_DIR = os.path.join(THIS_DIR, "..", "results")


def load_manifest(raw_dir):
    manifest_path = os.path.join(raw_dir, "metadata.jsonl")
    rows = []
    with open(manifest_path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_dataset(raw_dir):
    rows = load_manifest(raw_dir)
    img_dir = os.path.join(raw_dir, "images")
    X, y = [], []
    for row in rows:
        img_path = os.path.join(img_dir, row["file"])
        vec = build_feature_vector(img_path, manifest_row=None)
        X.append(vec)
        y.append(row["label"])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main():
    print("Re-extracting features (no training - just loading saved model)...")
    X, y = build_dataset(RAW_DIR)
    print(f"Dataset shape: {X.shape}, positives: {y.sum()}, negatives: {(y == 0).sum()}")

    # Same split as train.py -> same held-out test rows
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Loading saved model (no retraining)...")
    saved = joblib.load(MODEL_PATH)
    clf = saved["model"]
    names = saved.get("feature_names") or feature_names()

    proba = clf.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary")
    auc = roc_auc_score(y_test, proba)

    print("\n=== Evaluation ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()
    print(f"Saved -> {os.path.join(RESULTS_DIR, 'confusion_matrix.png')}")

    # ---- Feature Importance ----
    importance = pd.DataFrame({
        "Feature": names,
        "Importance": clf.feature_importances_
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