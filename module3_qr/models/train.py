"""
train.py
--------
Trains the QR Code Analysis Module's LightGBM classifier on concatenated
pixel + metadata + texture features (Section 5.3), evaluates it (Section 11.1
per-module metrics), and saves the model + a SHAP background sample for the
explainability layer (Section 7).
"""
import json
import os
import sys

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from feature_extraction.build_features import build_feature_vector, feature_names

THIS_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(THIS_DIR, "..", "data", "raw")
MODEL_OUT = os.path.join(THIS_DIR, "saved_model.pkl")
BG_SAMPLE_OUT = os.path.join(THIS_DIR, "shap_background.npy")


def load_manifest(raw_dir):
    manifest_path = os.path.join(raw_dir, "metadata.jsonl")
    rows = []
    with open(manifest_path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_dataset(raw_dir):
    """
    NOTE: metadata is deliberately extracted the SAME way as at inference
    time (estimated straight from the image via `from_image`, not the
    generation-time ground truth) so the model never sees a metadata
    distribution at train time that it won't also see in production. Using
    the exact generation parameters here would create train/serve skew and
    silently inflate offline metrics.
    """
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
    print("Loading manifest and extracting features...")
    X, y = build_dataset(RAW_DIR)
    print(f"Dataset shape: {X.shape}, positives: {y.sum()}, negatives: {(y == 0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )

    proba = clf.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary")
    auc = roc_auc_score(y_test, proba)

    print("\n=== Evaluation (Section 11.1 Per-Module Metrics) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    joblib.dump({"model": clf, "feature_names": feature_names()}, MODEL_OUT)
    # Small background sample for SHAP TreeExplainer expected-value baseline
    bg = X_train[np.random.RandomState(0).choice(len(X_train), size=min(100, len(X_train)), replace=False)]
    np.save(BG_SAMPLE_OUT, bg)

    print(f"\nSaved model -> {MODEL_OUT}")
    print(f"Saved SHAP background sample -> {BG_SAMPLE_OUT}")


if __name__ == "__main__":
    main()
