"""
evaluate.py

Loads the ALREADY-TRAINED models from saved_models/ and text_branch/saved_model/,
re-runs them on a held-out test split of data/processed/emails.csv, and prints the
same Accuracy / Precision / Recall / F1 / ROC-AUC report that train.py prints --
WITHOUT retraining anything. Takes 1-2 minutes instead of ~40.

Also saves:
  - results/confusion_matrix.png       (TEXT branch - this is the number
    reported in the paper as "Email - Text Branch (DistilBERT)")
  - results/feature_importance.png     (METADATA branch, LightGBM - text
    branch is a transformer, so it has no feature_importances_ equivalent)

Usage:
    python evaluate.py
    python evaluate.py --csv path/to/data.csv --test-size 0.25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)

sys.path.insert(0, str(Path(__file__).parent))

from metadata_branch.feature_extraction import extract_features
from metadata_branch.lightgbm_model import EmailMetadataClassifier
from text_branch.distilbert_branch import get_text_branch, FALLBACK_MODEL_PATH, DISTILBERT_MODEL_DIR
from fusion.mlp_fusion import EmailFusionMLP

MODEL_DIR = Path(__file__).parent / "saved_models"
METADATA_MODEL_PATH = MODEL_DIR / "metadata_lightgbm.joblib"
FUSION_MODEL_PATH = MODEL_DIR / "fusion_mlp.joblib"
RESULTS_DIR = Path(__file__).parent / "results"


def _report(name: str, y_true, y_pred_proba, threshold: float = 0.5):
    y_pred = (np.asarray(y_pred_proba) >= threshold).astype(int)
    print(f"\n[{name}]")
    print(f"  Accuracy : {accuracy_score(y_true, y_pred):.3f}")
    print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"  Recall   : {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"  F1       : {f1_score(y_true, y_pred, zero_division=0):.3f}")
    try:
        print(f"  ROC-AUC  : {roc_auc_score(y_true, y_pred_proba):.3f}")
    except ValueError:
        print("  ROC-AUC  : n/a (single class in this split)")
    return y_pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(Path(__file__).parent / "data" / "processed" / "emails.csv"))
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Dataset not found at {csv_path}.")
        sys.exit(1)

    if not METADATA_MODEL_PATH.exists() or not FUSION_MODEL_PATH.exists():
        print("Saved models not found in saved_models/. Run train.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} labeled emails from {csv_path} "
          f"({df['label'].sum()} phishing / {(df['label'] == 0).sum()} legitimate)")

    # Same random_state=42 split as train.py, so this reproduces the exact test set
    # the models were evaluated on during training.
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=42, stratify=df["label"]
    )
    print(f"Train: {len(train_df)}  |  Test: {len(test_df)}")

    y_test = test_df["label"].to_numpy()

    # ---------------------------------------------------------------------------
    # 1. Metadata branch
    # ---------------------------------------------------------------------------
    print("\n=== Loading metadata branch (LightGBM) ===")
    metadata_clf = EmailMetadataClassifier.load(METADATA_MODEL_PATH)
    test_meta_features = [extract_features(e) for e in test_df["raw_email"]]
    X_meta_test = np.array([f.to_vector() for f in test_meta_features])
    meta_test_proba = metadata_clf.predict_proba_batch(X_meta_test)
    _report("Metadata branch (test set)", y_test, meta_test_proba)
    print("Top metadata feature importances:", list(metadata_clf.feature_importances().items())[:5])

    # ---------------------------------------------------------------------------
    # 2. Text branch
    # ---------------------------------------------------------------------------
    print("\n=== Loading text branch ===")
    from metadata_branch.feature_extraction import extract_body_text
    test_bodies = [extract_body_text(e) or " " for e in test_df["raw_email"]]

    text_branch = get_text_branch(prefer_distilbert=True)
    is_fallback = text_branch.__class__.__name__ == "TfidfFallbackBranch"
    print(f"Using text branch implementation: {text_branch.__class__.__name__}")
    if is_fallback:
        print("  WARNING: DistilBERT weights not found/loaded -- this is the TF-IDF fallback, "
              "not the fine-tuned DistilBERT model.")

    text_test_proba = np.array([text_branch.predict_proba(t) for t in test_bodies])
    text_test_pred = _report("Text branch (test set)", y_test, text_test_proba)

    # ---------------------------------------------------------------------------
    # 3. Fusion MLP
    # ---------------------------------------------------------------------------
    print("\n=== Loading fusion MLP ===")
    fusion = EmailFusionMLP.load(FUSION_MODEL_PATH)
    X_fusion_test = np.array([
        fusion.build_feature_vector(tp, mp, mf)
        for tp, mp, mf in zip(text_test_proba, meta_test_proba, test_meta_features)
    ])
    fusion_test_proba = fusion.model.predict_proba(X_fusion_test)[:, 1]
    _report("Fused email module (test set)", y_test, fusion_test_proba)

    print("\nDone. No models were retrained -- this only re-evaluated saved_models/ "
          "and text_branch/saved_model/ on the held-out test split.")

    # ---------------------------------------------------------------------------
    # Save plots for the paper (matches the "Email - Text Branch" row in Table II)
    # ---------------------------------------------------------------------------
    RESULTS_DIR.mkdir(exist_ok=True)

    cm = confusion_matrix(y_test, text_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=300)
    plt.close()
    print(f"\nSaved -> {RESULTS_DIR / 'confusion_matrix.png'} (text branch)")

    importances = metadata_clf.feature_importances()
    names = list(importances.keys())[:20]
    values = list(importances.values())[:20]
    plt.figure(figsize=(10, 8))
    plt.barh(names, values)
    plt.gca().invert_yaxis()
    plt.title("Top 20 Feature Importance (Metadata Branch)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=300)
    plt.close()
    print(f"Saved -> {RESULTS_DIR / 'feature_importance.png'} (metadata branch)")


if __name__ == "__main__":
    main()