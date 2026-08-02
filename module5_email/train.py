"""
train.py

Trains the three components of the Email Detection Module end-to-end, in the order
required by the fusion layer's dependency (fusion needs branch outputs as inputs):

    1. Metadata branch  -> LightGBM on structured header/body features
    2. Text branch       -> DistilBERT (or TF-IDF fallback) on raw body text
    3. Fusion layer       -> small MLP on [text_prob, metadata_prob, + aux metadata signals]

Usage:
    python train.py                       # uses data/processed/emails.csv
    python train.py --csv path/to/data.csv --test-size 0.25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from metadata_branch.feature_extraction import extract_features, extract_body_text
from metadata_branch.lightgbm_model import EmailMetadataClassifier
from text_branch.distilbert_branch import get_text_branch, FALLBACK_MODEL_PATH, DISTILBERT_MODEL_DIR
from fusion.mlp_fusion import EmailFusionMLP

MODEL_DIR = Path(__file__).parent / "saved_models"
METADATA_MODEL_PATH = MODEL_DIR / "metadata_lightgbm.joblib"
FUSION_MODEL_PATH = MODEL_DIR / "fusion_mlp.joblib"


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


def main():
    parser = argparse.ArgumentParser(description="Train the Email Detection Module (Section 5.5).")
    parser.add_argument("--csv", default=str(Path(__file__).parent / "data" / "processed" / "emails.csv"))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--text-epochs", type=int, default=3, help="Only used if the real DistilBERT branch loads.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Dataset not found at {csv_path}.\nRun: python data/generate_synthetic_data.py")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} labeled emails from {csv_path} "
          f"({df['label'].sum()} phishing / {(df['label'] == 0).sum()} legitimate)")

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=42, stratify=df["label"]
    )
    print(f"Train: {len(train_df)}  |  Test: {len(test_df)}")

    # ---------------------------------------------------------------------------
    # 1. Metadata branch (LightGBM)
    # ---------------------------------------------------------------------------
    print("\n=== Training metadata branch (LightGBM) ===")
    train_meta_features = [extract_features(e) for e in train_df["raw_email"]]
    test_meta_features = [extract_features(e) for e in test_df["raw_email"]]

    X_meta_train = np.array([f.to_vector() for f in train_meta_features])
    X_meta_test = np.array([f.to_vector() for f in test_meta_features])
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    metadata_clf = EmailMetadataClassifier().fit(X_meta_train, y_train)
    meta_test_proba = metadata_clf.predict_proba_batch(X_meta_test)
    _report("Metadata branch (test set)", y_test, meta_test_proba)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metadata_clf.save(METADATA_MODEL_PATH)
    print(f"Saved metadata model -> {METADATA_MODEL_PATH}")
    print("Top metadata feature importances:", list(metadata_clf.feature_importances().items())[:5])

    # ---------------------------------------------------------------------------
    # 2. Text branch (DistilBERT, falling back to TF-IDF automatically)
    # ---------------------------------------------------------------------------
    print("\n=== Training text branch ===")
    train_bodies = [extract_body_text(e) or " " for e in train_df["raw_email"]]
    test_bodies = [extract_body_text(e) or " " for e in test_df["raw_email"]]

    text_branch = get_text_branch(prefer_distilbert=True)
    is_fallback = text_branch.__class__.__name__ == "TfidfFallbackBranch"
    print(f"Using text branch implementation: {text_branch.__class__.__name__}"
          + ("  (real DistilBERT/transformers not available in this environment — "
             "install torch+transformers and ensure network access to switch to it)" if is_fallback else ""))

    if is_fallback:
        text_branch.fit(train_bodies, y_train.tolist())
        text_branch.save(FALLBACK_MODEL_PATH)
        print(f"Saved TF-IDF fallback text model -> {FALLBACK_MODEL_PATH}")
    else:
        text_branch.fit(train_bodies, y_train.tolist(), epochs=args.text_epochs)
        text_branch.save(DISTILBERT_MODEL_DIR)
        print(f"Saved fine-tuned DistilBERT text model -> {DISTILBERT_MODEL_DIR}")

    text_test_proba = np.array([text_branch.predict_proba(t) for t in test_bodies])
    _report("Text branch (test set)", y_test, text_test_proba)

    # ---------------------------------------------------------------------------
    # 3. Fusion MLP
    # ---------------------------------------------------------------------------
    print("\n=== Training fusion MLP ===")
    text_train_proba = np.array([text_branch.predict_proba(t) for t in train_bodies])
    meta_train_proba = metadata_clf.predict_proba_batch(X_meta_train)

    fusion = EmailFusionMLP()
    X_fusion_train = np.array([
        fusion.build_feature_vector(tp, mp, mf)
        for tp, mp, mf in zip(text_train_proba, meta_train_proba, train_meta_features)
    ])
    X_fusion_test = np.array([
        fusion.build_feature_vector(tp, mp, mf)
        for tp, mp, mf in zip(text_test_proba, meta_test_proba, test_meta_features)
    ])

    fusion.fit(X_fusion_train, y_train)
    fusion_test_proba = fusion.model.predict_proba(X_fusion_test)[:, 1]
    _report("Fused email module (test set)", y_test, fusion_test_proba)

    fusion.save(FUSION_MODEL_PATH)
    print(f"Saved fusion model -> {FUSION_MODEL_PATH}")

    print("\nAll three components trained and saved under saved_models/ "
          "(plus text_branch/saved_model/). Run predict.py to try inference.")


if __name__ == "__main__":
    main()
