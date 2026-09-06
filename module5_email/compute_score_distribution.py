"""
compute_email_score_dist.py

Run this from inside module5_email/ (same folder as evaluate.py).

Loads the CURRENT saved email model (metadata + text + fusion), gets its
fused risk_probability on the held-out test split, and prints the
(mean, std) for malicious vs legitimate cases -- exactly the numbers
needed to update SCORE_DIST["email"] in
fusion_engine/generate_synthetic_dataset.py.

Usage:
    cd module5_email
    python compute_email_score_dist.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from metadata_branch.feature_extraction import extract_features, extract_body_text
from metadata_branch.lightgbm_model import EmailMetadataClassifier
from text_branch.distilbert_branch import get_text_branch
from fusion.mlp_fusion import EmailFusionMLP

MODEL_DIR = Path(__file__).parent / "saved_models"
CSV_PATH = Path(__file__).parent / "data" / "processed" / "emails.csv"


def main():
    df = pd.read_csv(CSV_PATH)
    _, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df["label"])
    y_test = test_df["label"].to_numpy()

    metadata_clf = EmailMetadataClassifier.load(MODEL_DIR / "metadata_lightgbm.joblib")
    test_meta_features = [extract_features(e) for e in test_df["raw_email"]]
    X_meta_test = np.array([f.to_vector() for f in test_meta_features])
    meta_test_proba = metadata_clf.predict_proba_batch(X_meta_test)

    text_branch = get_text_branch(prefer_distilbert=True)
    print(f"Using text branch implementation: {text_branch.__class__.__name__}")
    test_bodies = [extract_body_text(e) or " " for e in test_df["raw_email"]]
    text_test_proba = np.array([text_branch.predict_proba(t) for t in test_bodies])

    fusion = EmailFusionMLP.load(MODEL_DIR / "fusion_mlp.joblib")
    X_fusion_test = np.array([
        fusion.build_feature_vector(tp, mp, mf)
        for tp, mp, mf in zip(text_test_proba, meta_test_proba, test_meta_features)
    ])
    fused_proba = fusion.model.predict_proba(X_fusion_test)[:, 1]

    malicious_scores = fused_proba[y_test == 1]
    legit_scores = fused_proba[y_test == 0]

    mal_mean, mal_std = float(np.mean(malicious_scores)), float(np.std(malicious_scores))
    leg_mean, leg_std = float(np.mean(legit_scores)), float(np.std(legit_scores))

    # floor std so downstream Gaussian sampling in generate_synthetic_dataset.py
    # never divides by / samples from a zero-variance distribution
    mal_std = max(mal_std, 0.01)
    leg_std = max(leg_std, 0.01)

    print("\n=== Email module fused risk_probability distribution (test set) ===")
    print(f"malicious : mean={mal_mean:.4f}, std={mal_std:.4f}  (n={len(malicious_scores)})")
    print(f"legitimate: mean={leg_mean:.4f}, std={leg_std:.4f}  (n={len(legit_scores)})")

    print("\nPaste this line into fusion_engine/generate_synthetic_dataset.py, "
          "replacing the existing \"email\": {...} entry in SCORE_DIST:\n")
    print(f'    "email": {{"malicious": ({mal_mean:.4f}, {mal_std:.4f}), '
          f'"legitimate": ({leg_mean:.4f}, {leg_std:.4f})}},')


if __name__ == "__main__":
    main()