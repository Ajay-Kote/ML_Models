"""
train_meta_learner.py

Trains a Logistic Regression meta-learner that takes the 5 modules'
risk_probability + presence flags and outputs one combined fraud
probability -- this REPLACES the fixed-weight average in fuse.py.

Run:
    python train_meta_learner.py

Produces:
    meta_learner.joblib   -- trained sklearn Pipeline (scaler + LR)
    Prints test-set metrics + learned coefficients (per-module importance)
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from generate_synthetic_dataset import generate, MODULES

FEATURE_COLS = [f"{m}_risk" for m in MODULES] + [f"{m}_present" for m in MODULES]


def train():
    df = generate(n_rows=5000)
    X = df[FEATURE_COLS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    print("=== Test-set metrics ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1       : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

    print("\n=== Learned coefficients (higher = module weighted more) ===")
    coefs = pipe.named_steps["clf"].coef_[0]
    for name, coef in sorted(zip(FEATURE_COLS, coefs), key=lambda t: -abs(t[1])):
        print(f"  {name:15s} {coef:+.4f}")

    joblib.dump(pipe, "meta_learner.joblib")
    print("\nSaved -> meta_learner.joblib")
    return pipe


if __name__ == "__main__":
    train()
