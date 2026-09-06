"""
compute_score_distribution.py
Computes (mean, std) of risk_probability for malicious vs legitimate URLs,
for use in the fusion engine's SCORE_DIST.

FAST VERSION: uses the already-extracted data/processed/features.csv
(built by build_features.py, which already did all the network lookups
once) instead of re-running live DNS/SSL/WHOIS lookups per row. This
takes seconds instead of hours.

Uses the EXACT same train_test_split (test_size=0.20, random_state=42,
stratify=Label) as models/train.py, so this evaluates on the same
held-out test rows the model was tested on - not on training rows.
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES_PATH = "data/processed/features.csv"
MODEL_PATH = "models/saved_model.pkl"


def main():
    df = pd.read_csv(FEATURES_PATH)

    label_col = "Label" if "Label" in df.columns else "label"

    X = df.drop(columns=[label_col])
    y = df[label_col]

    # Same split as train.py - this reproduces the same held-out test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = joblib.load(MODEL_PATH)

    try:
        train_columns = model.feature_name_
        X_test = X_test.reindex(columns=train_columns, fill_value=0)
    except AttributeError:
        pass

    # Phishing probability = P(class 0), since Label: 0 = phishing, 1 = legitimate
    proba = model.predict_proba(X_test)
    class_index_0 = list(model.classes_).index(0)
    risk_scores = proba[:, class_index_0]

    result_df = pd.DataFrame({
        "risk_probability": risk_scores,
        "true_label": y_test.values,
    })

    malicious_mask = result_df["true_label"] == 0
    legit_mask = result_df["true_label"] == 1

    mal = result_df.loc[malicious_mask, "risk_probability"]
    leg = result_df.loc[legit_mask, "risk_probability"]

    print(f"n_malicious={len(mal)}  n_legitimate={len(leg)}")
    print(f'"malicious":   (mean={mal.mean():.4f}, std={mal.std():.4f})')
    print(f'"legitimate":  (mean={leg.mean():.4f}, std={leg.std():.4f})')


if __name__ == "__main__":
    main()