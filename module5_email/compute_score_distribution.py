"""
compute_score_distribution.py  (module5_email version)

Place this file inside module5_email/ (same folder as predict.py, train.py)
and run:
    python compute_score_distribution.py

Dataset is small (120 rows total, 60 phishing / 60 legitimate), so this
should finish fast even on CPU. Replicates train.py's exact split
(test_size=0.25, random_state=42, stratified by label).

label=1 -> phishing (malicious), label=0 -> legitimate (confirmed from
emails.csv value_counts + train.py stratify usage).
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from predict import predict as predict_fn

CSV_PATH = "data/processed/emails.csv"


def main():
    df = pd.read_csv(CSV_PATH)

    _, test_df = train_test_split(
        df, test_size=0.25, random_state=42, stratify=df["label"]
    )

    risk_scores = []
    true_labels = []

    total = len(test_df)
    for i, (_, row) in enumerate(test_df.iterrows(), start=1):
        print(f"  ...{i}/{total} rows processed", flush=True)

        result = predict_fn(row["raw_email"])
        risk_scores.append(result["phishing_probability"])
        true_labels.append(int(row["label"]))

    out = pd.DataFrame({"risk_probability": risk_scores, "true_label": true_labels})

    mal = out.loc[out["true_label"] == 1, "risk_probability"]  # 1 = phishing
    leg = out.loc[out["true_label"] == 0, "risk_probability"]  # 0 = legitimate

    print(f"n_malicious={len(mal)}  n_legitimate={len(leg)}")
    print(f'"malicious":   (mean={mal.mean():.4f}, std={mal.std():.4f})')
    print(f'"legitimate":  (mean={leg.mean():.4f}, std={leg.std():.4f})')


if __name__ == "__main__":
    main()