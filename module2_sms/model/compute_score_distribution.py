"""
compute_score_distribution.py  (module2_sms version)

Place this file inside module2_sms/model/ (same folder as predict.py)
and run:
    python compute_score_distribution.py

Uses the ALREADY-SAVED held-out test split (data/processed/test.csv),
so no re-splitting needed here -- these are the exact rows the model
never saw during training.
"""

import os

import pandas as pd

from predict import SmishingDetector

TEST_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "test.csv")


def main():
    test_df = pd.read_csv(TEST_CSV)
    detector = SmishingDetector()

    risk_scores = []
    true_labels = []

    total = len(test_df)
    for i, (_, row) in enumerate(test_df.iterrows(), start=1):
        if i == 1 or i % 100 == 0 or i == total:
            print(f"  ...{i}/{total} rows processed", flush=True)

        result = detector.predict(str(row["text"]))
        risk_scores.append(result["smishing_probability"])
        true_labels.append(int(row["label"]))

    df = pd.DataFrame({"risk_probability": risk_scores, "true_label": true_labels})

    # label=1 -> smishing (malicious), label=0 -> legitimate (confirmed from train.py)
    mal = df.loc[df["true_label"] == 1, "risk_probability"]
    leg = df.loc[df["true_label"] == 0, "risk_probability"]

    print(f"n_malicious={len(mal)}  n_legitimate={len(leg)}")
    print(f'"malicious":   (mean={mal.mean():.4f}, std={mal.std():.4f})')
    print(f'"legitimate":  (mean={leg.mean():.4f}, std={leg.std():.4f})')


if __name__ == "__main__":
    main()