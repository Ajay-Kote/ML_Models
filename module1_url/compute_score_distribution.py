"""
compute_score_distribution.py

Run this ONCE PER MODULE (with the module-specific block un-commented /
adapted) to get the (mean, std) numbers that go into SCORE_DIST inside
generate_synthetic_dataset.py.

Pattern (same for every module):
  1. Load the SAME raw dataset + SAME train_test_split(random_state=42)
     that was used during training, so you get the exact held-out test
     rows the model has never seen.
  2. Run predict() on every row of that test split.
  3. Split the resulting risk_probability values into two groups using
     the TRUE label (malicious vs legitimate).
  4. Print mean + std for each group -> paste into SCORE_DIST.

Below is a filled-in example for module1_url (based on its existing
test_model.py). Copy this file into each module's own folder and swap
the marked lines for that module's data path / predict function / label
column.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --- MODULE-SPECIFIC: change these 3 lines per module -----------------
from models.predict import predict_url as predict_fn      # <-- swap import
DATA_PATH = "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"   # <-- swap path
LABEL_COL = "label"                                         # <-- swap col name
# ------------------------------------------------------------------------


def main():
    raw = pd.read_csv(DATA_PATH)
    label_col = LABEL_COL if LABEL_COL in raw.columns else LABEL_COL.capitalize()

    # MUST match the exact split used at training time (same test_size +
    # random_state) or you'll leak train rows into this "test" evaluation.
    _, test_df = train_test_split(
        raw, test_size=0.20, random_state=42, stratify=raw[label_col]
    )

    # Quick sanity-check run first with a small sample. Set to None to
    # use the FULL held-out test split (slower, but gives the most
    # accurate mean/std numbers).
    SAMPLE_SIZE = None
    if SAMPLE_SIZE is not None and len(test_df) > SAMPLE_SIZE:
        test_df = test_df.sample(n=SAMPLE_SIZE, random_state=42)

    risk_scores = []
    true_labels = []

    total = len(test_df)
    for i, (_, row) in enumerate(test_df.iterrows(), start=1):
        if i == 1 or i % 200 == 0 or i == total:
            print(f"  ...{i}/{total} rows processed", flush=True)

        # --- MODULE-SPECIFIC: how you call predict + extract risk_probability ---
        result = predict_fn(row["URL"])
        risk = result["Phishing Probability"] / 100.0   # normalize to 0-1
        # -------------------------------------------------------------------------

        risk_scores.append(risk)
        true_labels.append(int(row[label_col]))

    df = pd.DataFrame({"risk_probability": risk_scores, "true_label": true_labels})

    # NOTE: adjust which numeric value means "malicious" for this module.
    # (module1_url's raw label: 1 = legitimate, 0 = phishing -- so malicious == 0 here.
    #  Check each module's own convention before copying this blindly!)
    malicious_mask = df["true_label"] == 0
    legit_mask = df["true_label"] == 1

    mal = df.loc[malicious_mask, "risk_probability"]
    leg = df.loc[legit_mask, "risk_probability"]

    print(f"n_malicious={len(mal)}  n_legitimate={len(leg)}")
    print(f'"malicious":   (mean={mal.mean():.4f}, std={mal.std():.4f})')
    print(f'"legitimate":  (mean={leg.mean():.4f}, std={leg.std():.4f})')


if __name__ == "__main__":
    main()