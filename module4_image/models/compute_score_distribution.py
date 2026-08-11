"""
compute_score_distribution.py  (module4_image version)

Place this file inside module4_image/models/ (same folder as predict.py)
and run:
    python compute_score_distribution.py

NOTE: This module's real predict() pipeline (OCR + EfficientNet embedding
+ LightGBM + SHAP) is heavy per-image, so this script:
  - skips heatmap generation (heatmap_out_path=None) since we only need
    the fraud_probability number, not the visual
  - defaults to a SAMPLE_SIZE so it finishes in reasonable time; set to
    None for the full held-out test split if you have time to spare

label=1 -> fraud (malicious), label=0 -> genuine (confirmed from train.py:
"{manifest['label'].sum()} fraud / ... genuine")
"""

import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.predict import PaymentImageDetector

THIS_DIR = os.path.dirname(__file__)
MANIFEST_PATH = os.path.join(THIS_DIR, "..", "data", "labels_all.csv")
DATASET_ROOT = os.path.join(THIS_DIR, "..", "data")
ARTIFACTS_DIR = os.path.join(THIS_DIR, "artifacts")  # <-- absolute-safe path, works regardless of cwd

SAMPLE_SIZE = 150  # set to None for the full ~20% held-out split


def resolve_image_path(filename: str) -> str:
    candidates = [
        os.path.join(DATASET_ROOT, filename),
        os.path.join(DATASET_ROOT, "real", filename),
        os.path.join(DATASET_ROOT, "fake", filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Could not resolve path for {filename}")


def main():
    manifest = pd.read_csv(MANIFEST_PATH)

    _, test_df = train_test_split(
        manifest, test_size=0.2, random_state=42, stratify=manifest["label"]
    )

    if SAMPLE_SIZE is not None and len(test_df) > SAMPLE_SIZE:
        test_df = test_df.sample(n=SAMPLE_SIZE, random_state=42)

    detector = PaymentImageDetector(artifacts_dir=ARTIFACTS_DIR)  # loads artifacts once

    risk_scores = []
    true_labels = []

    total = len(test_df)
    for i, (_, row) in enumerate(test_df.iterrows(), start=1):
        if i == 1 or i % 20 == 0 or i == total:
            print(f"  ...{i}/{total} rows processed", flush=True)

        img_path = resolve_image_path(row["filename"])
        result = detector.predict(img_path, heatmap_out_path=None)  # skip heatmap for speed
        risk_scores.append(result["fraud_probability"])
        true_labels.append(int(row["label"]))

    df = pd.DataFrame({"risk_probability": risk_scores, "true_label": true_labels})

    mal = df.loc[df["true_label"] == 1, "risk_probability"]  # 1 = fraud
    leg = df.loc[df["true_label"] == 0, "risk_probability"]  # 0 = genuine

    print(f"n_malicious={len(mal)}  n_legitimate={len(leg)}")
    print(f'"malicious":   (mean={mal.mean():.4f}, std={mal.std():.4f})')
    print(f'"legitimate":  (mean={leg.mean():.4f}, std={leg.std():.4f})')


if __name__ == "__main__":
    main()