"""
compute_score_distribution.py  (module3_qr version)

Place this file inside module3_qr/models/ (same folder as predict.py)
and run:
    python compute_score_distribution.py

Replicates the EXACT same manifest-order + train_test_split(random_state=42)
used in train.py, so we evaluate on the same held-out rows the model
never trained on. Then calls predict() (which takes an image PATH, not a
feature row) on each held-out image.
"""

import json
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from predict import predict as predict_fn

THIS_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(THIS_DIR, "..", "data", "raw")
IMG_DIR = os.path.join(RAW_DIR, "images")
MANIFEST_PATH = os.path.join(RAW_DIR, "metadata.jsonl")


def load_manifest():
    rows = []
    with open(MANIFEST_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    rows = load_manifest()
    labels = np.array([r["label"] for r in rows], dtype=np.int32)

    # Same split as train.py: stratified 80/20, random_state=42, applied
    # to row INDICES so we can recover which manifest rows are "test".
    indices = np.arange(len(rows))
    _, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )
    test_rows = [rows[i] for i in test_idx]

    risk_scores = []
    true_labels = []

    total = len(test_rows)
    for i, row in enumerate(test_rows, start=1):
        if i == 1 or i % 100 == 0 or i == total:
            print(f"  ...{i}/{total} rows processed", flush=True)

        img_path = os.path.join(IMG_DIR, row["file"])
        result = predict_fn(img_path)
        risk_scores.append(result["malicious_qr_probability"])
        true_labels.append(int(row["label"]))

    risk_scores = np.array(risk_scores)
    true_labels = np.array(true_labels)

    # label=1 -> malicious, label=0 -> legitimate (confirmed from metadata.jsonl)
    mal = risk_scores[true_labels == 1]
    leg = risk_scores[true_labels == 0]

    print(f"n_malicious={len(mal)}  n_legitimate={len(leg)}")
    print(f'"malicious":   (mean={mal.mean():.4f}, std={mal.std():.4f})')
    print(f'"legitimate":  (mean={leg.mean():.4f}, std={leg.std():.4f})')


if __name__ == "__main__":
    main()