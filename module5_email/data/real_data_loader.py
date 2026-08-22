"""
data/real_data_loader.py

Replaces generate_synthetic_data.py with the REAL Kaggle "Phishing Email
Dataset" (Naser Abdullah Alam) -- an aggregation of Nazario, SpamAssassin,
Enron, CEAS, Ling, and Nigerian Fraud corpora.

Source file used: raw/phishing_email.csv
  Columns: text_combined, label   (label: 0 = legitimate, 1 = phishing)

Output: processed/emails.csv
  Columns: raw_email, label   <-- SAME format generate_synthetic_data.py
                                   produced, so train.py / predict.py need
                                   NO changes downstream.

Usage:
    python real_data_loader.py
    python real_data_loader.py --sample 6000   # optional: subsample
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def main(sample: int | None, seed: int = 42):
    raw_path = Path(__file__).parent / "raw" / "phishing_email.csv"
    out_dir = Path(__file__).parent / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "emails.csv"

    print(f"Reading {raw_path} ...")
    df = pd.read_csv(raw_path)

    # normalize column names -> raw_email, label
    df = df.rename(columns={"text_combined": "raw_email"})
    df = df[["raw_email", "label"]].dropna()
    df["label"] = df["label"].astype(int)

    if sample is not None and sample < len(df):
        n_per_class = sample // 2
        df_phish = df[df["label"] == 1].sample(n=min(n_per_class, (df["label"] == 1).sum()), random_state=seed)
        df_legit = df[df["label"] == 0].sample(n=min(n_per_class, (df["label"] == 0).sum()), random_state=seed)
        df = pd.concat([df_phish, df_legit], ignore_index=True)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)

    n_total = len(df)
    n_phish = int(df["label"].sum())
    n_legit = n_total - n_phish
    print(f"Wrote {n_total} real emails ({n_phish} phishing / {n_legit} legitimate) to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=None,
                         help="Optional: subsample to N total rows (balanced across classes)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.sample, args.seed)