"""
prepare_data.py

Cleans the raw SMS data and splits it into train / validation / test sets
ready for training.

Steps:
    1. Load raw/sms.tsv
    2. Remove duplicate messages
    3. Convert labels to numbers: ham -> 0, spam -> 1
    4. Split into train (70%) / validation (15%) / test (15%)
    5. Save each split as a CSV in processed/

Run this after download_data.py.

Usage:
    python prepare_data.py
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = os.path.join(os.path.dirname(__file__), "raw", "sms.tsv")
EXTRA_EXAMPLES_PATH = os.path.join(os.path.dirname(__file__), "raw", "smishing_examples.csv")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")


def load_raw_data():
    df = pd.read_csv(RAW_PATH, sep="\t", header=None, names=["label_text", "text"])
    df["text"] = df["text"].astype(str).str.strip()

    # The base UCI dataset is old-style spam (2005-era UK telecom spam) and
    # is missing modern smishing patterns (fake bank alerts, KYC scams,
    # delivery fee scams, etc). If a hand-written examples file exists,
    # merge it in so the model actually learns those patterns too.
    if os.path.exists(EXTRA_EXAMPLES_PATH):
        extra_df = pd.read_csv(EXTRA_EXAMPLES_PATH)
        extra_df["text"] = extra_df["text"].astype(str).str.strip()
        extra_df = extra_df.rename(columns={"label": "label_text"})
        df = pd.concat([df, extra_df], ignore_index=True)
        print(f"Merged in {len(extra_df)} extra hand-written examples from {EXTRA_EXAMPLES_PATH}")

    return df


def clean_data(df):
    # Remove exact duplicate messages
    df = df.drop_duplicates(subset="text").reset_index(drop=True)

    # Convert text labels to numbers: 0 = legitimate, 1 = spam/smishing
    df["label"] = (df["label_text"] == "spam").astype(int)

    return df[["text", "label"]]


def split_data(df):
    # First split off the test set (15%)
    train_val_df, test_df = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=42
    )
    # Then split the remainder into train (70% of total) and val (15% of total)
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.1765, stratify=train_val_df["label"], random_state=42
    )  # 0.1765 * 0.85 ≈ 0.15 of the original total
    return train_df, val_df, test_df


def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"{RAW_PATH} not found. Run download_data.py first."
        )

    df = load_raw_data()
    df = clean_data(df)
    train_df, val_df, test_df = split_data(df)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

    print(f"Total cleaned messages: {len(df)}")
    print(f"  train: {len(train_df)}")
    print(f"  val  : {len(val_df)}")
    print(f"  test : {len(test_df)}")
    print(f"\nSaved to: {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()
