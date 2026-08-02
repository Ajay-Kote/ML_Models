"""
text_branch/pretrain_on_real_corpus.py

Stage 1 of a two-stage training plan for the email text branch:

    Stage 1 (this script): fine-tune DistilBERT on a large, diverse REAL email corpus
    (ealvaradob/phishing-dataset, "texts" config -- ~18k phishing/legitimate emails,
    largely sourced from the Enron corpus + Kaggle phishing-emails dataset) so the model
    learns genuine phishing-language patterns instead of memorizing a handful of synthetic
    templates.

    Stage 2 (existing train.py): run as normal. get_text_branch() will find this
    checkpoint already sitting at text_branch/saved_model/distilbert/ and load it, then
    train.py's own .fit() call will further fine-tune it (a few epochs, low LR) on the
    small header-rich synthetic dataset, so it aligns with the metadata branch + fusion
    layer that DO need the synthetic RFC822 structure. This is a standard "pretrain big,
    fine-tune small" transfer-learning setup.

Why not just add the real corpus rows into train.py's main CSV directly? Because the
metadata branch (SPF/DKIM/DMARC, reply-to mismatch, attachment type, etc.) needs real
RFC822 headers, and this corpus is body-text only -- mixing it into the main pipeline
would silently zero out most metadata features for those rows and pollute that branch's
training distribution. Keeping the two stages separate avoids that.

Requires: `requests` and `pandas` (already installed as part of this module's requirements.txt
/ transitively via huggingface_hub) -- no extra packages needed. Downloads the raw
texts.json directly rather than going through the `datasets` library, since newer
`datasets` versions no longer support this repo's loading-script format.

Usage:
    python text_branch/pretrain_on_real_corpus.py
    python text_branch/pretrain_on_real_corpus.py --max_samples 3000 --epochs 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from text_branch.distilbert_branch import DistilBertTextBranch, DISTILBERT_MODEL_DIR


TEXTS_JSON_URL = "https://huggingface.co/datasets/ealvaradob/phishing-dataset/resolve/main/texts.json"


def load_corpus(max_samples: int, seed: int = 42):
    import json
    import requests
    import pandas as pd

    cache_path = Path(__file__).resolve().parent / "saved_model" / "_texts_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Using cached corpus file -> {cache_path}")
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        print(f"Downloading {TEXTS_JSON_URL} ...")
        resp = requests.get(TEXTS_JSON_URL, timeout=120)
        resp.raise_for_status()
        raw = resp.json()
        cache_path.write_text(json.dumps(raw), encoding="utf-8")
        print(f"Cached corpus -> {cache_path}")

    df = pd.DataFrame(raw)  # columns: text, label
    df = df.drop_duplicates(subset="text").dropna(subset=["text", "label"])
    df = df[df["text"].str.strip().str.len() > 20]  # drop near-empty rows

    if max_samples and len(df) > max_samples:
        df, _ = train_test_split(
            df, train_size=max_samples, random_state=seed, stratify=df["label"]
        )

    print(f"Loaded {len(df)} usable emails "
          f"({int(df['label'].sum())} phishing / {int((df['label'] == 0).sum())} legitimate)")
    return df["text"].tolist(), df["label"].tolist()


def main():
    parser = argparse.ArgumentParser(description="Stage 1: pretrain the email text branch on a real corpus.")
    parser.add_argument("--max_samples", type=int, default=3000,
                         help="Cap on training rows (CPU training time scales linearly with this).")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test_size", type=float, default=0.15)
    args = parser.parse_args()

    texts, labels = load_corpus(args.max_samples)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=args.test_size, random_state=42, stratify=labels
    )
    print(f"Train: {len(train_texts)}  |  Test: {len(test_texts)}")

    print(f"\nFine-tuning DistilBERT for {args.epochs} epoch(s), batch_size={args.batch_size}...")
    print("(CPU training -- this will take a while; progress is silent per-epoch, be patient)")
    branch = DistilBertTextBranch()
    branch.fit(train_texts, train_labels, epochs=args.epochs, batch_size=args.batch_size)

    print("\nEvaluating on held-out real-corpus test split...")
    test_proba = np.array([branch.predict_proba(t) for t in test_texts])
    test_pred = (test_proba >= 0.5).astype(int)

    print(f"  Accuracy : {accuracy_score(test_labels, test_pred):.4f}")
    print(f"  Precision: {precision_score(test_labels, test_pred, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(test_labels, test_pred, zero_division=0):.4f}")
    print(f"  F1       : {f1_score(test_labels, test_pred, zero_division=0):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(test_labels, test_proba):.4f}")

    DISTILBERT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    branch.save(DISTILBERT_MODEL_DIR)
    print(f"\nSaved Stage-1 checkpoint -> {DISTILBERT_MODEL_DIR}")
    print("Now run: python train.py   (Stage 2 -- will load this checkpoint and fine-tune")
    print("it further on the synthetic header-rich dataset, alongside the metadata + fusion layers)")


if __name__ == "__main__":
    main()