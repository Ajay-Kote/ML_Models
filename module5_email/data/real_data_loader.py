"""
data/real_data_loader.py  (v2 - fixes dead metadata branch)

The merged phishing_email.csv only has {text_combined, label} -- no sender/subject
separation, so feature_extraction.py's From/Reply-To/Subject parsing always saw nothing
and every metadata feature (spf_pass, sender_domain_trusted, num_links, urgency_keyword_
count, ...) came out as a constant 0. That's why the metadata branch was stuck at ~52%
accuracy (pure guessing).

Fix: build the training data from the INDIVIDUAL source CSVs instead (CEAS_08.csv,
Nazario.csv, Enron.csv, Ling.csv, Nigerian_Fraud.csv, SpamAssasin.csv), which DO have
separate sender / subject / body columns. For each row we synthesize a minimal RFC822
email string:

    From: <sender>
    Subject: <subject>

    <body>

This is enough for feature_extraction.py's EXISTING .eml parser to recover real signals:
  - sender domain -> sender_domain_trusted / sender_domain_high_risk_tld
  - subject + body -> urgency_keyword_count / subject_has_urgency
  - URLs inside body text -> num_links (via the plain-text URL regex fallback)

SPF/DKIM/DMARC remain 0 for every row -- that's honest, since these corpora genuinely
don't carry authentication headers. Everything else that CAN be derived, now is.

Output: processed/emails.csv, same {raw_email, label} format train.py already expects.
No changes needed to train.py / predict.py / feature_extraction.py.

Usage:
    python real_data_loader.py                # uses all 6 source CSVs
    python real_data_loader.py --sample 6000   # balanced subsample
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

SOURCE_FILES = [
    "CEAS_08.csv",
    "Nazario.csv",
    "Enron.csv",
    "Ling.csv",
    "Nigerian_Fraud.csv",
    "SpamAssasin.csv",
]


def _safe_str(value, default: str = "") -> str:
    """Handles NaN (float) values that pandas leaves for missing CSV cells."""
    if value is None:
        return default
    if isinstance(value, float):
        return default  # NaN is a float; any other stray float is not valid text anyway
    return str(value)


def _build_pseudo_eml(sender, subject, body) -> str:
    sender = _safe_str(sender, "unknown@unknown.com").replace("\n", " ").strip()
    subject = _safe_str(subject, "").replace("\n", " ").strip()
    body = _safe_str(body, "")
    return f"From: {sender}\nSubject: {subject}\n\n{body}"


def load_all_sources(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for fname in SOURCE_FILES:
        fpath = raw_dir / fname
        if not fpath.exists():
            print(f"  (skipping {fname} -- not found)")
            continue
        df = pd.read_csv(fpath, usecols=lambda c: c.lower() in {"sender", "subject", "body", "label"})
        df.columns = [c.lower() for c in df.columns]
        missing = {"sender", "subject", "body", "label"} - set(df.columns)
        if missing:
            print(f"  (skipping {fname} -- missing columns {missing})")
            continue
        df = df.dropna(subset=["body", "label"])
        df["label"] = df["label"].astype(int)
        frames.append(df[["sender", "subject", "body", "label"]])
        print(f"  loaded {len(df)} rows from {fname}")
    if not frames:
        raise SystemExit("No usable source CSVs found in data/raw/.")
    return pd.concat(frames, ignore_index=True)


def main(sample: int | None, seed: int = 42):
    raw_dir = Path(__file__).parent / "raw"
    out_dir = Path(__file__).parent / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "emails.csv"

    print(f"Reading individual source CSVs from {raw_dir} ...")
    df = load_all_sources(raw_dir)

    if sample is not None and sample < len(df):
        n_per_class = sample // 2
        df_phish = df[df["label"] == 1].sample(n=min(n_per_class, (df["label"] == 1).sum()), random_state=seed)
        df_legit = df[df["label"] == 0].sample(n=min(n_per_class, (df["label"] == 0).sum()), random_state=seed)
        df = pd.concat([df_phish, df_legit], ignore_index=True)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    df["raw_email"] = df.apply(lambda r: _build_pseudo_eml(r["sender"], r["subject"], r["body"]), axis=1)
    out_df = df[["raw_email", "label"]]
    out_df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)

    n_total = len(out_df)
    n_phish = int(out_df["label"].sum())
    n_legit = n_total - n_phish
    print(f"Wrote {n_total} real emails ({n_phish} phishing / {n_legit} legitimate) to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=None,
                         help="Optional: subsample to N total rows (balanced across classes)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.sample, args.seed)