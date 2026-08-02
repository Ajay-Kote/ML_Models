"""
download_data.py

Downloads the SMS dataset used to train the smishing detector.

Dataset: UCI SMS Spam Collection (5,574 real SMS messages, labeled spam/ham)
Mirror used: a public GitHub copy of the same UCI dataset, as a clean TSV file.

This is a general spam-vs-ham dataset, not a phishing-only ("smishing")
dataset — there isn't one single clean public source for that. It's still a
good starting point because a lot of smishing patterns (urgency, fake
prizes, shortened links) overlap with spam patterns. Later, for a stronger
final model, you can add a phishing-specific dataset on top of this one —
see the note at the bottom of this file.

Run this first, before prepare_data.py.

Usage:
    python download_data.py
"""

import os
import urllib.request

DATASET_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
SAVE_PATH = os.path.join(os.path.dirname(__file__), "raw", "sms.tsv")


def download_dataset():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    print(f"Downloading dataset from:\n  {DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, SAVE_PATH)
    print(f"Saved to: {SAVE_PATH}")

    # Quick sanity check on what we downloaded
    with open(SAVE_PATH, encoding="utf-8") as f:
        lines = f.readlines()

    spam_count = sum(1 for line in lines if line.startswith("spam"))
    ham_count = sum(1 for line in lines if line.startswith("ham"))

    print(f"\nTotal messages : {len(lines)}")
    print(f"  spam (label=1): {spam_count}")
    print(f"  ham  (label=0): {ham_count}")


if __name__ == "__main__":
    download_dataset()


# ---------------------------------------------------------------------------
# OPTIONAL: adding a phishing-specific dataset later
# ---------------------------------------------------------------------------
# For a stronger final model, download a smishing-specific dataset and drop
# it into the raw/ folder next to sms.tsv, for example:
#
#   Kaggle: "SMS Phishing Dataset for Machine Learning and Pattern Recognition"
#   https://www.kaggle.com/datasets/galactus007/sms-phishing-dataset-for-machine-learning-and-pattern-recognition
#
# Make sure the file has two columns: label (spam/ham or 1/0) and text.
# Then update prepare_data.py to also read that file and merge it with
# sms.tsv before splitting into train/val/test.
