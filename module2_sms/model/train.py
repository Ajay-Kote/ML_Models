"""
train.py

Fine-tunes DistilBERT to classify SMS messages as smishing (1) or
legitimate (0).

What this script does, step by step:
    1. Load the train/val/test CSVs created by prepare_data.py
    2. Tokenize the text using DistilBERT's tokenizer
    3. Fine-tune a pretrained DistilBERT model on our data
    4. Evaluate it on the test set (accuracy, precision, recall, F1, ROC-AUC)
    5. Save the trained model to model/saved_model/

Requirements:
    Needs internet access to download the base "distilbert-base-uncased"
    model from Hugging Face the first time you run it.
    Works on CPU, but is much faster on a GPU (e.g. Google Colab, free tier).

Usage:
    python train.py
    python train.py --epochs 4 --batch_size 16
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL_NAME = "distilbert-base-uncased"
MAX_TOKEN_LENGTH = 96  # SMS messages are short, so this is plenty

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_model")


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------
class SMSDataset(Dataset):
    """Wraps tokenized SMS text + labels in the format Hugging Face's
    Trainer expects."""

    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=MAX_TOKEN_LENGTH,
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predicted_probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    predicted_labels = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted_labels, average="binary", zero_division=0
    )

    return {
        "accuracy": accuracy_score(labels, predicted_labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(labels, predicted_probs),
    }


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    args = parser.parse_args()

    print("Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    print("Loading tokenizer and base model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME, num_labels=2
    )

    train_dataset = SMSDataset(train_df["text"], train_df["label"], tokenizer)
    val_dataset = SMSDataset(val_df["text"], val_df["label"], tokenizer)
    test_dataset = SMSDataset(test_df["text"], test_df["label"], tokenizer)

    training_args = TrainingArguments(
        output_dir=os.path.join(os.path.dirname(__file__), "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to=[],  # disables wandb/tensorboard logging
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    trainer.train()

    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_dataset)
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    print(f"\nSaving model to {MODEL_SAVE_DIR}...")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)

    metrics_path = os.path.join(os.path.dirname(__file__), "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("Done. Model is ready for predict.py")


if __name__ == "__main__":
    main()
