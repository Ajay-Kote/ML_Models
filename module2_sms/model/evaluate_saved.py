import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)

model_path = "saved_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

test_df = pd.read_csv("../data/processed/test.csv")
texts = test_df["text"].tolist()   # column name apne test.csv ke hisaab se check kar
labels = test_df["label"].tolist()

all_preds, all_probs = [], []
with torch.no_grad():
    for i in range(0, len(texts), 32):
        batch = texts[i:i+32]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=1)[:, 1]
        preds = torch.argmax(out.logits, dim=1)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())

print(f"Accuracy : {accuracy_score(labels, all_preds):.4f}")
print(f"Precision: {precision_score(labels, all_preds):.4f}")
print(f"Recall   : {recall_score(labels, all_preds):.4f}")
print(f"F1       : {f1_score(labels, all_preds):.4f}")
print(f"ROC-AUC  : {roc_auc_score(labels, all_probs):.4f}")
print(f"Test set size: {len(labels)}")

# =====================================================
# Confusion Matrix (for paper / report)
# =====================================================

os.makedirs("../results", exist_ok=True)

cm = confusion_matrix(labels, all_preds)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.tight_layout()
plt.savefig("../results/confusion_matrix.png", dpi=300)
plt.close()

print("Saved -> ../results/confusion_matrix.png")