import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

# ==============================
# Create results folder
# ==============================

os.makedirs("results", exist_ok=True)

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("data/processed/features.csv")

if "Label" in df.columns:
    label_col = "Label"
elif "label" in df.columns:
    label_col = "label"
else:
    raise Exception("No Label column found!")

X = df.drop(columns=[label_col])
y = df[label_col]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==============================
# Load Model
# ==============================

model = joblib.load("models/saved_model.pkl")

# ==============================
# Prediction
# ==============================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ==============================
# Metrics
# ==============================

print("=" * 60)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

auc = roc_auc_score(y_test, y_prob)
print("ROC AUC  :", auc)

print("=" * 60)

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# ==============================
# Confusion Matrix
# ==============================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Phishing", "Legitimate"]
)

disp.plot()

plt.title("Confusion Matrix")

plt.savefig("results/confusion_matrix.png")

plt.close()

# ==============================
# ROC Curve
# ==============================

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig("results/roc_curve.png")

plt.close()

# ==============================
# Feature Importance
# ==============================

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance.to_csv(
    "results/feature_importance.csv",
    index=False
)

plt.figure(figsize=(10,6))

plt.barh(
    importance["Feature"][:15],
    importance["Importance"][:15]
)

plt.gca().invert_yaxis()

plt.title("Top 15 Important Features")

plt.tight_layout()

plt.savefig("results/feature_importance.png")

plt.close()

print("\nResults saved successfully!")