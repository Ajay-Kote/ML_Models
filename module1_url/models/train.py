import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# =====================================================
# Load Dataset
# =====================================================

print("Loading Features...\n")

df = pd.read_csv("data/processed/features.csv")

print("Dataset Shape :", df.shape)
print()

# -----------------------------------------------------
# Detect Label Column Automatically
# -----------------------------------------------------

if "Label" in df.columns:
    label_col = "Label"
elif "label" in df.columns:
    label_col = "label"
else:
    raise Exception("No Label column found!")

# =====================================================
# Features and Target
# =====================================================

X = df.drop(columns=[label_col])
y = df[label_col]

print("Number of Features :", X.shape[1])
print("Samples            :", X.shape[0])
print()

# =====================================================
# Train/Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================================
# Model
# =====================================================

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    random_state=42,
    n_jobs=-1
)

print("Training LightGBM...\n")

model.fit(X_train, y_train)

# =====================================================
# Prediction
# =====================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# =====================================================
# Metrics
# =====================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc:.4f}")

print("=" * 50)

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# =====================================================
# Save Model
# =====================================================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/saved_model.pkl")

print("\nModel Saved -> models/saved_model.pkl")

# =====================================================
# Confusion Matrix
# =====================================================

os.makedirs("results", exist_ok=True)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.close()

print("Saved -> results/confusion_matrix.png")

# =====================================================
# Feature Importance
# =====================================================

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

plt.figure(figsize=(10,8))

plt.barh(
    importance["Feature"][:20],
    importance["Importance"][:20]
)

plt.gca().invert_yaxis()

plt.title("Top 20 Feature Importance")

plt.tight_layout()

plt.savefig(
    "results/feature_importance.png",
    dpi=300
)

plt.close()

print("Saved -> results/feature_importance.png")

print("\nTraining Completed Successfully!")
print(df[label_col].value_counts())
print(df.groupby(label_col)["High_Risk_TLD"].mean())
print(df.groupby(label_col)["Brand_Count"].mean())
print(df.groupby(label_col)["Suspicious_Keyword_Count"].mean())