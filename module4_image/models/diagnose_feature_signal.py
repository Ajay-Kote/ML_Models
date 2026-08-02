"""
Diagnostic: which feature group is actually carrying signal?

Uses the feature_cache.joblib from a previous `models.train` run (so no OCR
re-run needed) to train 3 quick LightGBM models on:
  1. OCR-only structured/quality features
  2. Visual-only PCA-reduced embedding features
  3. Combined (what models/train.py normally uses)

and reports ROC-AUC for each on the same held-out test split, so you can see
which modality (if either) is actually contributing to fraud detection.

Usage (run from the project root, after a normal `models.train` run so the
cache exists):
    python -m models.diagnose_feature_signal --manifest data/labels_all.csv --out_dir models/artifacts
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from models.train import normalize_manifest
from fusion.feature_fusion import structured_features_to_row
from ocr.paddleocr_pipeline import OCRResult, OCRStructuredFields, OCRQualitySignals
from vision.ela_features import compute_ela_features, ela_features_to_row


def load_cached_rows(manifest: pd.DataFrame, cache: dict):
    ocr_rows, ela_rows, visual_embeddings, labels, kept = [], [], [], [], []
    for _, row in manifest.iterrows():
        path = row["image_path"]
        entry = cache.get(path)
        if not entry or "embedding" not in entry or "structured" not in entry:
            continue
        fake_ocr_result = OCRResult(
            raw_text=entry.get("raw_text", ""),
            structured=OCRStructuredFields(**entry["structured"]),
            quality=OCRQualitySignals(**entry["quality"]),
        )
        ocr_rows.append(structured_features_to_row(fake_ocr_result))
        ela_rows.append(ela_features_to_row(compute_ela_features(path)))  # fast, not cached
        visual_embeddings.append(np.asarray(entry["embedding"]))
        labels.append(row["label"])
        kept.append(path)
    return pd.DataFrame(ocr_rows), pd.DataFrame(ela_rows), np.stack(visual_embeddings), np.array(labels), kept


def eval_auc(X: pd.DataFrame, y: np.ndarray, label: str):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = lgb.LGBMClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, verbosity=-1
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"{label:20s} AUC = {auc:.4f}  (n_features={X.shape[1]}, n_train={len(X_train)}, n_test={len(X_test)})")
    return auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/labels_all.csv")
    parser.add_argument("--out_dir", default="models/artifacts")
    args = parser.parse_args()

    cache_path = os.path.join(args.out_dir, "feature_cache.joblib")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"{cache_path} not found -- run `python -m models.train` at least once first "
            "so OCR/embedding results are cached."
        )
    cache = joblib.load(cache_path)
    print(f"Loaded cache with {len(cache)} entries from {cache_path}")

    manifest = normalize_manifest(pd.read_csv(args.manifest), args.manifest)
    ocr_df, ela_df, visual_embeddings, labels, kept = load_cached_rows(manifest, cache)
    print(f"{len(kept)}/{len(manifest)} images had cached OCR+embedding results\n")

    # Visual-only: PCA-reduce raw 1280-d embeddings to 32 dims, same as training
    pca = PCA(n_components=32, random_state=42)
    visual_reduced = pca.fit_transform(visual_embeddings)
    visual_df = pd.DataFrame(visual_reduced, columns=[f"visual_pc_{i}" for i in range(visual_reduced.shape[1])])

    ocr_df = ocr_df.reset_index(drop=True)
    ela_df = ela_df.reset_index(drop=True)
    visual_df = visual_df.reset_index(drop=True)

    ocr_ela_df = pd.concat([ocr_df, ela_df], axis=1)
    full_combined_df = pd.concat([ocr_df, ela_df, visual_df], axis=1)

    print("=== Feature-group ablation (ROC-AUC on held-out test, higher = more signal) ===")
    eval_auc(ocr_df, labels, "OCR-only")
    eval_auc(ela_df, labels, "ELA-only")
    eval_auc(visual_df, labels, "Visual (EfficientNet)-only")
    eval_auc(ocr_ela_df, labels, "OCR + ELA")
    eval_auc(full_combined_df, labels, "Full combined (OCR+ELA+Visual)")
    print("\n0.50 = no signal (random). 1.00 = perfect separation.")


if __name__ == "__main__":
    main()
