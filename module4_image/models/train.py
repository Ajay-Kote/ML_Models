"""
Module 4 - Payment Image Detection
Training script: builds fused features for every labeled screenshot,
fits the PCA visual-dim reducer, trains a LightGBM classifier, evaluates,
and saves all artifacts needed for inference (models/predict.py).

Performance features:
  - Parallel OCR extraction (ThreadPoolExecutor, one PaddleOCR instance per
    worker thread) -- configurable via --workers.
  - Batched GPU visual-embedding extraction (EfficientNet-B0 processes many
    images per forward pass instead of one at a time) -- configurable via
    --embed_batch_size.
  - Automatic CPU/GPU detection for both OCR (paddlepaddle-gpu) and the
    visual embedder (torch.cuda). Falls back to CPU cleanly if no GPU /
    no GPU-enabled paddle build is found.
  - Per-image OCR+embedding caching so repeated runs (e.g. tuning LightGBM
    hyperparameters) don't redo the expensive part.
  - tqdm progress bars.
  - LightGBM trained with n_jobs=-1 (all CPU cores); optionally tries
    device_type="gpu" for LightGBM itself if --lgbm_gpu is passed (requires
    a GPU-built LightGBM -- the default pip wheel is CPU-only, so this is
    off by default and falls back to CPU automatically if it fails).

Expected input: a CSV manifest with columns:
    image_path, label   (label: 1 = fraudulent/forged, 0 = genuine)

Usage:
    python -m models.train --manifest data/labels_all.csv --out_dir models/artifacts
    python -m models.train --manifest data/labels_all.csv --out_dir models/artifacts --workers 8 --embed_batch_size 64
"""

import argparse
import os
import threading
from dataclasses import asdict
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from fusion.feature_fusion import VisualPCAReducer, build_feature_vector
from ocr.paddleocr_pipeline import PaymentOCRPipeline, OCRResult, OCRStructuredFields, OCRQualitySignals
from vision.efficientnet_extractor import VisualEmbeddingService


def normalize_manifest(manifest: pd.DataFrame, manifest_path: "str | os.PathLike") -> pd.DataFrame:
    """Support both image_path-based manifests and the workspace's filename-based labels CSV."""
    manifest = manifest.copy()
    manifest_path = Path(manifest_path)
    dataset_root = manifest_path.parent

    def resolve_image_path(name: str) -> str:
        if not name:
            return ""
        if os.path.isabs(name):
            return str(name)

        candidates = [
            dataset_root / name,
            dataset_root / "real" / name,
            dataset_root / "fake" / name,
            dataset_root / "data" / name,
            dataset_root / "data" / "real" / name,
            dataset_root / "data" / "fake" / name,
        ]

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        for match in dataset_root.rglob(name):
            if match.is_file():
                return str(match)

        return str(dataset_root / name)

    if "image_path" not in manifest.columns:
        if "filename" in manifest.columns:
            manifest["image_path"] = manifest["filename"].apply(resolve_image_path)
        elif "image" in manifest.columns:
            manifest["image_path"] = manifest["image"].apply(resolve_image_path)
        else:
            raise ValueError("Manifest must contain 'image_path', 'filename', or 'image' column")

    if "label" not in manifest.columns:
        if "target" in manifest.columns:
            manifest["label"] = manifest["target"]
        else:
            raise ValueError("Manifest must contain a 'label' column")

    manifest["image_path"] = manifest["image_path"].astype(str)
    manifest["label"] = manifest["label"].astype(int)

    # Keep only rows whose images actually exist to avoid failing mid-run on a bad manifest.
    manifest = manifest[manifest["image_path"].apply(lambda p: os.path.exists(p))].reset_index(drop=True)
    return manifest


# ---------------------------------------------------------------------------
# Parallel OCR extraction
# ---------------------------------------------------------------------------
_thread_local = threading.local()


def _get_thread_ocr_pipeline(ocr_device: str) -> PaymentOCRPipeline:
    """Each worker thread gets its own PaddleOCR instance (not safely
    shareable for concurrent .predict() calls across threads)."""
    if not hasattr(_thread_local, "ocr_pipeline"):
        _thread_local.ocr_pipeline = PaymentOCRPipeline(device=ocr_device)
    return _thread_local.ocr_pipeline


def _ocr_worker(path: str, ocr_device: str):
    try:
        pipeline = _get_thread_ocr_pipeline(ocr_device)
        return path, pipeline.extract(path), None
    except Exception as e:
        return path, None, str(e)


def extract_ocr_parallel(paths: list, ocr_device: str, workers: int, cache: dict, use_cache: bool):
    """Runs OCR extraction across a thread pool. Returns {path: OCRResult}
    for every path that succeeded (from cache or freshly computed)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    to_process = []

    for path in paths:
        if use_cache and path in cache and "structured" in cache[path]:
            entry = cache[path]
            structured = OCRStructuredFields(**entry["structured"]) if entry.get("structured") else OCRStructuredFields()
            quality = OCRQualitySignals(**entry["quality"]) if entry.get("quality") else OCRQualitySignals()
            results[path] = OCRResult(raw_text=entry.get("raw_text", ""), structured=structured, quality=quality)
        else:
            to_process.append(path)

    if not to_process:
        return results

    print(f"Running OCR on {len(to_process)} images ({len(paths) - len(to_process)} from cache) "
          f"using {workers} worker thread(s)...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_ocr_worker, p, ocr_device): p for p in to_process}
        for future in tqdm(as_completed(futures), total=len(futures), desc="OCR", unit="img"):
            path, ocr_result, error = future.result()
            if error:
                print(f"[skip OCR] {path}: {error}")
                continue
            results[path] = ocr_result
            if use_cache:
                cache.setdefault(path, {})
                cache[path]["raw_text"] = ocr_result.raw_text
                cache[path]["structured"] = asdict(ocr_result.structured)
                cache[path]["quality"] = asdict(ocr_result.quality)

    return results


# ---------------------------------------------------------------------------
# Batched visual embedding extraction (GPU-friendly)
# ---------------------------------------------------------------------------
def extract_embeddings_batched(paths: list, visual_service: VisualEmbeddingService,
                                batch_size: int, cache: dict, use_cache: bool):
    """Returns {path: embedding} using batched forward passes for anything
    not already cached."""
    results = {}
    to_process = []

    for path in paths:
        if use_cache and path in cache and "embedding" in cache[path]:
            results[path] = np.asarray(cache[path]["embedding"])
        else:
            to_process.append(path)

    if not to_process:
        return results

    print(f"Extracting visual embeddings for {len(to_process)} images "
          f"({len(paths) - len(to_process)} from cache), batch_size={batch_size}...")

    for start in tqdm(range(0, len(to_process), batch_size), desc="Embeddings", unit="batch"):
        chunk = to_process[start:start + batch_size]
        batch_results = visual_service.embed_batch(chunk, batch_size=batch_size)
        for path, emb in batch_results.items():
            results[path] = emb
            if use_cache:
                cache.setdefault(path, {})
                cache[path]["embedding"] = emb

    return results


def extract_raw_features(manifest: pd.DataFrame, ocr_device: str, visual_service: VisualEmbeddingService,
                          cache_path: str = None, use_cache: bool = True,
                          workers: int = 4, embed_batch_size: int = 32):
    """Combines parallel OCR + batched embeddings, matched back to the manifest."""
    cache = {}
    if cache_path and use_cache and os.path.exists(cache_path):
        try:
            cache = joblib.load(cache_path)
            print(f"Loaded feature cache with {len(cache)} entries from {cache_path}")
        except Exception:
            cache = {}

    paths = manifest["image_path"].tolist()

    ocr_by_path = extract_ocr_parallel(paths, ocr_device, workers, cache, use_cache)
    if cache_path and use_cache:
        joblib.dump(cache, cache_path)  # checkpoint after OCR pass

    embed_by_path = extract_embeddings_batched(paths, visual_service, embed_batch_size, cache, use_cache)
    if cache_path and use_cache:
        joblib.dump(cache, cache_path)  # checkpoint after embedding pass

    ocr_results, embeddings, labels, kept_paths = [], [], [], []
    for _, row in manifest.iterrows():
        path = row["image_path"]
        if path not in ocr_by_path or path not in embed_by_path:
            continue  # skipped during OCR or embedding due to an error
        ocr_results.append(ocr_by_path[path])
        embeddings.append(embed_by_path[path])
        labels.append(row["label"])
        kept_paths.append(path)

    return ocr_results, np.stack(embeddings), np.array(labels), kept_paths


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = normalize_manifest(pd.read_csv(args.manifest), args.manifest)
    print(f"Loaded manifest with {len(manifest)} rows "
          f"({manifest['label'].sum()} fraud / {len(manifest) - manifest['label'].sum()} genuine)")

    # Optionally subsample for quick/debug runs
    if args.sample_frac is not None and 0 < args.sample_frac < 1.0:
        manifest = manifest.sample(frac=args.sample_frac, random_state=42).reset_index(drop=True)
        print(f"Subsampled to {len(manifest)} rows (--sample_frac {args.sample_frac})")

    visual_service = VisualEmbeddingService()  # auto-detects CUDA internally

    cache_path = os.path.join(args.out_dir, "feature_cache.joblib")
    use_cache = not args.no_cache

    ocr_results, embeddings, labels, paths = extract_raw_features(
        manifest, ocr_device=args.ocr_device, visual_service=visual_service,
        cache_path=cache_path, use_cache=use_cache,
        workers=args.workers, embed_batch_size=args.embed_batch_size,
    )
    print(f"Feature extraction complete: {len(paths)}/{len(manifest)} images usable.")

    if embeddings.shape[0] < 2 or embeddings.shape[1] < 1:
        raise ValueError("Not enough valid samples were extracted from the manifest")

    n_components = min(args.pca_components, embeddings.shape[0] - 1, embeddings.shape[1])
    if n_components < 1:
        n_components = 1
    pca_reducer = VisualPCAReducer(n_components=n_components).fit(embeddings)

    rows = [
        build_feature_vector(ocr_res, emb, pca_reducer, image_path=path)
        for ocr_res, emb, path in zip(ocr_results, embeddings, paths)
    ]
    X = pd.DataFrame(rows).fillna(-1)
    y = labels

    if len(np.unique(y)) < 2:
        raise ValueError("Training requires both positive and negative labels in the manifest")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Separate validation split carved out of the *training* portion only,
    # used purely for early stopping. X_test/y_test are never touched until
    # final evaluation below -- using the test set itself for early stopping
    # (as an earlier version of this script did) lets the model implicitly
    # "see" the test set during training and biases the reported metrics.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
    )

    print(f"Training LightGBM on {X_train.shape[0]} samples "
          f"({X_val.shape[0]} held out for early-stopping validation, "
          f"{X_test.shape[0]} held out for final test), {X_train.shape[1]} features...")

    lgbm_kwargs = dict(
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,  # use all CPU cores for tree building regardless of GPU availability
        class_weight="balanced" if args.balance_classes else None,
    )

    clf = None
    if args.lgbm_gpu:
        try:
            clf = lgb.LGBMClassifier(device_type="gpu", **lgbm_kwargs)
            clf.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
            )
            print("Trained LightGBM on GPU.")
        except Exception as e:
            print(f"[lgbm_gpu failed, falling back to CPU] {e}")
            clf = None

    if clf is None:
        clf = lgb.LGBMClassifier(**lgbm_kwargs)
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

    # --- Probability calibration ---
    # class_weight="balanced" on this imbalanced dataset (and the now-harder,
    # more realistic forgeries) tends to compress raw LightGBM probabilities
    # toward 0.5 even for correctly-ranked predictions -- accuracy/AUC are
    # unaffected, but individual scores don't "look" confident, which hurts
    # explainability/demo readability (Section 5.4 wants a clear fraud
    # probability, not a number that's ambiguous on its face).
    #
    # Calibrated on X_val/y_val -- NOT X_test -- so this doesn't reintroduce
    # the test-set leakage bug fixed above. sklearn >=1.8 removed
    # cv="prefit" in favor of wrapping the already-fitted model in
    # FrozenEstimator (see sklearn.calibration.CalibratedClassifierCV docs).
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    calibrated_clf = CalibratedClassifierCV(FrozenEstimator(clf), method="sigmoid")
    calibrated_clf.fit(X_val, y_val)

    # Final metrics use the CALIBRATED model, since that's what predict.py
    # actually serves as "fraud_probability". roc_auc is threshold-
    # independent and should stay ~equal to the uncalibrated model since
    # ranking doesn't change under a monotonic calibration map (sigmoid/
    # Platt scaling is monotonic).
    y_test_proba = calibrated_clf.predict_proba(X_test)[:, 1]

    # --- Decision threshold tuning ---
    # class_weight="balanced" on this imbalanced dataset (1130 fraud : 226
    # genuine, ~5:1) pushes the model to prioritize catching fraud, which
    # at the default 0.5 threshold produces a very high false-positive
    # rate on genuine images (in practice, most genuine screenshots ended
    # up scored above 0.5). For a real fraud-screening tool that's a
    # serious usability problem -- flagging most legitimate transactions
    # as fraud makes the tool something people learn to ignore.
    #
    # Threshold is chosen to maximize Youden's J statistic
    # (J = sensitivity + specificity - 1 = TPR - FPR) on the VALIDATION
    # set (not test, to avoid the same leakage issue fixed earlier).
    # Note: maximizing F1 here does NOT fix the false-positive problem --
    # sklearn's F1 is computed w.r.t. the fraud (positive) class only, so
    # with fraud as the majority class it barely penalizes a high
    # false-positive rate on the genuine (minority) class. Youden's J
    # explicitly balances true-positive rate against false-positive rate
    # regardless of class size, which is what's actually needed here.
    y_val_proba = calibrated_clf.predict_proba(X_val)[:, 1]
    candidate_thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold, best_j = 0.5, -2.0
    genuine_mask_val = (y_val == 0)
    for t in candidate_thresholds:
        pred_t = (y_val_proba >= t).astype(int)
        tpr = recall_score(y_val, pred_t, zero_division=0)  # sensitivity on fraud class
        fpr = pred_t[genuine_mask_val].mean() if genuine_mask_val.sum() > 0 else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j, best_threshold = j, t
    print(f"Tuned decision threshold (max Youden's J on validation set): {best_threshold:.3f} "
          f"(val J at this threshold: {best_j:.4f})")

    def _metrics_at(threshold):
        pred = (y_test_proba >= threshold).astype(int)
        genuine_mask_test = (y_test == 0)
        fpr = pred[genuine_mask_test].mean() if genuine_mask_test.sum() > 0 else float("nan")
        return {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "false_positive_rate_on_genuine": fpr,  # fraction of genuine test images wrongly flagged as fraud
        }

    metrics_default = _metrics_at(0.5)
    metrics_tuned = _metrics_at(best_threshold)
    roc_auc = roc_auc_score(y_test, y_test_proba) if len(set(y_test)) > 1 else float("nan")

    print("Test metrics @ threshold=0.5 (default):", {**metrics_default, "roc_auc": roc_auc})
    print(f"Test metrics @ threshold={best_threshold:.3f} (tuned):", {**metrics_tuned, "roc_auc": roc_auc})

    # metrics.json reports the TUNED-threshold numbers, since that's the
    # threshold predict.py will actually use for its is_fraud decision.
    metrics = {**metrics_tuned, "roc_auc": roc_auc, "decision_threshold": float(best_threshold)}

    # Persist everything predict.py needs. Two classifier artifacts:
    #   - lgbm_fraud_classifier.pkl: RAW LightGBM, used by shap.TreeExplainer
    #     (TreeExplainer needs direct tree access; it can't see through the
    #     CalibratedClassifierCV wrapper below)
    #   - lgbm_fraud_classifier_calibrated.pkl: calibrated wrapper, used for
    #     the actual fraud_probability output
    joblib.dump(clf, os.path.join(args.out_dir, "lgbm_fraud_classifier.pkl"))
    joblib.dump(calibrated_clf, os.path.join(args.out_dir, "lgbm_fraud_classifier_calibrated.pkl"))
    joblib.dump(float(best_threshold), os.path.join(args.out_dir, "decision_threshold.pkl"))
    pca_reducer.save(os.path.join(args.out_dir, "visual_pca.pkl"))
    joblib.dump(list(X.columns), os.path.join(args.out_dir, "feature_columns.pkl"))
    pd.Series(metrics).to_json(os.path.join(args.out_dir, "eval_metrics.json"))

    print(f"Artifacts saved to {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="CSV with image_path,label (or filename,label) columns")
    parser.add_argument("--out_dir", default="models/artifacts")
    parser.add_argument("--pca_components", type=int, default=32)
    parser.add_argument("--n_estimators", type=int, default=400, help="Number of trees for LightGBM")
    parser.add_argument("--sample_frac", type=float, default=1.0, help="Fraction of manifest to sample for quick runs (0-1]")
    parser.add_argument("--no_cache", action="store_true", help="Disable caching of OCR/visual embeddings")
    parser.add_argument("--no_balance", dest="balance_classes", action="store_false",
                         help="Disable class_weight='balanced' (default: balanced ON)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel OCR worker threads")
    parser.add_argument("--embed_batch_size", type=int, default=32, help="Batch size for GPU visual embedding extraction")
    parser.add_argument("--ocr_device", default="auto", help="'auto' (default), 'cpu', or 'gpu:0' -- passed to PaddleOCR")
    parser.add_argument("--lgbm_gpu", action="store_true",
                         help="Try training LightGBM on GPU (requires a GPU-built LightGBM install; "
                              "falls back to CPU automatically if unavailable)")
    parser.set_defaults(balance_classes=True)
    main(parser.parse_args())