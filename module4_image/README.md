# Module 4 — Payment Image (Screenshot) Detection

Implements Section 5.4 of the System Design Document:
`OCR (PaddleOCR) -> EfficientNet-B0 -> Feature Fusion -> LightGBM -> Fraud Probability`

## Quick start (just these 3 commands)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python -m models.train --manifest data/labels_all.csv --out_dir models/artifacts

python -m models.predict data/fake/fake_0000_0_splice.jpg --heatmap_out heatmap.jpg
```

That's it. `data/labels_all.csv` is a pre-built combined manifest (542 rows: 173 genuine +
369 forged, merged into one pool from your originally separate gpay / phonepe-white /
phonepe-black datasets — see `data/CLEANING_REPORT.md`) with `class_weight="balanced"`
already ON by default in `train.py` to correct for the genuine:forged imbalance.

## Layout
```
module4_image/
├── data/labels_all.csv                 # pre-built combined manifest (542 rows) -- use this
├── data/real/                           # merged real screenshots, real_0000.jpg, real_0001.jpg, ...
├── data/fake/                           # merged forged screenshots, fake_{real_idx}_{variant}_{type}.jpg
├── data/meta/                           # per-fake-image bbox + manipulation_type + origin_app json
├── data/CLEANING_REPORT.md              # how the 3 original per-app datasets were merged into the above
├── data/generate_forgeries.py           # forgery generator (splice/recompression/text_overlay/local_inconsistency)
├── data/add_new_batch.py                # adds new real photos + auto-generates their forged variants
├── ocr/paddleocr_pipeline.py            # PaddleOCR 3.x text extraction + field parsing + quality signals
├── vision/efficientnet_extractor.py     # EfficientNet-B0 embeddings (also Grad-CAM target layer)
├── fusion/feature_fusion.py             # merges OCR + PCA-reduced visual embedding into one feature row
├── models/train.py                      # trains LightGBM on a manifest (supports filename or image_path CSVs)
├── models/predict.py                    # end-to-end inference (prob + fields + heatmap + SHAP)
├── explainability/shap_explainer.py     # SHAP TreeExplainer over the LightGBM classifier
├── explainability/gradcam_explainer.py  # Grad-CAM tampering-localization heatmap
├── api/app.py                           # standalone FastAPI service for this module
└── tests/test_manifest_loading.py       # sanity test for the filename->image_path resolver
```

## Training options
```powershell
# Quick debug run on 20% of the data, fewer trees, faster iteration:
python -m models.train --manifest data/labels_all.csv --out_dir models/artifacts --sample_frac 0.2 --n_estimators 50

# Train on a subset filtered by origin app (origin_app is recorded per-fake-image
# in data/meta/*.json -- see data/CLEANING_REPORT.md for the index ranges):
python -c "
import pandas as pd, json
from pathlib import Path
df = pd.read_csv('data/labels_all.csv')
fake_origins = {f'data/fake/{j.stem}.jpg': json.loads(j.read_text()).get('origin_app')
                for j in Path('data/meta').glob('*.json')}
subset = df[(df['label'] == 0) | (df['image_path'].map(fake_origins) == 'phonepeblack')]
subset.to_csv('data/phonepeblack_subset.csv', index=False)
"
python -m models.train --manifest data/phonepeblack_subset.csv --out_dir models/artifacts

# Disable class balancing (not recommended given the 1:3 skew):
python -m models.train --manifest data/labels_all.csv --out_dir models/artifacts --no_balance

# On a machine with a GPU (see GPU_SETUP.md first): more OCR worker threads
# and a bigger embedding batch size for faster extraction:
python -m models.train --manifest data/labels_all.csv --out_dir models/artifacts --workers 8 --embed_batch_size 64
```

**Performance flags** (all optional, sensible defaults if omitted):
- `--workers N` (default 4) — parallel OCR extraction threads
- `--embed_batch_size N` (default 32) — images per GPU forward pass for visual embeddings
- `--ocr_device auto|cpu|gpu:0` (default `auto`) — auto-detects GPU via paddle; see GPU_SETUP.md
- `--lgbm_gpu` — try GPU LightGBM (requires a GPU-built LightGBM install; falls back to CPU otherwise)

Training prints progress bars (OCR pass, then embedding pass) so it never looks frozen, and
device auto-detection prints what it picked, e.g. `[VisualEmbeddingService] Using GPU: NVIDIA GeForce RTX 4090`.

OCR + visual-embedding results are cached per-image in `models/artifacts/feature_cache.joblib`
after the first run, so re-running training (e.g. after changing LightGBM hyperparameters)
is much faster the second time. Use `--no_cache` to force fresh extraction.

## Run inference on one screenshot
```powershell
python -m models.predict path\to\screenshot.jpg --heatmap_out heatmap.jpg
```
Prints fraud probability, extracted structured fields, top SHAP drivers, and saves a
Grad-CAM tampering heatmap.

## Run as a live API
```powershell
uvicorn api.app:app --reload --port 8004
```
POST an image to `http://localhost:8004/predict`.

## Run the sanity test
```powershell
python -m unittest tests.test_manifest_loading
```

## Known gaps (be upfront about these in your report/viva)
- **Dataset is entirely programmatic forgeries** (splice/recompression/text_overlay via
  `generate_forgeries.py`), not real-world tampered screenshots. Good for testing the
  pipeline end-to-end and demonstrating the architecture works; before final evaluation
  numbers go in the paper, real forged samples (or at least visually-inspected manual
  edits) strengthen the claim significantly.
- **Grad-CAM's `classifier_head`** in `EfficientNetB0FeatureExtractor` is untrained —
  fine-tune it on your labeled data before trusting heatmaps as tampering-specific;
  right now it highlights generic salient regions.
- **1:3 genuine:forged imbalance** is corrected via `class_weight="balanced"`, but with
  only 123 genuine examples total, watch `eval_metrics.json`'s precision/recall gap —
  if precision on the genuine class stays low, you likely need more genuine samples,
  not just reweighting.
- Only 3 app layouts covered (GPay, PhonePe light, PhonePe dark) — regex patterns in
  `ocr/paddleocr_pipeline.py` (`KNOWN_APPS`, field regexes) are tuned to these; extend
  before adding Paytm/other layouts.
