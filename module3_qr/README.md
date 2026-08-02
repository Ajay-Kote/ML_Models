# Module 3 — QR Code Analysis (Quishing Detection)

Implements Section 5.3 of the *Adaptive Risk Fusion* design document: detects
malicious ("quishing") QR codes directly from the image — pixel + structural
metadata + visual texture — **without decoding and visiting the embedded URL
first**.

```
QR Code Image -> [Pixel + Metadata + Texture Features] -> LightGBM -> Malicious-QR Probability
```

## Folder structure

```
module3_qr/
├── data/
│   ├── generate_dataset.py   # synthesizes labeled QR image dataset (no public dataset exists)
│   └── raw/                  # generated images/ + metadata.jsonl (created by the script above)
├── feature_extraction/
│   ├── pixel.py               # flattened grayscale pixel matrix
│   ├── metadata.py            # QR version, error-correction level, module density, quiet-zone size
│   ├── texture.py             # Local Binary Patterns, edge density, contrast/entropy
│   └── build_features.py      # concatenates all three into one feature vector
├── models/
│   ├── train.py                # trains + evaluates the LightGBM classifier
│   ├── predict.py               # single-image inference + explanation
│   └── saved_model.pkl / shap_background.npy   (produced by train.py)
├── explainability/
│   └── shap_explainer.py      # SHAP TreeExplainer -> human-readable justification
├── api/
│   └── app.py                  # FastAPI service: POST /analyze-qr
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate the synthetic labeled dataset (legit vs. quishing-style QR codes)
python data/generate_dataset.py --n_per_class 600

# 2. Train the LightGBM classifier (prints Accuracy/Precision/Recall/F1/ROC-AUC)
python models/train.py

# 3. Run inference on a single QR image
python models/predict.py data/raw/images/qr_00001.png

# 4. (Optional) Serve as a standalone API
uvicorn api.app:app --reload --port 8003
# then: curl -F "file=@data/raw/images/qr_00001.png" http://localhost:8003/analyze-qr
```

## Design notes / how this maps to the spec

- **Why synthetic data**: Section 10 of the design doc notes no public
  malicious-QR image dataset exists. `generate_dataset.py` builds one by
  rendering real QR codes for legitimate vs. phishing-style URLs (reusing
  the URL module's own risk patterns: IP hosts, brand-keyword subdomains,
  high-risk TLDs, long random strings), while varying error-correction
  level, box size, and border in a way that correlates with the label —
  mirroring how quishing generators favor low error-correction / dense
  encoding to cram payloads into small printed or screenshotted codes.
- **Train/serve consistency**: metadata features are extracted the *same
  way* at train and inference time (estimated from the raw image via
  `cv2.QRCodeDetector`, not from generation-time ground truth), to avoid
  train/serve skew that would silently inflate offline metrics.
- **No decode-before-analysis**: the classifier never needs to decode or
  visit the embedded URL to produce a risk score; `predict.py` decodes the
  payload only as a *secondary*, informational field it can hand off to the
  URL Detection Module (module1_url), matching the "optionally forwarded"
  behavior described in Section 5.3.
- **Explainability (Section 7)**: `shap_explainer.py` uses SHAP's
  `TreeExplainer` on the LightGBM model. Raw pixel/LBP features are
  aggregated into single "visual pixel pattern" / "local texture pattern"
  line items (they're meaningless individually), while metadata features
  are surfaced by name — e.g. *"low error-correction level"* — exactly the
  human-interpretable explanation style called out in the design note.
- **Citation**: this module extends the pixel-only approach of Trad & Chehab
  (2025), *"Detecting Quishing Attacks with Machine Learning Techniques
  Through QR Code Analysis,"* arXiv:2505.03451, by adding structural
  metadata and texture descriptors — cite this in the related-work section
  of the IEEE paper as noted in the design document.

## Output schema (`predict.py` / `POST /analyze-qr`)

```json
{
  "malicious_qr_probability": 0.87,
  "label": "malicious",
  "decoded_url": "http://192.0.2.10/paypal-verify",
  "top_contributing_features": [
    {"feature": "error-correction level", "shap_contribution": 0.21, "direction": "increases risk"},
    {"feature": "quiet-zone (border) size", "shap_contribution": 0.14, "direction": "increases risk"}
  ],
  "explanation": "Flagged as high risk primarily due to error-correction level, quiet-zone (border) size."
}
```
