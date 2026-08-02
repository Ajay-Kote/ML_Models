# Module 1 — Website URL Phishing Detection

Detects phishing URLs using lexical, structural, and host-based features fed
into a LightGBM classifier. Part of the "Adaptive Risk Fusion for Explainable
Multi-Modal Phishing and Digital Fraud Detection" project.

## How it works

1. `feature_extraction/url_feature_extractor.py` parses a raw URL and derives
   ~30+ features — length stats, dot/slash/hyphen counts, digit ratios,
   entropy, brand-name edit distance, subdomain count, suspicious keyword
   flags, etc. (`lexical_features.py` and `utils.py` hold the individual
   feature functions.)
2. `models/train.py` trains a LightGBM classifier on
   `data/processed/features.csv` (derived from the PhiUSIIL Phishing URL
   Dataset) and saves it to `models/saved_model.pkl`.
3. `models/predict.py` loads the saved model and exposes `predict_url(url)`,
   which returns the prediction, risk score, and class probabilities.
4. `explainability/shap_explainer.py` provides SHAP-based feature attribution
   for individual predictions.
5. `api/app.py` wraps `predict_url` in a FastAPI `/predict` endpoint.

## Test-set performance

(`models/evaluate.py`, 20% held-out split, stratified, random_state=42 —
42,247 test samples)

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 97.67% |
| Precision | 96.09% |
| Recall    | 99.39% |
| F1 Score  | 97.71% |
| ROC-AUC   | 99.47% |

Top predictive features (see `results/feature_importance.csv`): path length,
domain length, host length, brand edit-distance, dot count.

Confusion matrix, ROC curve, and feature importance plots are in `results/`.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Predict a single URL
python main.py "http://paypal-verify-login.suspicious-domain.tk/secure"

# Interactive mode
python main.py

# Launch the FastAPI server (POST /predict with {"url": "..."})
python main.py --serve

# Re-run evaluation on the held-out test split
python main.py --evaluate
```

Direct predict/evaluate scripts can also be run standalone:

```bash
python models/predict.py
python models/train.py
python models/evaluate.py
```

## Notes

- On first run, `tldextract` downloads and caches the public suffix list. If
  offline, it silently falls back to a bundled snapshot — predictions are
  unaffected.
- Label convention: `0 = Phishing`, `1 = Legitimate` (matches training data).