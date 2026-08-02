# Adaptive Risk Fusion — Explainable Multi-Modal Phishing & Digital Fraud Detection

A modular machine learning system that detects phishing and fraud across five different
attack surfaces — malicious URLs, smishing (SMS), quishing (malicious QR codes), forged
payment screenshots, and phishing emails. Each module is independently trainable and
testable, and produces a **probabilistic, explainable** risk score rather than a bare
label, so results can be fused into a single adaptive risk engine.

```
                ┌────────────────────┐
   URL      ───▶│  Module 1 — URL    │───┐
                └────────────────────┘   │
                ┌────────────────────┐   │
   SMS      ───▶│  Module 2 — SMS    │───┤
                └────────────────────┘   │
                ┌────────────────────┐   │      Adaptive Risk Fusion Engine
   QR Code  ───▶│  Module 3 — QR     │───┼────▶   (design-stage, out of scope
                └────────────────────┘   │        for this repo)
                ┌────────────────────┐   │
 Screenshot ───▶│  Module 4 — Image  │───┤
                └────────────────────┘   │
                ┌────────────────────┐   │
   Email    ───▶│  Module 5 — Email  │───┘
                └────────────────────┘
```

## Modules

| Module | What it detects | Approach | Status |
|---|---|---|---|
| [`module1_url`](module1_url) | Phishing URLs | Lexical/structural/host features → LightGBM | ✅ Trained & evaluated |
| [`module2_sms`](module2_sms) | Smishing SMS | DistilBERT text classifier | 🟡 Pipeline ready, needs training run |
| [`module3_qr`](module3_qr) | Malicious ("quishing") QR codes | Pixel + metadata + texture features → LightGBM | ✅ Trained on synthetic data |
| [`module4_image`](module4_image) | Forged payment screenshots | PaddleOCR + EfficientNet-B0 → fusion → LightGBM | 🟡 Trained on synthetic forgeries |
| [`module5_email`](module5_email) | Phishing emails | DistilBERT (body) + LightGBM (headers/metadata) → MLP fusion | ✅ Trained (two-stage) |

Each module folder is self-contained: its own `data/`, `models/`, `explainability/`,
`api/`, and `requirements.txt`, plus a detailed README covering setup, training,
inference, and known limitations.

## Results snapshot

| Module | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| URL (module1) | 97.67% | 96.09% | 99.39% | 97.71% | 99.47% |
| Email — text branch (module5) | 95.11% | 89.89% | 98.26% | 93.89% | 99.04% |

See each module's README for full evaluation details, held-out split sizes, and caveats
(some numbers, e.g. QR and image module metrics, are on synthetic data — flagged clearly
in those READMEs).

## Explainability

Every module surfaces *why* it made a prediction, not just the probability:

- **SHAP** (TreeExplainer) — feature attribution for the LightGBM-based modules (URL, QR,
  metadata branch of email, fusion layer of image)
- **LIME** — token-level attribution for text models (SMS, email body)
- **Grad-CAM** — visual tampering-localization heatmaps for the image module

## Getting started

Each module is independent — install and run only the ones you need:

```bash
cd module1_url        # or module2_sms, module3_qr, module4_image, module5_email
pip install -r requirements.txt
```

Then follow the module's own README for dataset setup, training, and inference commands.
Most modules also expose a FastAPI service (`api/app.py`) for standalone use.

## Repository layout

```
ML_Models/
├── module1_url/      # Website URL phishing detection
├── module2_sms/       # SMS smishing detection
├── module3_qr/         # QR code (quishing) detection
├── module4_image/       # Payment screenshot forgery detection
├── module5_email/        # Email phishing detection
└── .gitignore
```

## Notes

- No public dataset exists for quishing QR codes or malicious payment screenshots, so
  modules 3 and 4 use synthetically generated data — see their READMEs for how and why,
  and what would be needed to move to real-world data.
- Modules are designed to be combined by a top-level Adaptive Risk Fusion Engine that
  takes each module's probabilistic output and produces one unified risk score. That
  fusion engine is not yet implemented in this repo.

## Requirements

Python 3.9+. See each module's `requirements.txt` for exact dependencies (LightGBM,
scikit-learn, PyTorch/transformers, PaddleOCR, FastAPI, SHAP, LIME, etc.).
