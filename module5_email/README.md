# Module 5 — Email Detection Module
### Adaptive Risk Fusion — Explainable Multi-Modal Phishing & Fraud Detection

This folder implements **only Module 5 (Email Detection)** from the system design document,
Section 5.5. It is a self-contained, independently trainable/testable unit, per the design
principle of modularity (Section 4.1).

```
Email (Text + Headers) → DistilBERT (Body Text) ┐
                                                  ├─→ MLP Fusion Layer → Phishing Probability
                        → LightGBM (Metadata)   ┘
```

## Architecture (per Section 5.5)

| Branch | Model | Signal |
|---|---|---|
| Text branch | DistilBERT | Semantic embedding of the email body |
| Metadata branch | LightGBM | SPF/DKIM/DMARC, sender reputation, reply-to mismatch, link count, link/display-text mismatch, urgency keywords, attachments |
| Fusion | Small MLP | Combines both branch outputs into one calibrated phishing probability |
| Explainability | LIME (text) + SHAP (metadata) | Per-branch, human-readable justification |

## Folder layout

```
module5_email/
├── data/
│   ├── generate_synthetic_data.py   # builds a labeled demo dataset (no public email-phishing
│   │                                  dataset ships with this repo — see design doc Section 10)
│   └── sample_emails/                # example .eml files for a quick manual test
├── text_branch/
│   └── distilbert_branch.py          # DistilBERT semantic branch (+ safe fallback)
├── metadata_branch/
│   ├── feature_extraction.py         # header/body → structured metadata features
│   └── lightgbm_model.py             # LightGBM classifier wrapper
├── fusion/
│   └── mlp_fusion.py                 # small MLP that fuses text_prob + metadata_prob (+ aux features)
├── explainability/
│   ├── lime_explainer.py             # LIME text explanations
│   └── shap_explainer.py             # SHAP TreeExplainer on LightGBM metadata model
├── api/
│   ├── schemas.py                    # pydantic request/response models
│   └── app.py                        # FastAPI orchestrator for this module
├── train.py                          # trains metadata branch, text branch, and fusion layer
├── predict.py                        # end-to-end inference on a raw .eml or text+headers
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate a small synthetic labeled dataset (header-rich, needed by the metadata branch
#    — see "Two-stage training" below for why this is kept separate from the real corpus)
python data/generate_synthetic_data.py

# 2. (Recommended) Stage 1 — pretrain the text branch on a real, diverse email corpus
python text_branch/pretrain_on_real_corpus.py

# 3. Stage 2 — train metadata branch, fine-tune text branch further, train fusion MLP
python train.py

# 4. Run inference on a sample email
python predict.py --eml data/sample_emails/phishing_1.eml
```

## Two-stage text-branch training

The only public email corpora with real RFC822 headers (Nazario, SpamAssassin) require
external downloads/agreements and don't ship with this repo (design doc Section 10). To
still get genuine language diversity into the DistilBERT branch instead of training only
on this repo's small templated synthetic set, training is split in two stages:

1. **`text_branch/pretrain_on_real_corpus.py`** fine-tunes DistilBERT on ~3,000 real,
   deduplicated emails sampled from `ealvaradob/phishing-dataset` (Hugging Face; Enron
   corpus + Kaggle phishing-emails, body-text only, no synthetic headers). This is where
   the model learns real phishing-language patterns.
2. **`train.py`** loads that checkpoint and fine-tunes it further (few epochs, low LR) on
   this repo's small header-rich synthetic set, alongside training the metadata branch
   (which needs real SPF/DKIM/From/Reply-To headers the real corpus doesn't have) and the
   fusion MLP. This is a standard "pretrain big, fine-tune small" transfer-learning setup.

If `text_branch/pretrain_on_real_corpus.py` hasn't been run, `train.py` falls back to
fine-tuning DistilBERT from the base `distilbert-base-uncased` checkpoint directly on the
90-sample synthetic training split (still works, just less diverse language exposure).

## Training results

**Stage 1 — text branch on real corpus** (2,550 train / 450 test, held-out split of a
3,000-email sample from `ealvaradob/phishing-dataset`):

| Metric | Score |
|---|---|
| Accuracy | 95.11% |
| Precision | 89.89% |
| Recall | 98.26% |
| F1 | 93.89% |
| ROC-AUC | 99.04% |

This is the representative estimate of text-branch performance on unseen, realistic email
language — treat this as the module's headline text-branch metric.

**Stage 2 — full pipeline on the synthetic set** (90 train / 30 test): metadata branch,
text branch (further fine-tuned), and fusion MLP all score **1.000 accuracy / F1 / ROC-AUC**
on this split. This is expected, not a sign of a well-generalizing model in isolation: the
synthetic generator (`generate_synthetic_data.py`) ties `spf_pass`/`dkim_pass` deterministically
to the label (legit = always pass, phishing = always fail) and draws from a small set of
~10 subject/body templates, so the synthetic test split is trivially separable. It's a useful
end-to-end pipeline sanity check (fusion wiring, explainability output, API contract), not a
generalization benchmark — don't quote 100% as a real-world number in the report; cite the
Stage 1 numbers instead, and note this limitation explicitly (examiners will ask about a
100% score).

## Notes on the DistilBERT branch

`text_branch/distilbert_branch.py` uses `transformers.DistilBertForSequenceClassification`
by default. Downloading `distilbert-base-uncased` requires internet access to
huggingface.co. The branch auto-detects whether `torch`/`transformers` and a usable
checkpoint/network are available and, if not, falls back to a lightweight TF‑IDF +
Logistic Regression text model implementing the exact same interface (`predict_proba`,
`embed`, `explain`) — useful for offline sandboxes, but the real DistilBERT branch (as
trained via the two-stage process above) should be used for the actual deliverable.

## Output contract (matches Section 5.5 "Output")

```json
{
  "phishing_probability": 0.87,
  "branch_confidence": {"text": 0.81, "metadata": 0.92},
  "top_metadata_features": [
    {"feature": "dkim_pass", "value": 0, "shap_contribution": 0.21},
    {"feature": "link_display_mismatch", "value": 1, "shap_contribution": 0.18}
  ],
  "top_text_tokens": [["verify", 0.34], ["urgent", 0.29], ["account", 0.11]],
  "explanation": "Flagged as high risk primarily due to failed DKIM authentication and a
   link whose visible text does not match its destination domain, corroborated by urgency
   language in the message body ('verify', 'urgent')."
}
```

This module's fused `phishing_probability` (plus its embedding/confidence, per the design
principle in Section 4.1 that every module returns probabilistic — not bare-label — output)
is what would be forwarded on to the top-level **Adaptive Risk Fusion Engine** (Section 6)
in the full five-module system; that cross-module engine is out of scope for this deliverable.