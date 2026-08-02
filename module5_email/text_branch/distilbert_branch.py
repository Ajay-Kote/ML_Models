"""
text_branch/distilbert_branch.py

Text branch of the Email Detection Module (design doc, Section 5.5):
"Text branch: Email body text -> DistilBERT semantic embedding".

Two implementations share one interface (`fit`, `predict_proba`, `embed`, `top_tokens`,
`save`, `load`):

  * DistilBertTextBranch   — the real model: `distilbert-base-uncased-finetuned-sst-2-english`
                              style fine-tuning via HuggingFace `transformers`, used in any
                              environment with `torch`+`transformers` installed AND network
                              access to download pretrained weights.
  * TfidfFallbackBranch    — a lightweight TF-IDF + LogisticRegression model used whenever
                              torch/transformers aren't available or weights can't be
                              downloaded (e.g. an offline sandbox). This keeps the module
                              runnable end-to-end everywhere without changing any other file.

`get_text_branch()` auto-selects the best available implementation.
"""

from __future__ import annotations

import re
import joblib
import numpy as np
from pathlib import Path
from typing import List, Tuple

_TORCH_TRANSFORMERS_AVAILABLE = True
try:
    import torch
    import torch.nn as nn
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
except Exception:  # pragma: no cover - exercised only when deps are missing
    _TORCH_TRANSFORMERS_AVAILABLE = False


# ---------------------------------------------------------------------------------------
# Real DistilBERT implementation
# ---------------------------------------------------------------------------------------

class DistilBertTextBranch:
    """
    Fine-tuned DistilBERT sequence classifier for phishing-email body text.

    embed() returns the 768-dim pooled [CLS] hidden state, which the fusion MLP can use as
    an auxiliary signal alongside the scalar phishing probability, per design doc Section
    5.2/5.5 ("pooled embedding vector ... forwarded to the fusion engine as an auxiliary
    feature").
    """

    MODEL_NAME = "distilbert-base-uncased"

    def __init__(self, model_dir: str | None = None, device: str | None = None):
        if not _TORCH_TRANSFORMERS_AVAILABLE:
            raise RuntimeError("torch/transformers not installed — use get_text_branch() instead.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        source = model_dir or self.MODEL_NAME
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(source)
        self.model = DistilBertForSequenceClassification.from_pretrained(source, num_labels=2)
        self.model.to(self.device)
        self.model.eval()

    def fit(self, texts: List[str], labels: List[int], epochs: int = 3, lr: float = 2e-5, batch_size: int = 8):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_labels = torch.tensor(labels[i:i + batch_size]).to(self.device)
                enc = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
                optimizer.zero_grad()
                out = self.model(**enc, labels=batch_labels)
                out.loss.backward()
                optimizer.step()
        self.model.eval()
        return self

    def predict_proba(self, text: str) -> float:
        with torch.no_grad():
            enc = self.tokenizer(text, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            return float(probs[0, 1].item())  # index 1 = phishing

    def embed(self, text: str) -> np.ndarray:
        with torch.no_grad():
            enc = self.tokenizer(text, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            out = self.model.distilbert(**enc)
            pooled = out.last_hidden_state[:, 0, :]  # [CLS] token
            return pooled.squeeze(0).cpu().numpy()

    def top_tokens(self, text: str, top_k: int = 8) -> List[Tuple[str, float]]:
        """Delegates to explainability/lime_explainer.py for a model-agnostic LIME explanation."""
        from explainability.lime_explainer import explain_text
        return explain_text(self, text, top_k=top_k)

    def save(self, path: str | Path):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str | Path):
        return cls(model_dir=str(path))


# ---------------------------------------------------------------------------------------
# Lightweight fallback implementation (no internet / no torch required)
# ---------------------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zA-Z']{2,}")


class TfidfFallbackBranch:
    """
    Drop-in replacement for DistilBertTextBranch with the same public interface, backed by
    TF-IDF + LogisticRegression (both pure scikit-learn, CPU-only, no downloads required).

    embed() returns the TF-IDF sparse vector densified (dimensionality = vocab size) so the
    fusion layer can still consume a "semantic embedding"-shaped signal; top_tokens() returns
    the highest-weighted TF-IDF terms scaled by the logistic-regression coefficient, which
    approximates what LIME would surface for the real model.
    """

    def __init__(self, max_features: int = 2000):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words="english")
        self.classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._fitted = False

    def fit(self, texts: List[str], labels: List[int], **_ignored):
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self._fitted = True
        return self

    def predict_proba(self, text: str) -> float:
        if not self._fitted:
            raise RuntimeError("TfidfFallbackBranch has not been trained. Call fit() or load() first.")
        X = self.vectorizer.transform([text])
        return float(self.classifier.predict_proba(X)[0, 1])

    def embed(self, text: str) -> np.ndarray:
        X = self.vectorizer.transform([text])
        return X.toarray().squeeze(0)

    def top_tokens(self, text: str, top_k: int = 8) -> List[Tuple[str, float]]:
        from explainability.lime_explainer import explain_text
        return explain_text(self, text, top_k=top_k)

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "classifier": self.classifier}, path)

    @classmethod
    def load(cls, path: str | Path):
        payload = joblib.load(path)
        obj = cls()
        obj.vectorizer = payload["vectorizer"]
        obj.classifier = payload["classifier"]
        obj._fitted = True
        return obj


# ---------------------------------------------------------------------------------------
# Auto-selection
# ---------------------------------------------------------------------------------------

FALLBACK_MODEL_PATH = Path(__file__).parent / "saved_model" / "tfidf_fallback.joblib"
DISTILBERT_MODEL_DIR = Path(__file__).parent / "saved_model" / "distilbert"


def get_text_branch(prefer_distilbert: bool = True):
    """
    Returns a trained/loadable text-branch instance, preferring real DistilBERT when both
    (a) torch/transformers are installed and (b) a fine-tuned checkpoint or internet access
    to download pretrained weights is available. Otherwise returns the TF-IDF fallback.
    """
    if prefer_distilbert and _TORCH_TRANSFORMERS_AVAILABLE:
        try:
            if DISTILBERT_MODEL_DIR.exists():
                return DistilBertTextBranch.load(DISTILBERT_MODEL_DIR)
            return DistilBertTextBranch()  # downloads distilbert-base-uncased
        except Exception as e:  # network unavailable, etc.
            print(f"[text_branch] Falling back to TF-IDF text model ({e.__class__.__name__}: {e})")
    if FALLBACK_MODEL_PATH.exists():
        return TfidfFallbackBranch.load(FALLBACK_MODEL_PATH)
    return TfidfFallbackBranch()
