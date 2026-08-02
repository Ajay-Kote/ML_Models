"""
explainability/lime_explainer.py

Design doc Section 7: "DistilBERT (SMS, Email text) -> LIME (text explainer) ->
Which words/phrases drove the classification."

Model-agnostic: works against anything exposing `predict_proba(text: str) -> float`
(both DistilBertTextBranch and TfidfFallbackBranch satisfy this).
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

try:
    from lime.lime_text import LimeTextExplainer
    _LIME_AVAILABLE = True
except Exception:  # pragma: no cover
    _LIME_AVAILABLE = False


def _proba_fn_factory(text_branch):
    """LIME expects a function: list[str] -> np.ndarray[n_samples, n_classes]."""
    def proba_fn(texts: List[str]) -> np.ndarray:
        probs = [text_branch.predict_proba(t) for t in texts]
        return np.array([[1 - p, p] for p in probs])
    return proba_fn


def explain_text(text_branch, text: str, top_k: int = 8) -> List[Tuple[str, float]]:
    """
    Returns the top_k (token, weight) pairs contributing to the phishing prediction,
    sorted by absolute weight descending. Positive weight = pushes toward "phishing".
    """
    if not text or not text.strip():
        return []

    if _LIME_AVAILABLE:
        try:
            explainer = LimeTextExplainer(class_names=["legitimate", "phishing"])
            proba_fn = _proba_fn_factory(text_branch)
            n_features = min(top_k, max(1, len(text.split())))
            # exp = explainer.explain_instance(text, proba_fn, num_features=n_features, labels=(1,))
            exp = explainer.explain_instance(text, proba_fn, num_features=n_features, labels=(1,), num_samples=200)
            weights = exp.as_list(label=1)
            weights.sort(key=lambda kv: -abs(kv[1]))
            return weights[:top_k]
        except Exception as e:  # pragma: no cover — LIME can fail on very short/odd texts
            print(f"[lime_explainer] LIME failed ({e.__class__.__name__}: {e}), using fallback ranking.")

    return _keyword_fallback(text, top_k)


def _keyword_fallback(text: str, top_k: int) -> List[Tuple[str, float]]:
    """
    Deterministic fallback used only if the `lime` package itself is unavailable: ranks
    tokens by overlap with the design doc's urgency-keyword list (Section 5.5 metadata
    branch feature set), reused here as a coarse proxy signal.
    """
    from metadata_branch.feature_extraction import URGENCY_KEYWORDS

    tokens = [t.strip(".,!?:;\"'()").lower() for t in text.split()]
    scored = []
    for tok in set(tokens):
        if not tok:
            continue
        weight = 0.3 if any(tok in kw or kw in tok for kw in URGENCY_KEYWORDS) else 0.02
        scored.append((tok, weight))
    scored.sort(key=lambda kv: -kv[1])
    return scored[:top_k]
