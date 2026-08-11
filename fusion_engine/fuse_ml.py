"""
fuse_ml.py

ML-based fusion using the trained Logistic Regression meta-learner.
Drop this file into fusion_engine/ alongside the existing fuse.py.

This does NOT replace fuse.py's rule-based fuse() -- it adds a new
fuse_ml() so you can keep the rule-based one as a fallback (e.g. if
meta_learner.joblib fails to load) and still call it "adaptive" fusion
because the weights are now learned, not hand-picked.

Usage:
    from fuse_ml import fuse_ml
    result = fuse_ml({"url": adapt_url(raw_url), "sms": adapt_sms(raw_sms)})
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from fuse import FusionInputError, OVERRIDE_THRESHOLD, SAFE_MAX, SUSPICIOUS_MAX

MODULES = ["url", "sms", "qr", "image", "email"]
FEATURE_COLS = [f"{m}_risk" for m in MODULES] + [f"{m}_present" for m in MODULES]
MODEL_PATH = Path(__file__).resolve().parent / "meta_learner.joblib"

_model = None  # lazy-loaded singleton


def _load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{MODEL_PATH} not found. Run train_meta_learner.py first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def _verdict_from_score(score: float) -> str:
    if score < SAFE_MAX:
        return "safe"
    if score < SUSPICIOUS_MAX:
        return "suspicious"
    return "high_risk"


def _build_feature_vector(module_outputs: dict[str, dict]) -> list[float]:
    """10 features in fixed order: [url_risk..email_risk, url_present..email_present]"""
    risks = []
    presents = []
    for m in MODULES:
        if m in module_outputs:
            risks.append(module_outputs[m]["risk_probability"])
            presents.append(1)
        else:
            risks.append(0.5)  # neutral placeholder — matches training-time convention
            presents.append(0)
    return risks + presents


def fuse_ml(module_outputs: dict[str, dict]) -> dict:
    """
    Same input/output contract as fuse.fuse(), but the combined score comes
    from a trained Logistic Regression meta-learner instead of a fixed
    weighted average. The same >=90% single-module override rule from
    fuse.py is kept, since that's a safety rule, not a weighting choice.
    """
    if not module_outputs:
        raise FusionInputError("fuse_ml() requires at least one module output.")

    model = _load_model()

    # --- Keep the same hard safety override as the rule-based version ---
    override_module = None
    override_prob = 0.0
    for name, out in module_outputs.items():
        if out["label"] == "malicious" and out["risk_probability"] >= OVERRIDE_THRESHOLD:
            if out["risk_probability"] > override_prob:
                override_module = name
                override_prob = out["risk_probability"]

    if override_module is not None:
        combined_score = override_prob
        verdict = "high_risk"
        override_triggered = True
        explanation = (
            f"High-risk override: '{override_module}' module reported "
            f"{override_prob:.2%} risk probability, which alone is enough "
            f"to flag this as high risk regardless of other modules."
        )
    else:
        features = pd.DataFrame([_build_feature_vector(module_outputs)], columns=FEATURE_COLS)
        combined_score = float(model.predict_proba(features)[0][1])
        verdict = _verdict_from_score(combined_score)
        override_triggered = False
        explanation = (
            f"Combined risk score {combined_score:.2%} from the trained fusion "
            f"model using {len(module_outputs)} module(s): "
            + ", ".join(f"{m} ({module_outputs[m]['risk_probability']:.0%})" for m in module_outputs)
            + f". Verdict: {verdict}."
        )

    # Feature importance for THIS request, from the model's learned coefficients,
    # so contributing_modules stays explainable like the rule-based version.
    clf = model.named_steps["clf"]
    coefs = dict(zip(
        [f"{m}_risk" for m in MODULES] + [f"{m}_present" for m in MODULES],
        clf.coef_[0],
    ))

    contributing_modules = []
    for m in module_outputs:
        out = module_outputs[m]
        contributing_modules.append({
            "module": m,
            "risk_probability": out["risk_probability"],
            "label": out["label"],
            "learned_weight": round(float(coefs[f"{m}_risk"]), 4),
            "explanation": out.get("explanation", ""),
        })

    return {
        "combined_risk_probability": round(combined_score, 4),
        "verdict": verdict,
        "override_triggered": override_triggered,
        "override_module": override_module,
        "contributing_modules": contributing_modules,
        "explanation": explanation,
        "fusion_method": "ml_meta_learner",
    }


if __name__ == "__main__":
    sample = {
        "url": {"module": "url", "risk_probability": 0.82, "label": "malicious",
                 "confidence": 0.82, "explanation": "Suspicious domain age + typo-squatting."},
        "email": {"module": "email", "risk_probability": 0.70, "label": "malicious",
                   "confidence": 0.70, "explanation": "Urgent language, failed DKIM."},
    }
    import json
    print(json.dumps(fuse_ml(sample), indent=2))
