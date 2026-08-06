"""
fusion_engine/adapters.py

Thin wrapper functions that convert each module's raw predict() output into
one common contract, so fuse.py never has to deal with 5 different schemas.

Common contract (returned by every adapt_*() function):
{
    "module": "url" | "sms" | "qr" | "image" | "email",
    "risk_probability": float in [0, 1],   # higher = more malicious/fraudulent
    "label": "malicious" | "legitimate",
    "confidence": float in [0, 1],         # how sure the model is of its label
    "explanation": str,
}

Usage:
    from adapters import adapt_url, adapt_sms, adapt_qr, adapt_image, adapt_email

    raw = predict_url("http://bit.ly/xyz")
    normalized = adapt_url(raw)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Module 1 - URL
# Raw keys: "URL", "Prediction", "Risk Score", "Confidence",
#           "Legitimate Probability", "Phishing Probability"
# Raw scale: 0-100 (percentage), Capitalized keys with spaces
# ---------------------------------------------------------------------------
def adapt_url(raw: dict) -> dict:
    phishing_prob = raw["Phishing Probability"] / 100.0
    label = "malicious" if raw["Prediction"] == "Phishing" else "legitimate"
    confidence = raw["Confidence"] / 100.0

    explanation = (
        f"URL classified as {raw['Prediction'].lower()} "
        f"with a phishing probability of {raw['Phishing Probability']}%."
    )

    return {
        "module": "url",
        "risk_probability": round(phishing_prob, 4),
        "label": label,
        "confidence": round(confidence, 4),
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# Module 2 - SMS
# Raw keys: text, label, smishing_probability, confidence, embedding
# Raw scale: 0-1, snake_case
# ---------------------------------------------------------------------------
def adapt_sms(raw: dict) -> dict:
    label = "malicious" if raw["label"] == "smishing" else "legitimate"

    explanation = (
        f"SMS text classified as {raw['label']} "
        f"with probability {raw['smishing_probability']:.2f}."
    )

    return {
        "module": "sms",
        "risk_probability": round(raw["smishing_probability"], 4),
        "label": label,
        "confidence": round(raw["confidence"], 4),
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# Module 3 - QR
# Raw keys: malicious_qr_probability, label, decoded_url,
#           top_contributing_features, explanation
# Raw scale: 0-1, snake_case
# ---------------------------------------------------------------------------
def adapt_qr(raw: dict) -> dict:
    prob = raw["malicious_qr_probability"]
    confidence = prob if prob >= 0.5 else 1 - prob

    return {
        "module": "qr",
        "risk_probability": round(prob, 4),
        "label": raw["label"],  # already "malicious" / "legitimate"
        "confidence": round(confidence, 4),
        "explanation": raw.get("explanation", ""),
    }


# ---------------------------------------------------------------------------
# Module 4 - Payment Image
# Raw keys: fraud_probability, is_fraud, decision_threshold,
#           structured_fields, ocr_quality, top_contributing_features,
#           tampering_heatmap_path
# Raw scale: 0-1, snake_case
# ---------------------------------------------------------------------------
def adapt_image(raw: dict) -> dict:
    prob = raw["fraud_probability"]
    confidence = prob if prob >= 0.5 else 1 - prob
    label = "malicious" if raw["is_fraud"] else "legitimate"

    top_features = raw.get("top_contributing_features") or []
    if top_features:
        feature_summary = ", ".join(str(f) for f in top_features[:3])
        explanation = f"Payment screenshot flagged as {label} based on: {feature_summary}."
    else:
        explanation = f"Payment screenshot classified as {label}."

    return {
        "module": "image",
        "risk_probability": round(prob, 4),
        "label": label,
        "confidence": round(confidence, 4),
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# Module 5 - Email
# Raw keys: phishing_probability, branch_confidence {text, metadata},
#           top_metadata_features, top_text_tokens, explanation,
#           raw_metadata_features
# Raw scale: 0-1, snake_case
# ---------------------------------------------------------------------------
def adapt_email(raw: dict) -> dict:
    prob = raw["phishing_probability"]
    label = "malicious" if prob >= 0.5 else "legitimate"

    # Average of the two branch confidences as the overall confidence signal
    branch_conf = raw.get("branch_confidence", {})
    text_conf = branch_conf.get("text", prob)
    meta_conf = branch_conf.get("metadata", prob)
    confidence = (text_conf + meta_conf) / 2

    return {
        "module": "email",
        "risk_probability": round(prob, 4),
        "label": label,
        "confidence": round(confidence, 4),
        "explanation": raw.get("explanation", ""),
    }


# ---------------------------------------------------------------------------
# Convenience lookup, so fuse.py can do: ADAPTERS["url"](raw_output)
# ---------------------------------------------------------------------------
ADAPTERS = {
    "url": adapt_url,
    "sms": adapt_sms,
    "qr": adapt_qr,
    "image": adapt_image,
    "email": adapt_email,
}


if __name__ == "__main__":
    # Quick smoke test with fake raw outputs matching each module's real schema
    print(adapt_url({
        "URL": "http://bit.ly/xyz", "Prediction": "Phishing",
        "Risk Score": 91.23, "Confidence": 91.23,
        "Legitimate Probability": 8.77, "Phishing Probability": 91.23,
    }))

    print(adapt_sms({
        "text": "You won a prize!", "label": "smishing",
        "smishing_probability": 0.9788, "confidence": 0.9788, "embedding": [],
    }))

    print(adapt_qr({
        "malicious_qr_probability": 0.72, "label": "malicious",
        "decoded_url": "http://scam.example", "top_contributing_features": [],
        "explanation": "QR points to a newly registered domain.",
    }))

    print(adapt_image({
        "fraud_probability": 0.767, "is_fraud": True,
        "decision_threshold": 0.6, "top_contributing_features": ["ela_score", "font_mismatch"],
    }))

    print(adapt_email({
        "phishing_probability": 0.85,
        "branch_confidence": {"text": 0.8, "metadata": 0.9},
        "explanation": "Failed DKIM, urgent language in body.",
    }))
