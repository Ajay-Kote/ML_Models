"""
predict.py

End-to-end inference for the Email Detection Module (design doc, Section 5.5 "Output":
"Phishing probability, per-branch confidence (text vs metadata), and top contributing
metadata features.") Also generates the LIME text explanation and a fused human-readable
justification, matching the Explainability Layer described in Section 7.

Usage:
    python predict.py --eml data/sample_emails/phishing_1.eml
    python predict.py --eml data/sample_emails/legitimate_1.eml
    echo "raw rfc822 text" | python predict.py --stdin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from metadata_branch.feature_extraction import extract_features, extract_body_text
from metadata_branch.lightgbm_model import EmailMetadataClassifier
from text_branch.distilbert_branch import get_text_branch
from fusion.mlp_fusion import EmailFusionMLP
from explainability.shap_explainer import explain_metadata

MODEL_DIR = Path(__file__).parent / "saved_models"
METADATA_MODEL_PATH = MODEL_DIR / "metadata_lightgbm.joblib"
FUSION_MODEL_PATH = MODEL_DIR / "fusion_mlp.joblib"


def _build_explanation(meta_top: list[dict], text_top: list[tuple], fused_prob: float) -> str:
    if fused_prob < 0.5:
        return ("No strong phishing indicators found. Authentication checks passed and the "
                "message body does not show urgency or mismatch patterns typical of phishing.")

    meta_phrases = []
    for item in meta_top[:2]:
        feat, val = item["feature"], item["value"]
        if item["shap_contribution"] <= 0:
            continue
        if feat in ("spf_pass", "dkim_pass", "dmarc_pass") and val == 0:
            meta_phrases.append(f"failed {feat.replace('_pass', '').upper()} authentication")
        elif feat == "link_display_mismatch_count" and val > 0:
            meta_phrases.append("a link whose visible text does not match its destination domain")
        elif feat == "reply_to_mismatch" and val == 1:
            meta_phrases.append("a Reply-To address that does not match the sender's domain")
        elif feat == "urgency_keyword_count" and val > 0:
            meta_phrases.append("urgent/time-pressure language")
        elif feat == "sender_domain_high_risk_tld" and val == 1:
            meta_phrases.append("a sender domain on a high-risk top-level domain")
        elif feat == "attachment_is_executable_or_archive" and val == 1:
            meta_phrases.append("a risky attachment type")

    text_words = [tok for tok, w in text_top[:3] if w > 0]

    parts = []
    if meta_phrases:
        parts.append("primarily due to " + ", ".join(meta_phrases))
    if text_words:
        parts.append(f"corroborated by suspicious language in the body ('{', '.join(text_words)}')")

    if not parts:
        return "Flagged as high risk by the fused model, though no single dominant signal stood out."
    return "Flagged as high risk " + " and ".join(parts) + "."


def predict(raw_email: str) -> dict:
    if not METADATA_MODEL_PATH.exists() or not FUSION_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained models not found. Run `python train.py` first "
            f"(expected {METADATA_MODEL_PATH} and {FUSION_MODEL_PATH})."
        )

    metadata_clf = EmailMetadataClassifier.load(METADATA_MODEL_PATH)
    fusion = EmailFusionMLP.load(FUSION_MODEL_PATH)
    text_branch = get_text_branch(prefer_distilbert=True)

    meta_features = extract_features(raw_email)
    body_text = extract_body_text(raw_email) or " "

    metadata_prob = metadata_clf.predict_proba(meta_features)
    text_prob = text_branch.predict_proba(body_text)

    fusion_vec = fusion.build_feature_vector(text_prob, metadata_prob, meta_features)
    fused_prob = fusion.predict_proba(fusion_vec)

    meta_explanation = explain_metadata(metadata_clf, meta_features, top_k=5)
    text_explanation = text_branch.top_tokens(body_text, top_k=5)

    explanation = _build_explanation(meta_explanation, text_explanation, fused_prob)

    return {
        "phishing_probability": round(fused_prob, 4),
        "branch_confidence": {
            "text": round(text_prob, 4),
            "metadata": round(metadata_prob, 4),
        },
        "top_metadata_features": meta_explanation,
        "top_text_tokens": [[tok, round(w, 4)] for tok, w in text_explanation],
        "explanation": explanation,
        "raw_metadata_features": meta_features.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description="Run the Email Detection Module on a raw email.")
    parser.add_argument("--eml", help="Path to an .eml file")
    parser.add_argument("--stdin", action="store_true", help="Read raw RFC822 email text from stdin")
    args = parser.parse_args()

    if args.eml:
        raw_email = Path(args.eml).read_text(encoding="utf-8", errors="ignore")
    elif args.stdin:
        raw_email = sys.stdin.read()
    else:
        parser.error("Provide --eml <path> or --stdin")
        return

    result = predict(raw_email)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
