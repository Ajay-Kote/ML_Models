"""
lime_explainer.py
------------------
Implements the Explainability Layer requirement for the SMS module
(Section 7 of the design doc): LIME (text explainer) -> "which words/
phrases drove the classification".

Usage (CLI):
    python lime_explainer.py "Congratulations! You won a free prize, claim now at bit.ly/xyz"

Usage (import):
    from lime_explainer import explain
    result = explain("some sms text")
"""
import os
import sys
import numpy as np
from lime.lime_text import LimeTextExplainer

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model"))
from predict import SmishingDetector  # noqa: E402

CLASS_NAMES = ["legitimate", "smishing"]


def _predict_proba_fn(detector: SmishingDetector):
    """Wraps the detector so LIME can call model(list_of_texts) -> (n, 2) probs."""

    def predict_proba(texts):
        probs = []
        for t in texts:
            r = detector.predict(t)
            p_smishing = r["smishing_probability"]
            probs.append([1 - p_smishing, p_smishing])
        return np.array(probs)

    return predict_proba


# NOTE: this file depends on model/predict.py and will only work once you've
# run train.py and have a saved model in model/saved_model/.



def explain(text: str, num_features: int = 8, detector: SmishingDetector = None) -> dict:
    """
    Returns a human-readable explanation:
        {
          "text": str,
          "label": "smishing" | "legitimate",
          "probability": float,
          "top_features": [(word, weight), ...],   # weight > 0 pushes toward smishing
          "explanation_summary": str
        }
    """
    detector = detector or SmishingDetector()
    explainer = LimeTextExplainer(class_names=CLASS_NAMES)

    prediction = detector.predict(text)
    exp = explainer.explain_instance(
        text,
        _predict_proba_fn(detector),
        num_features=num_features,
        labels=(1,),
    )
    top_features = exp.as_list(label=1)

    risk_words = [w for w, weight in top_features if weight > 0]
    if risk_words:
        summary = (
            f"Flagged as {prediction['label']} primarily due to: "
            + ", ".join(f'"{w}"' for w in risk_words[:5])
        )
    else:
        summary = f"Classified as {prediction['label']}; no strong individual risk words found."

    return {
        "text": text,
        "label": prediction["label"],
        "probability": prediction["smishing_probability"],
        "top_features": top_features,
        "explanation_summary": summary,
    }


def save_html_explanation(text: str, out_path: str = "explanation.html", detector: SmishingDetector = None):
    """Saves LIME's interactive HTML explanation (word highlighting) to disk."""
    detector = detector or SmishingDetector()
    explainer = LimeTextExplainer(class_names=CLASS_NAMES)
    exp = explainer.explain_instance(text, _predict_proba_fn(detector), num_features=8, labels=(1,))
    exp.save_to_file(out_path)
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python lime_explainer.py "<sms text>"')
        sys.exit(1)
    result = explain(sys.argv[1])
    print(f"Text       : {result['text']}")
    print(f"Label      : {result['label']}  (prob={result['probability']})")
    print("Top features (word, weight -- positive = pushes toward smishing):")
    for word, weight in result["top_features"]:
        print(f"  {word:20s} {weight:+.4f}")
    print(f"\nSummary: {result['explanation_summary']}")
