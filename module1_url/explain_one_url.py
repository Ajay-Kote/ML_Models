"""
explain_one_url.py
Quick diagnostic: shows which features pushed a specific URL's
prediction toward Phishing vs Legitimate. Run from the project root.

Usage:
    python explain_one_url.py "https://leetcode.com/problemset/"
"""

import sys
import joblib
import pandas as pd
import shap

from feature_extraction.url_feature_extractor import URLFeatureExtractor
from feature_extraction.host_features import get_host_features


def main():
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter URL: ").strip()

    model = joblib.load("models/saved_model.pkl")

    lexical = URLFeatureExtractor(url).extract()
    host = get_host_features(url)
    features = {**lexical, **host}

    X = pd.DataFrame([features])

    try:
        train_columns = model.feature_name_
        X = X.reindex(columns=train_columns, fill_value=0)
    except AttributeError:
        pass

    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    print(f"\nURL: {url}")
    print(f"Prediction: {'Phishing' if prediction == 0 else 'Legitimate'}")
    print(f"Phishing probability: {proba[0]*100:.2f}%")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    values = shap_values.values[0]
    base_value = shap_values.base_values[0]

    if values.ndim == 2:
        # (n_features, n_classes) -> take the column matching class index 1
        values = values[:, 1] if values.shape[1] > 1 else values[:, 0]
        base_value = base_value[1] if hasattr(base_value, "__len__") else base_value

    # Don't assume which class the raw shap output aligns to - verify it
    # empirically against predict_proba (ground truth) instead of guessing.
    import math
    raw_margin = base_value + values.sum()
    reconstructed_prob_class1 = 1 / (1 + math.exp(-raw_margin))
    actual_prob_class1 = proba[1]  # Legitimate, since model.classes_ == [0, 1]

    # If reconstructed prob matches actual P(Legitimate), then positive shap = pushes toward Legitimate.
    # If it matches actual P(Phishing) instead, positive shap = pushes toward Phishing.
    matches_legit = abs(reconstructed_prob_class1 - actual_prob_class1) < 0.01
    matches_phish = abs(reconstructed_prob_class1 - proba[0]) < 0.01

    if matches_legit:
        positive_means = "LEGITIMATE"
        negative_means = "PHISHING"
    elif matches_phish:
        positive_means = "PHISHING"
        negative_means = "LEGITIMATE"
    else:
        # Fallback: couldn't verify cleanly, state both possibilities
        positive_means = "LEGITIMATE (best guess, unverified)"
        negative_means = "PHISHING (best guess, unverified)"

    names = X.columns.tolist()
    result = sorted(zip(names, values), key=lambda x: abs(x[1]), reverse=True)

    print(f"\n(Verified: positive shap -> pushes toward {positive_means})")
    print("\nTop 15 features driving this prediction")
    print("-" * 65)
    for feature, value in result[:15]:
        raw_value = features.get(feature, "N/A")
        direction = f"-> {positive_means}" if value > 0 else f"-> {negative_means}"
        print(f"{feature:<28} shap={value:+.4f}  {direction:<28} (value={raw_value})")


if __name__ == "__main__":
    main()