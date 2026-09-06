import os
import joblib
import pandas as pd

from feature_extraction.url_feature_extractor import URLFeatureExtractor
from feature_extraction.host_features import get_host_features

# ==========================================
# Load Trained Model
# ==========================================

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model.pkl")
model = joblib.load(MODEL_PATH)


def extract_all_features(url: str) -> dict:
    """
    Builds the full feature dict for a URL: lexical features (fast, no
    network) + host/network features (DNS, SSL, domain age - live lookups,
    ~1-3 seconds). This must mirror what build_features.py does at
    training time, or the model sees a different feature distribution
    than it was trained on.
    """
    lexical_features = URLFeatureExtractor(url).extract()
    host_features = get_host_features(url)
    return {**lexical_features, **host_features}


def predict_url(url: str):
    """
    Predict whether a URL is phishing or legitimate.
    """

    # ======================================
    # Feature Extraction (lexical + host/network)
    # ======================================

    features = extract_all_features(url)

    X = pd.DataFrame([features])

    # ======================================
    # Match Training Feature Order
    # ======================================

    try:
        train_columns = model.feature_name_
        X = X.reindex(columns=train_columns, fill_value=0)
    except AttributeError:
        # Older versions of LightGBM may not expose feature_name_
        pass

    # ======================================
    # Prediction
    # ======================================

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    # Map probabilities to class labels safely
    class_prob = dict(zip(model.classes_, probabilities))

    # Your training data indicates:
    # 0 = Phishing
    # 1 = Legitimate

    phishing_probability = class_prob.get(0, 0.0)
    legitimate_probability = class_prob.get(1, 0.0)

    if prediction == 0:
        prediction_text = "Phishing"
        confidence = phishing_probability
    else:
        prediction_text = "Legitimate"
        confidence = legitimate_probability

    result = {
        "URL": url,
        "Prediction": prediction_text,
        "Risk Score": round(float(phishing_probability * 100), 2),
        "Confidence": round(float(confidence * 100), 2),
        "Legitimate Probability": round(float(legitimate_probability * 100), 2),
        "Phishing Probability": round(float(phishing_probability * 100), 2),
    }

    return result


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("Website URL Phishing Detection")
    print("=" * 60)

    url = input("\nEnter URL : ").strip()

    result = predict_url(url)

    print("\nPrediction Result")
    print("-" * 60)

    for key, value in result.items():
        print(f"{key:<28}: {value}")

    # ======================================
    # Debug Information (Temporary)
    # ======================================

    print("\nDebug Information")
    print("-" * 60)

    features = extract_all_features(url)

    X = pd.DataFrame([features])

    try:
        train_columns = model.feature_name_
        X = X.reindex(columns=train_columns, fill_value=0)
    except AttributeError:
        pass

    print("Model Classes :", model.classes_)
    print("Raw Prediction:", model.predict(X)[0])
    print("Probabilities :", model.predict_proba(X)[0])
    print("\nHost/Network Features:")
    for k in ["Domain_Age_Days", "Domain_Age_Unknown", "Domain_Age_Under_30_Days",
              "has_valid_cert", "cert_days_remaining", "cert_has_org_info",
              "has_mx_record", "ns_record_count"]:
        if k in features:
            print(f"  {k:<28}: {features[k]}")