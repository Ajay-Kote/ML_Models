import os
import joblib
import pandas as pd

from feature_extraction.url_feature_extractor import URLFeatureExtractor

# ==========================================
# Load Trained Model
# ==========================================

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model.pkl")
model = joblib.load(MODEL_PATH)


def predict_url(url: str):
    """
    Predict whether a URL is phishing or legitimate.
    """

    # ======================================
    # Feature Extraction
    # ======================================

    extractor = URLFeatureExtractor(url)

    features = extractor.extract()

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

    extractor = URLFeatureExtractor(url)
    features = extractor.extract()

    X = pd.DataFrame([features])

    try:
        train_columns = model.feature_name_
        X = X.reindex(columns=train_columns, fill_value=0)
    except AttributeError:
        pass

    print("Model Classes :", model.classes_)
    print("Raw Prediction:", model.predict(X)[0])
    print("Probabilities :", model.predict_proba(X)[0])