"""
Test the phishing URL detection model against labeled data and known URL patterns.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from models.predict import predict_url


def prediction_to_label(prediction_text: str) -> int:
    return 1 if prediction_text == "Legitimate" else 0


def test_dataset_sample(n_samples: int = 500):
    """Test on URLs from the held-out test split (same random_state as training)."""
    raw = pd.read_csv("data/raw/PhiUSIIL_Phishing_URL_Dataset.csv")
    label_col = "label" if "label" in raw.columns else "Label"

    _, test_df = train_test_split(
        raw, test_size=0.20, random_state=42, stratify=raw[label_col]
    )
    test_df = test_df.sample(n=min(n_samples, len(test_df)), random_state=42)

    correct = 0
    wrong = []

    for _, row in test_df.iterrows():
        url = row["URL"]
        actual = int(row[label_col])
        result = predict_url(url)
        predicted = prediction_to_label(result["Prediction"])

        if predicted == actual:
            correct += 1
        else:
            wrong.append(
                {
                    "url": url,
                    "actual": "Legitimate" if actual == 1 else "Phishing",
                    "predicted": result["Prediction"],
                    "risk": result["Risk Score"],
                }
            )

    accuracy = correct / len(test_df) * 100

    print("=" * 70)
    print(f"DATASET TEST ({len(test_df)} URLs from held-out test split)")
    print("=" * 70)
    print(f"Accuracy     : {correct}/{len(test_df)} = {accuracy:.2f}%")
    print(f"Misclassified: {len(wrong)}")

    if wrong:
        print("\nMisclassified examples (up to 10):")
        print("-" * 70)
        for w in wrong[:10]:
            url = w["url"][:65] + "..." if len(w["url"]) > 68 else w["url"]
            print(f"  Actual: {w['actual']:<12} Predicted: {w['predicted']:<12} Risk: {w['risk']}%")
            print(f"  URL: {url}")
            print()

    return accuracy, len(wrong)


def test_manual_cases():
    """Spot-check obvious legitimate and suspicious URL patterns."""
    cases = [
        ("https://www.google.com", "Legitimate"),
        ("https://www.github.com", "Legitimate"),
        ("https://www.amazon.com", "Legitimate"),
        ("https://www.microsoft.com", "Legitimate"),
        ("https://www.uni-mainz.de", "Legitimate"),
        ("http://192.168.1.1/login/verify-account", "Phishing"),
        ("http://secure-paypal-login.xyz/verify", "Phishing"),
        ("http://google-security-update.tk/signin", "Phishing"),
        ("http://bit.ly/fake-bank-login", "Phishing"),
        ("http://paypal-secure-update.ml/account/verify", "Phishing"),
    ]

    print("=" * 70)
    print("MANUAL SPOT-CHECK (known URL patterns)")
    print("=" * 70)
    print(f"{'URL':<42} {'Expected':<12} {'Predicted':<12} {'Risk%':>7} {'Pass':>5}")
    print("-" * 70)

    passed = 0
    for url, expected in cases:
        result = predict_url(url)
        predicted = result["Prediction"]
        ok = predicted == expected
        passed += ok
        short = url[:40] + ".." if len(url) > 42 else url
        status = "YES" if ok else "NO"
        print(
            f"{short:<42} {expected:<12} {predicted:<12} "
            f"{result['Risk Score']:>7.1f} {status:>5}"
        )

    print("-" * 70)
    print(f"Passed: {passed}/{len(cases)}")
    print()

    return passed, len(cases)


def test_class_balance():
    """Verify model handles both classes — not always predicting one label."""
    samples = [
        "https://www.google.com",
        "http://secure-paypal-login.xyz/verify",
    ]
    predictions = {predict_url(u)["Prediction"] for u in samples}

    print("=" * 70)
    print("CLASS BALANCE CHECK")
    print("=" * 70)
    if len(predictions) == 2:
        print("PASS: Model produces both 'Legitimate' and 'Phishing' predictions.")
    else:
        print(f"FAIL: Model only predicted: {predictions}")
    print()

    return len(predictions) == 2


if __name__ == "__main__":
    manual_pass, manual_total = test_manual_cases()
    balanced = test_class_balance()
    accuracy, errors = test_dataset_sample(n_samples=500)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Manual spot-check : {manual_pass}/{manual_total} passed")
    print(f"Class balance     : {'PASS' if balanced else 'FAIL'}")
    print(f"Dataset accuracy  : {accuracy:.2f}% ({errors} errors on 500 samples)")
    print()

    all_ok = manual_pass == manual_total and balanced and accuracy >= 90
    if all_ok:
        print("Overall: MODEL IS WORKING CORRECTLY")
    elif accuracy >= 90:
        print("Overall: MODEL IS MOSTLY CORRECT (check manual cases above)")
    else:
        print("Overall: MODEL MAY HAVE ISSUES — review misclassifications above")
