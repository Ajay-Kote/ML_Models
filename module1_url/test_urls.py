"""Quick URL prediction demo."""

from models.predict import predict_url

URLS = [
    # Legitimate
    "https://www.google.com",
    "https://www.flipkart.com/",
    "https://www.microsoft.com",
    "https://www.facebook.com",
    "https://www.wikipedia.org",
    "https://www.stackoverflow.com",
    "https://github.com/login",
    # Suspicious / phishing-like
    "http://192.168.1.1/login/verify-account",
    "http://secure-paypal-login.xyz/verify",
    "http://google-security-update.tk/signin",
    "http://amazon-account-update.cf/confirm",
    "http://bit.ly/fake-bank-login",
    "http://microsoft-login-support.top/auth",
    "http://paypal-secure-update.ml/account/verify",
]

print("=" * 80)
print("URL PHISHING DETECTION — LIVE TEST")
print("=" * 80)
print(f"{'URL':<48} {'Result':<12} {'Risk%':>7} {'Confidence':>11}")
print("-" * 80)

for url in URLS:
    r = predict_url(url)
    short = url[:46] + ".." if len(url) > 48 else url
    print(
        f"{short:<48} {r['Prediction']:<12} {r['Risk Score']:>7.1f} {r['Confidence']:>10.1f}%"
    )

print("-" * 80)
print("Risk% = probability of phishing  |  Confidence = model certainty in its prediction")
