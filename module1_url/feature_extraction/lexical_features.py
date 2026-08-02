import random
import pandas as pd
from feature_extraction.url_feature_extractor import URLFeatureExtractor


INPUT_FILE = "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
OUTPUT_FILE = "data/processed/features.csv"

RANDOM_STATE = 42

# In the raw dataset every single legitimate URL is a bare "www." homepage
# with no path, while ~27% of phishing URLs have a path. A model trained on
# that will just learn "has a path -> phishing" and "no www -> phishing",
# instead of anything about actual phishing structure. These augmentations
# add legitimate deep-link and apex-domain examples so that signal goes away.
# Ordinary pages a real site has. Deliberately does NOT overlap with
# URLFeatureExtractor.SUSPICIOUS_KEYWORDS (login/signin/verify/secure/
# update/account/password/bank/payment/wallet/crypto/bitcoin/invoice/
# admin/support/confirm/auth) - reusing those words here would teach the
# model that suspicious keywords make a URL MORE legitimate, inverting
# a signal that's actually useful.
GENERIC_PATHS = [
    "about", "contact", "pricing", "careers", "faq", "team",
    "products/123", "blog/2024/update", "docs/getting-started",
    "news/latest", "gallery", "events/2024", "download",
    "terms", "privacy-policy", "search?q=example", "sitemap.xml",
]

# A small, realistic minority of pages on a real site (e.g. github.com/login
# is a genuine legitimate page) - kept rare so the keyword signal isn't lost.
LOGIN_STYLE_PATHS = ["login", "account", "signin", "user/settings"]

# The raw PhiUSIIL dataset contains zero examples of these domains, in
# either class - every brand-keyword occurrence it has ever seen comes
# from phishing impersonation, so the model learns "brand keyword anywhere
# -> phishing" and flags the real sites too. Inject real examples so it can
# separate "is the brand's own domain" from "brand name stuffed elsewhere".
BRAND_DOMAINS = {
    "google": "google.com", "paypal": "paypal.com", "amazon": "amazon.com",
    "facebook": "facebook.com", "apple": "apple.com", "microsoft": "microsoft.com",
    "github": "github.com", "netflix": "netflix.com", "instagram": "instagram.com",
    "linkedin": "linkedin.com", "discord": "discord.com",
    "steampowered": "steampowered.com",
}
BRAND_SUBDOMAINS = ["", "www", "accounts", "mail", "help", "support", "docs"]
PER_BRAND_SAMPLES = 400


def _apex(url):
    return url.replace("://www.", "://", 1) if "://www." in url else url


PATH_WORDS = [
    "news", "world", "technology", "business", "sports", "article",
    "category", "tutorial", "guide", "reference", "overview", "topic",
    "section", "archive", "post", "story", "release", "update-notes",
]
PATH_EXTENSIONS = ["", "", "", ".html", ".php", ".aspx"]


def _random_deep_path(rng):
    """
    Real legitimate pages have multi-segment paths with numeric IDs,
    hyphenated slugs, and file extensions - not just single clean words.
    """
    depth = rng.randint(1, 4)
    segments = []

    for _ in range(depth):
        kind = rng.random()
        if kind < 0.4:
            segments.append(rng.choice(PATH_WORDS))
        elif kind < 0.7:
            segments.append(str(rng.randint(1, 99999)))
        else:
            segments.append("-".join(rng.sample(PATH_WORDS, k=rng.randint(2, 3))))

    path = "/".join(segments) + rng.choice(PATH_EXTENSIONS)

    if rng.random() < 0.2:
        path += "?" + rng.choice(["id=123", "ref=home", "page=2", "utm_source=x"])

    return path


SUSPICIOUS_PATHS = [
    "login", "signin", "verify", "secure", "update", "account",
    "password", "confirm", "auth", "wallet/connect",
]
RISKY_TLDS = ["com", "com", "com", "tk", "top", "xyz", "support", "click"]


def _typo(word, rng):
    i = rng.randrange(len(word))
    kind = rng.choice(["delete", "insert", "substitute", "swap"])

    if kind == "delete":
        return word[:i] + word[i + 1:]
    if kind == "insert":
        return word[:i] + rng.choice("abcdefghijklmnopqrstuvwxyz") + word[i:]
    if kind == "substitute":
        return word[:i] + rng.choice("abcdefghijklmnopqrstuvwxyz") + word[i + 1:]
    j = min(i + 1, len(word) - 1)
    chars = list(word)
    chars[i], chars[j] = chars[j], chars[i]
    return "".join(chars)


def build_typosquat_phishing_urls(rng):
    urls = []

    for brand in BRAND_DOMAINS:
        for _ in range(PER_BRAND_SAMPLES):
            typo_domain = _typo(brand, rng)
            if typo_domain == brand:
                continue

            tld = rng.choice(RISKY_TLDS)
            path = rng.choice([""] + SUSPICIOUS_PATHS)
            url = f"https://{typo_domain}.{tld}" + (f"/{path}" if path else "")
            urls.append(url)

    return urls


def build_brand_legit_urls(rng):
    urls = []

    for domain in BRAND_DOMAINS.values():
        for _ in range(PER_BRAND_SAMPLES):
            sub = rng.choice(BRAND_SUBDOMAINS)
            host = f"{sub}.{domain}" if sub else domain

            path = rng.choices(
                [
                    "",
                    rng.choice(LOGIN_STYLE_PATHS),
                    rng.choice(GENERIC_PATHS),
                    _random_deep_path(rng),
                ],
                weights=[0.25, 0.1, 0.2, 0.45],
            )[0]
            url = f"https://{host}" + (f"/{path}" if path else "")
            urls.append(url)

    return urls


def augment_legitimate_urls(urls, rng, target_size):
    """
    Build a legitimate-class training pool where "has a path" and "has www"
    are no longer near-perfect proxies for the label: ~40% bare homepage,
    ~20% apex (non-www) homepage, ~40% with a path (occasionally a login-
    style path, mostly ordinary pages).
    """
    augmented = []

    while len(augmented) < target_size:
        url = rng.choice(urls)
        roll = rng.random()

        if roll < 0.40:
            augmented.append(url)
        elif roll < 0.60:
            augmented.append(_apex(url))
        else:
            base = rng.choice([url, _apex(url)])
            path = rng.choices(
                [
                    rng.choice(LOGIN_STYLE_PATHS),
                    rng.choice(GENERIC_PATHS),
                    _random_deep_path(rng),
                ],
                weights=[0.1, 0.3, 0.6],
            )[0]
            augmented.append(base.rstrip("/") + "/" + path)

    return augmented


def main():

    rng = random.Random(RANDOM_STATE)

    print("Loading Dataset...")

    df = pd.read_csv(INPUT_FILE)

    if "Label" in df.columns:
        label_col = "Label"
    elif "label" in df.columns:
        label_col = "label"
    else:
        label_col = None

    if label_col is not None:
        print("Augmenting legitimate URLs to remove path/www bias...")

        legit_urls = df.loc[df[label_col] == 1, "URL"].tolist()
        phishing_count = int((df[label_col] == 0).sum())

        legit_pool = augment_legitimate_urls(legit_urls, rng, phishing_count)
        brand_pool = build_brand_legit_urls(rng)
        typosquat_pool = build_typosquat_phishing_urls(rng)
        phishing_urls = df.loc[df[label_col] == 0, "URL"].tolist()

        urls = legit_pool + brand_pool + typosquat_pool + phishing_urls
        labels = (
            [1] * (len(legit_pool) + len(brand_pool))
            + [0] * (len(typosquat_pool) + len(phishing_urls))
        )

        combined = list(zip(urls, labels))
        rng.shuffle(combined)
        urls, labels = zip(*combined)

        print(f"Final dataset: {len(urls)} URLs "
              f"({labels.count(1)} legitimate / {labels.count(0)} phishing)")
    else:
        urls = df["URL"].tolist()
        labels = [None] * len(urls)

    print("Extracting Features...")

    rows = []

    for i, url in enumerate(urls):

        extractor = URLFeatureExtractor(url)

        features = extractor.extract()

        if labels[i] is not None:
            features["Label"] = labels[i]

        rows.append(features)

        if (i + 1) % 5000 == 0:
            print(f"{i+1}/{len(urls)} URLs processed")

    feature_df = pd.DataFrame(rows)

    feature_df.to_csv(OUTPUT_FILE, index=False)

    print("\nFeature Extraction Completed Successfully.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()