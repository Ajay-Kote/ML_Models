import random
import pandas as pd
from feature_extraction.url_feature_extractor import URLFeatureExtractor
from feature_extraction.host_features import get_host_features_bulk


INPUT_FILE = "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
OUTPUT_FILE = "data/processed/features.csv"

RANDOM_STATE = 42

# The raw dataset has 235k+ rows. Running live network lookups (WHOIS/DNS/SSL)
# on that many unique domains would take hours and risks WHOIS rate-limiting
# (which would silently poison the data with -1/"unknown" values). We instead
# take a balanced sample BEFORE augmentation - this keeps network lookup time
# to ~20-30 minutes while still giving LightGBM more than enough data.
SAMPLE_SIZE_PER_CLASS = 15000  # -> ~30,000 raw rows before augmentation

# In the raw dataset every single legitimate URL is a bare "www." homepage
# with no path, while ~27% of phishing URLs have a path. A model trained on
# that will just learn "has a path -> phishing" and "no www -> phishing",
# instead of anything about actual phishing structure. These augmentations
# add legitimate deep-link and apex-domain examples so that signal goes away.
GENERIC_PATHS = [
    "about", "contact", "pricing", "careers", "faq", "team",
    "products/123", "blog/2024/update", "docs/getting-started",
    "news/latest", "gallery", "events/2024", "download",
    "terms", "privacy-policy", "search?q=example", "sitemap.xml",
]

LOGIN_STYLE_PATHS = ["login", "account", "signin", "user/settings"]

BRAND_DOMAINS = {
    "google": "google.com", "paypal": "paypal.com", "amazon": "amazon.com",
    "facebook": "facebook.com", "apple": "apple.com", "microsoft": "microsoft.com",
    "github": "github.com", "netflix": "netflix.com", "instagram": "instagram.com",
    "linkedin": "linkedin.com", "discord": "discord.com",
    "steampowered": "steampowered.com",
}
BRAND_SUBDOMAINS = ["", "www", "accounts", "mail", "help", "support", "docs"]
PER_BRAND_SAMPLES = 400

# Well-known real, non-brand-keyword websites that are commonly accessed as
# a bare apex domain + a single simple path segment (e.g. leetcode.com/problemset/,
# github.com/trending). This exact shape - apex domain, no subdomain, short
# clean path - is ALSO the shape of our synthetic typosquat phishing URLs
# (e.g. paypa1.tk/login). Without real legitimate examples of this shape,
# the model learns "apex + short path -> phishing", which misfires on
# ordinary sites like LeetCode. These add that missing signal.
WELL_KNOWN_APEX_DOMAINS = [
    "leetcode.com", "github.com", "stackoverflow.com", "wikipedia.org",
    "reddit.com", "medium.com", "quora.com", "imdb.com", "spotify.com",
    "twitch.tv", "producthunt.com", "npmjs.com", "pypi.org", "kaggle.com",
    "geeksforgeeks.org", "w3schools.com", "coursera.org", "udemy.com",
    "khanacademy.org", "codecademy.com",
]
WELL_KNOWN_SIMPLE_PATHS = [
    "problemset", "trending", "questions", "explore", "learn", "about",
    "pricing", "docs", "blog", "help", "search", "discover", "courses",
    "profile", "dashboard", "settings", "notifications", "leaderboard",
]
PER_WELL_KNOWN_SAMPLES = 300


def build_well_known_apex_legit_urls(rng):
    """
    Real sites, accessed as apex domain (no www/subdomain) with a single
    short path segment - the same URL shape as our synthetic phishing
    typosquat examples, but genuinely legitimate.
    """
    urls = []
    for domain in WELL_KNOWN_APEX_DOMAINS:
        for _ in range(PER_WELL_KNOWN_SAMPLES):
            if rng.random() < 0.3:
                url = f"https://{domain}"
            else:
                path = rng.choice(WELL_KNOWN_SIMPLE_PATHS)
                url = f"https://{domain}/{path}/" if rng.random() < 0.5 else f"https://{domain}/{path}"
            urls.append(url)
    return urls


def _apex(url):
    return url.replace("://www.", "://", 1) if "://www." in url else url


PATH_WORDS = [
    "news", "world", "technology", "business", "sports", "article",
    "category", "tutorial", "guide", "reference", "overview", "topic",
    "section", "archive", "post", "story", "release", "update-notes",
]
PATH_EXTENSIONS = ["", "", "", ".html", ".php", ".aspx"]


def _random_deep_path(rng):
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

    # -----------------------------------------------------
    # Sample down to a manageable size (balanced across classes)
    # before doing anything else, so downstream augmentation +
    # network lookups stay fast.
    # -----------------------------------------------------
    if "Label" in df.columns or "label" in df.columns:
        _label_col = "Label" if "Label" in df.columns else "label"
        sampled_parts = []
        for cls, group in df.groupby(_label_col):
            n = min(SAMPLE_SIZE_PER_CLASS, len(group))
            sampled_parts.append(group.sample(n=n, random_state=RANDOM_STATE))
        df = pd.concat(sampled_parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"Sampled down to {len(df)} rows ({SAMPLE_SIZE_PER_CLASS} per class, capped by availability)")

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
        well_known_pool = build_well_known_apex_legit_urls(rng)
        typosquat_pool = build_typosquat_phishing_urls(rng)
        phishing_urls = df.loc[df[label_col] == 0, "URL"].tolist()

        urls = legit_pool + brand_pool + well_known_pool + typosquat_pool + phishing_urls
        labels = (
            [1] * (len(legit_pool) + len(brand_pool) + len(well_known_pool))
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

    urls = list(urls)

    # -----------------------------------------------------
    # STEP 1: Lexical features (fast, no network, all rows)
    # -----------------------------------------------------
    print("Extracting lexical features...")

    lexical_rows = []
    for i, url in enumerate(urls):
        extractor = URLFeatureExtractor(url)
        lexical_rows.append(extractor.extract())
        if (i + 1) % 5000 == 0:
            print(f"  lexical: {i+1}/{len(urls)} URLs processed")

    # -----------------------------------------------------
    # STEP 2: Host/network features (slow, deduplicated by
    # hostname, run in parallel across unique domains only -
    # e.g. the 400x repeated paypal.com / typo-brand domains
    # each only get looked up once).
    # -----------------------------------------------------
    print("Looking up host/network features (DNS, SSL, domain age)...")
    host_features_by_url = get_host_features_bulk(urls, max_workers=40)

    # -----------------------------------------------------
    # STEP 3: Merge lexical + host features per row
    # -----------------------------------------------------
    rows = []
    for i, url in enumerate(urls):
        features = {**lexical_rows[i], **host_features_by_url[url]}
        if labels[i] is not None:
            features["Label"] = labels[i]
        rows.append(features)

    feature_df = pd.DataFrame(rows)

    feature_df.to_csv(OUTPUT_FILE, index=False)

    print("\nFeature Extraction Completed Successfully.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()