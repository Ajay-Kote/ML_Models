"""
generate_dataset.py
--------------------
Synthesizes a labeled QR-code image dataset for the Quishing Detection module.

Why synthetic:
No public malicious-QR image dataset exists (see Section 10 of the design doc),
so this script builds one by:
  1. Sampling legitimate-looking and phishing-like URLs (reusing the same
     lexical patterns the URL module treats as risky: IP-based hosts, long
     subdomains, high-risk TLDs, brand-keyword stuffing, etc.)
  2. Rendering each URL as a real QR image with `qrcode`, varying generation
     parameters (version, error-correction level, box size, border/quiet-zone)
     in a way that correlates with the label -- malicious QR generators in
     the wild frequently use minimal error-correction and irregular module
     density to cram more payload into a small code.
  3. Persisting both the PNG and a metadata.jsonl with the ground-truth label
     and the *generation* parameters (version / error-correction / box size /
     border), which doubles as the "QR metadata" feature source described in
     Section 5.3 of the design document.

Usage:
    python generate_dataset.py --n_per_class 600
"""
import argparse
import json
import os
import random
import string

import qrcode
from qrcode.constants import ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H

RNG = random.Random(42)

LEGIT_DOMAINS = [
    "github.com", "wikipedia.org", "google.com", "apple.com", "amazon.com",
    "nytimes.com", "microsoft.com", "spotify.com", "reddit.com", "stackoverflow.com",
    "who.int", "un.org", "mit.edu", "bbc.com", "npr.org",
]

HIGH_RISK_TLDS = [".zip", ".top", ".xyz", ".click", ".gq", ".tk", ".cf", ".loan", ".fit"]
BRAND_WORDS = ["paypal", "bankofamerica", "netflix", "amazon", "appleid", "microsoft365", "hdfcbank", "upi"]


def _rand_str(n):
    return "".join(RNG.choices(string.ascii_lowercase + string.digits, k=n))


def make_legit_url():
    domain = RNG.choice(LEGIT_DOMAINS)
    path = "/" + "/".join(_rand_str(RNG.randint(3, 8)) for _ in range(RNG.randint(0, 2)))
    return f"https://{domain}{path}"


def make_malicious_url():
    style = RNG.choice(["ip_host", "brand_subdomain", "high_risk_tld", "long_random", "punycode_like"])
    if style == "ip_host":
        ip = ".".join(str(RNG.randint(1, 255)) for _ in range(4))
        return f"http://{ip}/{_rand_str(10)}/{RNG.choice(BRAND_WORDS)}-verify"
    if style == "brand_subdomain":
        brand = RNG.choice(BRAND_WORDS)
        return f"http://{brand}-secure-{_rand_str(6)}.{_rand_str(8)}.com/login"
    if style == "high_risk_tld":
        return f"http://{RNG.choice(BRAND_WORDS)}-update{_rand_str(4)}{RNG.choice(HIGH_RISK_TLDS)}/verify"
    if style == "long_random":
        return "http://" + _rand_str(40) + ".com/" + _rand_str(20)
    # punycode_like homograph attempt
    return f"http://xn--{_rand_str(12)}.com/{RNG.choice(BRAND_WORDS)}"


def generation_params(label):
    """
    Choose QR generation parameters. Malicious samples skew toward lower
    error-correction (attackers maximize payload density) and tighter quiet
    zones (fit more into a small printed/screenshotted code); legitimate
    samples skew toward safer, more standard defaults.
    """
    if label == 1:  # malicious
        ec = RNG.choices([ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H],
                          weights=[0.55, 0.30, 0.10, 0.05])[0]
        box_size = RNG.choice([4, 5, 6, 10, 11, 12])   # more variable / extreme
        border = RNG.choice([1, 1, 2])                  # tight quiet zone
    else:  # legitimate
        ec = RNG.choices([ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H],
                          weights=[0.15, 0.45, 0.30, 0.10])[0]
        box_size = RNG.choice([7, 8, 9])
        border = RNG.choice([4, 4, 5])
    return ec, box_size, border


def make_qr(url, ec, box_size, border, out_path):
    qr = qrcode.QRCode(error_correction=ec, box_size=box_size, border=border)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("L")
    img.save(out_path)
    return qr.version


EC_NAME = {ERROR_CORRECT_L: "L", ERROR_CORRECT_M: "M", ERROR_CORRECT_Q: "Q", ERROR_CORRECT_H: "H"}


def main(n_per_class, out_dir):
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "metadata.jsonl")

    rows = []
    idx = 0
    for label, maker in ((0, make_legit_url), (1, make_malicious_url)):
        for _ in range(n_per_class):
            url = maker()
            ec, box_size, border = generation_params(label)
            fname = f"qr_{idx:05d}.png"
            fpath = os.path.join(img_dir, fname)
            version = make_qr(url, ec, box_size, border, fpath)
            rows.append({
                "file": fname,
                "url": url,
                "label": label,  # 1 = malicious/quishing, 0 = legitimate
                "qr_version": version,
                "error_correction": EC_NAME[ec],
                "box_size": box_size,
                "border": border,
            })
            idx += 1

    with open(manifest_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"Generated {len(rows)} QR images -> {img_dir}")
    print(f"Manifest written -> {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_class", type=int, default=600)
    parser.add_argument("--out_dir", type=str, default=os.path.join(os.path.dirname(__file__), "raw"))
    args = parser.parse_args()
    main(args.n_per_class, args.out_dir)
