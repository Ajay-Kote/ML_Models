"""
generate_synthetic_dataset.py

Builds a synthetic multi-modal training set for the fusion meta-learner.

WHY SYNTHETIC:
Each module (URL / SMS / QR / Image / Email) was trained on its own
independent dataset. There is no real-world dataset where the SAME case
has ground-truth labels across all 5 channels at once. So instead of
faking that, we simulate REQUESTS: each synthetic row represents one
"incident" where anywhere from 1 to 5 modules ran, using each module's
OWN real risk_probability distribution (pulled from its test-set
predictions) to keep the numbers realistic.

HOW A ROW IS LABELED:
overall_malicious = 1 if the incident is fraudulent, else 0.
Given that ground truth, each present module's risk_probability is drawn
from that module's real "malicious" or "legitimate" score distribution
(so a malicious incident tends to produce high risk_probability from
present modules, but with realistic noise/disagreement -- exactly the
kind of messy signal the meta-learner needs to learn to weigh).

OUTPUT FEATURES (10 total, matches what fuse.py will build per request):
  url_risk, sms_risk, qr_risk, image_risk, email_risk        -> 0.5 if module absent (neutral/unknown)
  url_present, sms_present, qr_present, image_present, email_present -> 0/1

Replace load_module_score_distributions() with real numbers pulled from
each module's saved test-set predictions before running this for real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

MODULES = ["url", "sms", "qr", "image", "email"]

# ---------------------------------------------------------------------------
# Per-module score distributions, conditioned on the TRUE label of the
# incident. Pulled from each module's real held-out evaluation:
#   - "malicious" params: (mean, std) of risk_probability when the
#     underlying case IS malicious/fraudulent
#   - "legitimate" params: (mean, std) of risk_probability when it's NOT
#
# url: recomputed 27-08-2026 after url module retrain (network features +
# apex-domain MX/NS/WHOIS fix) via compute_score_distribution.py.
# ---------------------------------------------------------------------------
SCORE_DIST = {
    "url":   {"malicious": (0.9708, 0.1335), "legitimate": (0.0172, 0.0746)},
    "sms":   {"malicious": (0.9665, 0.1593), "legitimate": (0.0097, 0.0780)},
    "qr":    {"malicious": (0.8122, 0.2303), "legitimate": (0.2107, 0.2853)},
    "image": {"malicious": (0.8610, 0.1318), "legitimate": (0.6914, 0.1866)},
    "email": {"malicious": (0.9871, 0.0918), "legitimate": (0.0512, 0.1639)},
}

# Probability that each module is even present in a given synthetic incident
# (in real traffic not every channel is always available for a case).
MODULE_PRESENCE_PROB = {
    "url": 0.55, "sms": 0.45, "qr": 0.30, "image": 0.35, "email": 0.50,
}


def _sample_score(module: str, is_malicious: bool) -> float:
    key = "malicious" if is_malicious else "legitimate"
    mean, std = SCORE_DIST[module][key]
    val = RNG.normal(mean, std)
    return float(np.clip(val, 0.0, 1.0))


def generate(n_rows: int = 5000, malicious_ratio: float = 0.45) -> pd.DataFrame:
    rows = []
    for _ in range(n_rows):
        is_malicious = RNG.random() < malicious_ratio

        # Decide which modules are present for this incident. Force at
        # least one present so every row is usable.
        present = {m: RNG.random() < MODULE_PRESENCE_PROB[m] for m in MODULES}
        if not any(present.values()):
            present[RNG.choice(MODULES)] = True

        row = {}
        for m in MODULES:
            if present[m]:
                row[f"{m}_risk"] = _sample_score(m, is_malicious)
                row[f"{m}_present"] = 1
            else:
                row[f"{m}_risk"] = 0.5   # neutral placeholder, ignored via *_present flag
                row[f"{m}_present"] = 0

        row["label"] = int(is_malicious)
        rows.append(row)

    cols = [f"{m}_risk" for m in MODULES] + [f"{m}_present" for m in MODULES] + ["label"]
    return pd.DataFrame(rows, columns=cols)


if __name__ == "__main__":
    df = generate(n_rows=5000)
    df.to_csv("synthetic_fusion_dataset.csv", index=False)
    print(f"Generated {len(df)} rows -> synthetic_fusion_dataset.csv")
    print(df["label"].value_counts(normalize=True))
    print(df.head())