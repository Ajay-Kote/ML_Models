"""
fusion_engine/fuse.py

Combines the (already-normalized) outputs of any subset of the 5 detection
modules into one final risk score + verdict.

Approach: weighted average of risk_probability across whichever modules ran,
plus a rule-based override -- if any single module is extremely confident
(>= OVERRIDE_THRESHOLD) that something is malicious, that overrides the
weighted average. This mirrors how real fraud systems work: a single very
strong signal (e.g. a known-malicious URL) shouldn't get diluted just
because other channels look clean.

Usage:
    from adapters import adapt_url, adapt_sms
    from fuse import fuse

    normalized = {
        "url": adapt_url(raw_url_output),
        "sms": adapt_sms(raw_sms_output),
    }
    result = fuse(normalized)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

# Relative importance of each module when combining scores. These only need
# to be relative to each other -- they get renormalized to sum to 1 based on
# whichever modules are actually present in a given request (see
# _renormalize_weights below).
DEFAULT_WEIGHTS = {
    "url": 0.25,
    "sms": 0.20,
    "qr": 0.15,
    "image": 0.20,
    "email": 0.20,
}

# If any single module reports risk_probability >= this, override the
# weighted average and mark the whole request high-risk regardless of what
# the other modules say.
OVERRIDE_THRESHOLD = 0.90

# Verdict buckets for the final combined score.
SAFE_MAX = 0.30          # combined_risk_probability < SAFE_MAX          -> "safe"
SUSPICIOUS_MAX = 0.60    # SAFE_MAX <= combined_risk_probability < this  -> "suspicious"
# combined_risk_probability >= SUSPICIOUS_MAX                            -> "high_risk"


class FusionInputError(ValueError):
    """Raised when fuse() is called with no usable module outputs."""


def _renormalize_weights(modules_present: list[str]) -> dict[str, float]:
    """Scale DEFAULT_WEIGHTS down to just the modules present, summing to 1."""
    raw = {m: DEFAULT_WEIGHTS.get(m, 0.0) for m in modules_present}
    total = sum(raw.values())

    if total == 0:
        # None of the present modules are in DEFAULT_WEIGHTS (shouldn't
        # normally happen) -- fall back to equal weighting.
        n = len(modules_present)
        return {m: 1.0 / n for m in modules_present}

    return {m: w / total for m, w in raw.items()}


def _verdict_from_score(score: float) -> str:
    if score < SAFE_MAX:
        return "safe"
    if score < SUSPICIOUS_MAX:
        return "suspicious"
    return "high_risk"


def fuse(module_outputs: dict[str, dict]) -> dict:
    """
    Combine normalized outputs from any subset of the 5 modules.

    Args:
        module_outputs: dict keyed by module name ("url", "sms", "qr",
            "image", "email"), values are the common-contract dicts
            produced by adapters.py:
                {"module", "risk_probability", "label", "confidence", "explanation"}
            Only include modules that actually ran for this request.

    Returns:
        {
            "combined_risk_probability": float 0-1,
            "verdict": "safe" | "suspicious" | "high_risk",
            "override_triggered": bool,
            "override_module": str | None,
            "contributing_modules": [
                {"module": str, "risk_probability": float, "label": str,
                 "weight_used": float, "explanation": str},
                ...
            ],
            "explanation": str,
        }
    """
    if not module_outputs:
        raise FusionInputError("fuse() requires at least one module output.")

    modules_present = list(module_outputs.keys())

    # --- Rule override check -------------------------------------------------
    override_module = None
    override_prob = 0.0
    for name, out in module_outputs.items():
        if out["label"] == "malicious" and out["risk_probability"] >= OVERRIDE_THRESHOLD:
            if out["risk_probability"] > override_prob:
                override_module = name
                override_prob = out["risk_probability"]

    if override_module is not None:
        combined_score = override_prob
        verdict = "high_risk"
        override_triggered = True
        explanation = (
            f"High-risk override: '{override_module}' module reported "
            f"{override_prob:.2%} risk probability, which alone is enough "
            f"to flag this as high risk regardless of other modules."
        )
    else:
        # --- Weighted average -------------------------------------------------
        weights = _renormalize_weights(modules_present)
        combined_score = sum(
            module_outputs[m]["risk_probability"] * weights[m] for m in modules_present
        )
        verdict = _verdict_from_score(combined_score)
        override_triggered = False
        explanation = (
            f"Combined risk score {combined_score:.2%} from a weighted average of "
            f"{len(modules_present)} module(s): "
            + ", ".join(f"{m} ({module_outputs[m]['risk_probability']:.0%})" for m in modules_present)
            + f". Verdict: {verdict}."
        )
        weights_for_output = weights

    contributing_modules = []
    weights_used = weights if not override_triggered else _renormalize_weights(modules_present)
    for m in modules_present:
        out = module_outputs[m]
        contributing_modules.append({
            "module": m,
            "risk_probability": out["risk_probability"],
            "label": out["label"],
            "weight_used": round(weights_used[m], 4),
            "explanation": out.get("explanation", ""),
        })

    return {
        "combined_risk_probability": round(combined_score, 4),
        "verdict": verdict,
        "override_triggered": override_triggered,
        "override_module": override_module,
        "contributing_modules": contributing_modules,
        "explanation": explanation,
    }


if __name__ == "__main__":
    # Quick manual smoke test
    sample = {
        "url": {"module": "url", "risk_probability": 0.91, "label": "malicious",
                 "confidence": 0.91, "explanation": "Phishing URL detected."},
        "sms": {"module": "sms", "risk_probability": 0.12, "label": "legitimate",
                 "confidence": 0.88, "explanation": "No smishing indicators."},
    }
    import json
    print(json.dumps(fuse(sample), indent=2))
