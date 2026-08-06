"""
fusion_engine/test_fuse.py

Manual test scenarios for fuse.py -- no pytest needed, just run:
    python test_fuse.py

Covers:
  1. Single module, low risk               -> "safe"
  2. Two modules, both mid risk             -> "suspicious" (weighted average)
  3. One module >= 90% confidence malicious -> override triggers "high_risk"
     even though other modules look clean
  4. All modules high risk                  -> "high_risk" via weighted average
  5. Only 1 of 5 modules ran (others missing/null) -> still works correctly
"""

from fuse import fuse, FusionInputError


def make_output(module, risk_probability, label=None, confidence=None, explanation=""):
    if label is None:
        label = "malicious" if risk_probability >= 0.5 else "legitimate"
    if confidence is None:
        confidence = risk_probability if risk_probability >= 0.5 else 1 - risk_probability
    return {
        "module": module,
        "risk_probability": risk_probability,
        "label": label,
        "confidence": confidence,
        "explanation": explanation,
    }


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def run_all():
    results = []

    # --- Scenario 1: single module, low risk -> safe ---
    out = fuse({"sms": make_output("sms", 0.05)})
    results.append(check(
        "Scenario 1: single low-risk module -> safe",
        out["verdict"] == "safe" and not out["override_triggered"]
    ))

    # --- Scenario 2: two modules, both mid risk -> suspicious ---
    out = fuse({
        "url": make_output("url", 0.45),
        "sms": make_output("sms", 0.50),
    })
    results.append(check(
        "Scenario 2: two mid-risk modules -> suspicious",
        out["verdict"] == "suspicious" and not out["override_triggered"]
    ))

    # --- Scenario 3: one module >=90% malicious overrides everything else ---
    out = fuse({
        "url": make_output("url", 0.95, label="malicious"),
        "sms": make_output("sms", 0.05, label="legitimate"),
        "email": make_output("email", 0.10, label="legitimate"),
    })
    results.append(check(
        "Scenario 3: high-confidence single module triggers override",
        out["verdict"] == "high_risk"
        and out["override_triggered"] is True
        and out["override_module"] == "url"
    ))

    # --- Scenario 4: all modules high risk -> high_risk via weighted avg ---
    out = fuse({
        "url": make_output("url", 0.75),
        "sms": make_output("sms", 0.70),
        "qr": make_output("qr", 0.65),
        "image": make_output("image", 0.80),
        "email": make_output("email", 0.72),
    })
    results.append(check(
        "Scenario 4: all modules high risk -> high_risk",
        out["verdict"] == "high_risk"
    ))

    # --- Scenario 5: only 1 of 5 modules ran ---
    out = fuse({"qr": make_output("qr", 0.40)})
    results.append(check(
        "Scenario 5: only QR module ran, weight renormalized to 1.0",
        len(out["contributing_modules"]) == 1
        and out["contributing_modules"][0]["weight_used"] == 1.0
        and out["combined_risk_probability"] == 0.40
    ))

    # --- Bonus: empty input should raise ---
    try:
        fuse({})
        results.append(check("Bonus: empty input raises FusionInputError", False))
    except FusionInputError:
        results.append(check("Bonus: empty input raises FusionInputError", True))

    print()
    passed = sum(results)
    print(f"{passed}/{len(results)} scenarios passed")
    return passed == len(results)


if __name__ == "__main__":
    all_passed = run_all()
    exit(0 if all_passed else 1)
