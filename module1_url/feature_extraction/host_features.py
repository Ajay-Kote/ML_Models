"""
host_features.py

Host-level features for phishing URL detection: signals that come from
looking up the DOMAIN itself (not just parsing the URL string).

Why these matter for phishing detection:
  - Domain age: phishing domains are usually registered days/weeks before
    an attack, then abandoned. A domain that's 10+ years old is very
    unlikely to be a phishing site.
  - SSL certificate: most phishing kits now use free/auto-issued certs
    (short validity, no organization info) rather than no HTTPS at all,
    so cert *type* is more informative than just "has HTTPS".
  - DNS records: legitimate businesses almost always have MX records
    (they receive email on their domain); many disposable phishing
    domains don't bother setting one up.

STATUS: This is a standalone, working extractor. It is NOT yet wired
into url_feature_extractor.py / the trained model (saved_model.pkl),
because doing that properly requires re-running these lookups across
the full ~47k-row training set and retraining -- these are network
calls (WHOIS/DNS/SSL), so that's slow and rate-limit-prone to do in bulk.

To integrate later:
  1. Run get_host_features(url) over the training CSV (expect this to
     take a while / need retries -- WHOIS servers rate-limit).
  2. Merge the resulting columns into the training feature matrix.
  3. Retrain models/train.py with the expanded feature set.
  4. In models/predict.py, call get_host_features() alongside
     URLFeatureExtractor(url).extract() and merge both dicts before
     building the DataFrame.

Until then, this can be used standalone/for explainability write-ups,
or as a secondary signal outside the trained model.
"""

from __future__ import annotations

import socket
import ssl
from datetime import datetime, timezone
from urllib.parse import urlparse

try:
    import whois  # pip install python-whois
    _HAS_WHOIS = True
except ImportError:
    _HAS_WHOIS = False

try:
    import dns.resolver  # pip install dnspython
    _HAS_DNS = True
except ImportError:
    _HAS_DNS = False

NETWORK_TIMEOUT_SECONDS = 5


def _get_hostname(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.netloc.split(":")[0].split("@")[-1]
    return host.lower().lstrip("www.")


def get_domain_age_days(hostname: str) -> int | None:
    """Days since domain registration. Returns None if lookup fails/unavailable."""
    if not _HAS_WHOIS:
        return None
    try:
        w = whois.whois(hostname)
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if creation is None:
            return None
        if creation.tzinfo is None:
            creation = creation.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - creation).days
        return max(age, 0)
    except Exception:
        return None


def get_ssl_cert_info(hostname: str, port: int = 443) -> dict:
    """
    Connects over TLS and inspects the certificate.
    Returns dict with has_valid_cert, cert_days_remaining, cert_has_org_info.
    All fields default to "unknown/false" values if the connection fails
    (e.g. site doesn't support HTTPS, or is unreachable) -- that itself
    is a mildly suspicious signal for a claimed-legitimate site.
    """
    result = {
        "has_valid_cert": 0,
        "cert_days_remaining": -1,
        "cert_has_org_info": 0,
    }
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=NETWORK_TIMEOUT_SECONDS) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()

        result["has_valid_cert"] = 1

        not_after = datetime.strptime(cert["notAfter"], "%b %d %H:%M:%S %Y %Z")
        not_after = not_after.replace(tzinfo=timezone.utc)
        result["cert_days_remaining"] = max((not_after - datetime.now(timezone.utc)).days, 0)

        subject = dict(x[0] for x in cert.get("subject", []))
        result["cert_has_org_info"] = int("organizationName" in subject)

    except Exception:
        pass

    return result


def get_dns_features(hostname: str) -> dict:
    """
    Checks presence of MX (mail) and NS (nameserver) records.
    has_mx_record: legitimate businesses almost always have mail set up
    on their domain; short-lived phishing domains frequently don't.
    """
    result = {"has_mx_record": 0, "ns_record_count": 0}
    if not _HAS_DNS:
        return result

    resolver = dns.resolver.Resolver()
    resolver.timeout = NETWORK_TIMEOUT_SECONDS
    resolver.lifetime = NETWORK_TIMEOUT_SECONDS
    # Explicit public DNS server -- on some Windows setups the OS default
    # resolver doesn't respond properly to dnspython's raw UDP queries,
    # silently causing every lookup to fail. Google's public resolver
    # sidesteps that.
    resolver.nameservers = ["8.8.8.8", "1.1.1.1"]

    try:
        mx_answers = resolver.resolve(hostname, "MX")
        result["has_mx_record"] = int(len(mx_answers) > 0)
    except Exception:
        pass

    try:
        ns_answers = resolver.resolve(hostname, "NS")
        result["ns_record_count"] = len(ns_answers)
    except Exception:
        pass

    return result


def get_host_features(url: str) -> dict:
    """
    Main entry point. Runs all host-level lookups for a URL and returns
    one flat dict, ready to merge into URLFeatureExtractor's output.

    NOTE: this does real network I/O (WHOIS + DNS + a live TLS handshake),
    so expect ~1-3 seconds per call, and expect some fields to come back
    as "unknown" (-1 / 0) for unreachable or newly-registered domains --
    that's expected behavior, not a bug.
    """
    hostname = _get_hostname(url)

    domain_age_days = get_domain_age_days(hostname)
    ssl_info = get_ssl_cert_info(hostname)
    dns_info = get_dns_features(hostname)

    return {
        "Domain_Age_Days": domain_age_days if domain_age_days is not None else -1,
        "Domain_Age_Unknown": int(domain_age_days is None),
        "Domain_Age_Under_30_Days": int(domain_age_days is not None and domain_age_days < 30),
        **ssl_info,
        **dns_info,
    }


if __name__ == "__main__":
    test_url = input("Enter URL: ")
    for k, v in get_host_features(test_url).items():
        print(f"{k}: {v}")