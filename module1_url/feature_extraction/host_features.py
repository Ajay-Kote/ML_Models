"""
host_features.py

Host-level features for phishing URL detection: signals that come from
looking up the DOMAIN itself (not just parsing the URL string).

  - Domain age: phishing domains are usually registered days/weeks before
    an attack, then abandoned. A domain that's 10+ years old is very
    unlikely to be a phishing site.
  - SSL certificate: most phishing kits now use free/auto-issued certs
    (short validity, no organization info) rather than no HTTPS at all,
    so cert *type* is more informative than just "has HTTPS".
  - DNS records: legitimate businesses almost always have MX records
    (they receive email on their domain); many disposable phishing
    domains don't bother setting one up.

IMPORTANT: Domain age (WHOIS) and MX/NS records are properties of the
APEX/registrable domain (e.g. "google.com"), not of a specific subdomain
(e.g. "mail.google.com" itself has no MX/NS records - that's normal DNS
behavior, not a red flag). Looking these up on the literal subdomain
under-reports legitimacy for perfectly normal subdomains. SSL certs,
on the other hand, ARE issued per-hostname, so that check still uses the
exact hostname from the URL.

This version:
  1. Splits lookups: WHOIS/DNS use the apex domain; SSL uses the literal
     hostname.
  2. Caches by hostname/apex domain (get_host_features_cached) - so if the
     same domain (e.g. paypal.com) appears hundreds of times in the
     training set, it's only looked up ONCE.
  3. get_host_features_bulk() - runs lookups for many URLs in parallel
     threads, deduplicating by hostname/apex first. This is the function
     the training pipeline should call.
"""

from __future__ import annotations

import socket
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from urllib.parse import urlparse

import tldextract

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
    """Literal hostname as it appears in the URL (used for SSL cert check)."""
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.netloc.split(":")[0].split("@")[-1]
    return host.lower().lstrip("www.")


def _get_apex_domain(hostname: str) -> str:
    """
    Registrable/apex domain (used for WHOIS domain age + MX/NS lookups).
    e.g. "mail.google.com" -> "google.com", "accounts.google.co.uk" -> "google.co.uk"
    """
    ext = tldextract.extract(hostname)
    if not ext.domain or not ext.suffix:
        return hostname
    return f"{ext.domain}.{ext.suffix}"


def get_domain_age_days(apex_domain: str) -> int | None:
    """Days since domain registration. Returns None if lookup fails/unavailable."""
    if not _HAS_WHOIS:
        return None
    try:
        w = whois.whois(apex_domain)
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
    Connects over TLS and inspects the certificate for the EXACT hostname
    from the URL (certs are issued per-hostname, so this should NOT use
    the apex domain).
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


def get_dns_features(apex_domain: str) -> dict:
    """Checks presence of MX (mail) and NS (nameserver) records on the APEX domain."""
    result = {"has_mx_record": 0, "ns_record_count": 0}
    if not _HAS_DNS:
        return result

    resolver = dns.resolver.Resolver()
    resolver.timeout = NETWORK_TIMEOUT_SECONDS
    resolver.lifetime = NETWORK_TIMEOUT_SECONDS
    resolver.nameservers = ["8.8.8.8", "1.1.1.1"]

    try:
        mx_answers = resolver.resolve(apex_domain, "MX")
        result["has_mx_record"] = int(len(mx_answers) > 0)
    except Exception:
        pass

    try:
        ns_answers = resolver.resolve(apex_domain, "NS")
        result["ns_record_count"] = len(ns_answers)
    except Exception:
        pass

    return result


def get_host_features(url: str) -> dict:
    """
    Main entry point for a SINGLE url. Runs all host-level lookups and
    returns one flat dict, ready to merge into URLFeatureExtractor's output.
    Expect ~1-3 seconds per call (real network I/O).
    """
    hostname = _get_hostname(url)
    apex = _get_apex_domain(hostname)
    return _lookup_all(hostname, apex)


def _lookup_all(hostname: str, apex: str) -> dict:
    domain_age_days = get_domain_age_days(apex)
    ssl_info = get_ssl_cert_info(hostname)  # exact hostname, not apex
    dns_info = get_dns_features(apex)

    return {
        "Domain_Age_Days": domain_age_days if domain_age_days is not None else -1,
        "Domain_Age_Unknown": int(domain_age_days is None),
        "Domain_Age_Under_30_Days": int(domain_age_days is not None and domain_age_days < 30),
        **ssl_info,
        **dns_info,
    }


@lru_cache(maxsize=None)
def _cached_lookup(hostname: str, apex: str) -> tuple:
    """Cached by (hostname, apex) pair. Returns a hashable tuple for lru_cache."""
    feats = _lookup_all(hostname, apex)
    return tuple(sorted(feats.items()))


def get_host_features_cached(url: str) -> dict:
    """
    Same as get_host_features(), but memoized within this process. Use this
    instead of get_host_features() whenever the same domain is likely to
    repeat many times (e.g. augmented training data with hundreds of
    paypal.com / google.com variants).
    """
    hostname = _get_hostname(url)
    apex = _get_apex_domain(hostname)
    return dict(_cached_lookup(hostname, apex))


def get_host_features_bulk(urls: list[str], max_workers: int = 40, progress: bool = True) -> dict:
    """
    Looks up host features for a list of URLs, in parallel, deduplicating
    SSL checks by exact hostname and WHOIS/DNS checks by apex domain
    (so e.g. mail.google.com and accounts.google.com share one WHOIS/DNS
    lookup for "google.com", but each still gets its own SSL check).

    Returns: {url: features_dict, ...} covering every url in the input.
    """
    url_to_host = {url: _get_hostname(url) for url in urls}
    url_to_apex = {url: _get_apex_domain(url_to_host[url]) for url in urls}

    unique_hosts = sorted(set(url_to_host.values()))
    unique_apexes = sorted(set(url_to_apex.values()))

    print(f"[host_features] {len(urls)} URLs -> {len(unique_hosts)} unique hostnames "
          f"(SSL), {len(unique_apexes)} unique apex domains (WHOIS/DNS)")

    # ---- SSL lookups, keyed by exact hostname ----
    ssl_results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_ssl_cert_info, h): h for h in unique_hosts}
        done = 0
        for future in as_completed(futures):
            h = futures[future]
            try:
                ssl_results[h] = future.result()
            except Exception:
                ssl_results[h] = {"has_valid_cert": 0, "cert_days_remaining": -1, "cert_has_org_info": 0}
            done += 1
            if progress and done % 200 == 0:
                print(f"[host_features] SSL: {done}/{len(unique_hosts)} done")

    # ---- WHOIS + DNS lookups, keyed by apex domain ----
    apex_results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        def _apex_lookup(apex):
            age = get_domain_age_days(apex)
            dns_info = get_dns_features(apex)
            return {
                "Domain_Age_Days": age if age is not None else -1,
                "Domain_Age_Unknown": int(age is None),
                "Domain_Age_Under_30_Days": int(age is not None and age < 30),
                **dns_info,
            }

        futures = {executor.submit(_apex_lookup, a): a for a in unique_apexes}
        done = 0
        for future in as_completed(futures):
            a = futures[future]
            try:
                apex_results[a] = future.result()
            except Exception:
                apex_results[a] = {
                    "Domain_Age_Days": -1, "Domain_Age_Unknown": 1,
                    "Domain_Age_Under_30_Days": 0, "has_mx_record": 0, "ns_record_count": 0,
                }
            done += 1
            if progress and done % 200 == 0:
                print(f"[host_features] WHOIS/DNS: {done}/{len(unique_apexes)} done")

    # ---- Combine per url ----
    return {
        url: {**apex_results[url_to_apex[url]], **ssl_results[url_to_host[url]]}
        for url in urls
    }


if __name__ == "__main__":
    test_url = input("Enter URL: ")
    for k, v in get_host_features(test_url).items():
        print(f"{k}: {v}")