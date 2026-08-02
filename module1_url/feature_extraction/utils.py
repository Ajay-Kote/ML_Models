"""
utils.py

Helper functions for URL feature extraction.
"""

import math
import re
import ipaddress
from urllib.parse import urlparse
import tldextract


def get_domain(url):
    """
    Extract domain name.
    Example:
    https://mail.google.com/login

    returns:
    google
    """
    ext = tldextract.extract(url)
    return ext.domain


def get_subdomain(url):
    """
    Extract subdomain.
    """
    ext = tldextract.extract(url)
    return ext.subdomain


def get_suffix(url):
    """
    Extract TLD.
    Example:
    google.com -> com
    """
    ext = tldextract.extract(url)
    return ext.suffix


def url_length(url):
    return len(url)


def domain_length(url):
    return len(get_domain(url))


def count_digits(url):
    return sum(c.isdigit() for c in url)


def count_letters(url):
    return sum(c.isalpha() for c in url)


def count_special_characters(url):
    special = re.findall(r"[^a-zA-Z0-9]", url)
    return len(special)


def count_hyphen(url):
    return url.count("-")


def count_dot(url):
    return url.count(".")


def count_slash(url):
    return url.count("/")


def count_question(url):
    return url.count("?")


def count_equal(url):
    return url.count("=")


def count_ampersand(url):
    return url.count("&")


def count_at(url):
    return url.count("@")


def count_percent(url):
    return url.count("%")


def count_underscore(url):
    return url.count("_")


def has_https(url):
    return int(url.lower().startswith("https"))


def has_ip_address(url):
    """
    Check whether URL contains IP instead of domain.
    """
    try:
        hostname = urlparse(url).hostname
        ipaddress.ip_address(hostname)
        return 1
    except:
        return 0


def number_of_subdomains(url):
    sub = get_subdomain(url)

    if sub == "":
        return 0

    return len(sub.split("."))


def calculate_entropy(url):
    """
    Shannon Entropy
    """
    if len(url) == 0:
        return 0

    probability = []

    for char in set(url):
        p = float(url.count(char)) / len(url)
        probability.append(p)

    entropy = -sum(p * math.log2(p) for p in probability)

    return round(entropy, 4)