"""
metadata_branch/feature_extraction.py

Converts a raw email (.eml bytes/string, or an already-split text+headers dict) into the
structured metadata feature vector described in the design document, Section 5.5:

    SPF / DKIM / DMARC pass-fail, sender-domain reputation, reply-to mismatch,
    number of links, link-domain vs display-text mismatch, urgency keyword count,
    attachment presence/type.

No third-party dependencies — everything here is Python standard library (`email`, `re`,
`urllib.parse`) so this file can be unit-tested in complete isolation from the rest of the
pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from email import message_from_string, message_from_bytes
from email.message import Message
from email.utils import parseaddr
from typing import Optional
from urllib.parse import urlparse

# --------------------------------------------------------------------------------------
# Static reference lists (small, hand-curated for the demo — extend in production with a
# real sender-reputation feed / allow-list, e.g. Tranco top domains or an internal CRM).
# --------------------------------------------------------------------------------------

TRUSTED_DOMAINS = {
    "gmail.com", "outlook.com", "yahoo.com", "icloud.com",
    "microsoft.com", "google.com", "apple.com", "amazon.com",
    "paypal.com", "github.com", "linkedin.com",
}

HIGH_RISK_TLDS = {
    "zip", "top", "xyz", "click", "country", "kim", "gq", "tk", "ml", "cf", "work", "loan",
}

URGENCY_KEYWORDS = {
    "urgent", "verify", "suspend", "suspended", "immediately", "action required",
    "limited time", "confirm your account", "unusual activity", "click here",
    "password expires", "act now", "final notice", "restricted", "unauthorized",
    "locked", "security alert", "update your information", "expire", "expires",
}

URL_REGEX = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
HTML_LINK_REGEX = re.compile(
    r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL
)
TAG_STRIP_REGEX = re.compile(r"<[^>]+>")


@dataclass
class EmailMetadataFeatures:
    spf_pass: int
    dkim_pass: int
    dmarc_pass: int
    sender_domain_trusted: int
    sender_domain_high_risk_tld: int
    reply_to_mismatch: int
    num_links: int
    link_display_mismatch_count: int
    urgency_keyword_count: int
    has_attachment: int
    attachment_is_executable_or_archive: int
    subject_has_urgency: int
    num_recipients: int
    display_name_domain_mismatch: int

    def to_vector(self) -> list:
        """Ordered numeric vector for the LightGBM model. Keep FEATURE_ORDER in sync."""
        return [getattr(self, f) for f in FEATURE_ORDER]

    def to_dict(self) -> dict:
        return asdict(self)


FEATURE_ORDER = [
    "spf_pass",
    "dkim_pass",
    "dmarc_pass",
    "sender_domain_trusted",
    "sender_domain_high_risk_tld",
    "reply_to_mismatch",
    "num_links",
    "link_display_mismatch_count",
    "urgency_keyword_count",
    "has_attachment",
    "attachment_is_executable_or_archive",
    "subject_has_urgency",
    "num_recipients",
    "display_name_domain_mismatch",
]

RISKY_ATTACHMENT_EXTS = {
    "exe", "scr", "js", "vbs", "bat", "cmd", "zip", "rar", "7z", "jar", "iso", "docm", "xlsm",
}


def _domain_of(address: Optional[str]) -> Optional[str]:
    if not address:
        return None
    _, addr = parseaddr(address)
    if "@" not in addr:
        return None
    return addr.rsplit("@", 1)[-1].strip().lower()


def _parse_auth_results(msg: Message) -> dict:
    """
    Parses the 'Authentication-Results' header (as added by receiving mail servers) for
    spf=, dkim=, dmarc= pass/fail. Falls back to explicit X-SPF / X-DKIM style headers if
    Authentication-Results is absent, and to 'unknown -> treat as fail' otherwise.
    """
    result = {"spf": 0, "dkim": 0, "dmarc": 0}
    auth_header = msg.get("Authentication-Results", "") or ""
    for mech in ("spf", "dkim", "dmarc"):
        m = re.search(rf"{mech}=(\w+)", auth_header, re.IGNORECASE)
        if m:
            result[mech] = 1 if m.group(1).lower() == "pass" else 0
        else:
            explicit = msg.get(f"X-{mech.upper()}-Result") or msg.get(f"X-{mech.upper()}")
            if explicit:
                result[mech] = 1 if "pass" in explicit.lower() else 0
    return result


def _get_body_text(msg: Message) -> str:
    """Extracts the best-effort plain text body (prefers text/plain, falls back to stripped HTML)."""
    if msg.is_multipart():
        plain_parts, html_parts = [], []
        for part in msg.walk():
            ctype = part.get_content_type()
            if part.get_content_disposition() == "attachment":
                continue
            try:
                payload = part.get_payload(decode=True)
                text = payload.decode(part.get_content_charset() or "utf-8", errors="ignore") if payload else ""
            except Exception:
                text = ""
            if ctype == "text/plain":
                plain_parts.append(text)
            elif ctype == "text/html":
                html_parts.append(text)
        if plain_parts:
            return "\n".join(plain_parts)
        if html_parts:
            return TAG_STRIP_REGEX.sub(" ", "\n".join(html_parts))
        return ""
    else:
        try:
            payload = msg.get_payload(decode=True)
            text = payload.decode(msg.get_content_charset() or "utf-8", errors="ignore") if payload else str(msg.get_payload())
        except Exception:
            text = str(msg.get_payload())
        if msg.get_content_type() == "text/html":
            text = TAG_STRIP_REGEX.sub(" ", text)
        return text


def _get_raw_html(msg: Message) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/html" and part.get_content_disposition() != "attachment":
                try:
                    payload = part.get_payload(decode=True)
                    return payload.decode(part.get_content_charset() or "utf-8", errors="ignore") if payload else ""
                except Exception:
                    return ""
        return ""
    if msg.get_content_type() == "text/html":
        return _get_body_text(msg)
    return ""


def _count_link_display_mismatches(html: str, plain_text: str) -> tuple[int, int]:
    """
    Returns (num_links, mismatch_count).
    A mismatch is a link whose *visible* text is itself a URL/domain that differs from the
    href target domain (the classic "looks like paypal.com but goes to evil.ru" pattern).
    """
    links_found = 0
    mismatches = 0
    if html:
        for href, display in HTML_LINK_REGEX.findall(html):
            links_found += 1
            display_clean = TAG_STRIP_REGEX.sub("", display).strip()
            display_url_match = URL_REGEX.search(display_clean) or re.search(
                r"[a-z0-9\-]+\.[a-z]{2,}", display_clean, re.IGNORECASE
            )
            if display_url_match:
                href_domain = urlparse(href).netloc.lower().lstrip("www.")
                disp_domain = re.sub(r"^https?://", "", display_url_match.group(0)).split("/")[0].lower().lstrip("www.")
                if href_domain and disp_domain and href_domain != disp_domain:
                    mismatches += 1
    else:
        links_found = len(URL_REGEX.findall(plain_text))
    return links_found, mismatches


def extract_features(raw_email: str | bytes, override_fields: Optional[dict] = None) -> EmailMetadataFeatures:
    """
    Main entry point.

    Args:
        raw_email: full RFC822 email as str or bytes (e.g. an .eml file's contents).
        override_fields: optional dict to force specific header values for synthetic/test
            data generation, e.g. {"spf": "fail", "dkim": "pass"}.

    Returns:
        EmailMetadataFeatures dataclass.
    """
    msg = message_from_bytes(raw_email) if isinstance(raw_email, bytes) else message_from_string(raw_email)

    auth = _parse_auth_results(msg)
    if override_fields:
        for k in ("spf", "dkim", "dmarc"):
            if k in override_fields:
                auth[k] = 1 if str(override_fields[k]).lower() == "pass" else 0

    from_domain = _domain_of(msg.get("From"))
    reply_to_domain = _domain_of(msg.get("Reply-To"))
    reply_to_mismatch = int(bool(reply_to_domain) and reply_to_domain != from_domain)

    sender_domain_trusted = int(from_domain in TRUSTED_DOMAINS) if from_domain else 0
    tld = from_domain.rsplit(".", 1)[-1] if from_domain and "." in from_domain else ""
    sender_domain_high_risk_tld = int(tld in HIGH_RISK_TLDS)

    display_name, addr = parseaddr(msg.get("From", ""))
    display_name_domain_mismatch = 0
    if display_name:
        dn_domain_match = re.search(r"[a-z0-9\-]+\.[a-z]{2,}", display_name, re.IGNORECASE)
        if dn_domain_match and from_domain:
            dn_domain = dn_domain_match.group(0).lower()
            if dn_domain != from_domain:
                display_name_domain_mismatch = 1

    body_text = _get_body_text(msg)
    html = _get_raw_html(msg)
    num_links, mismatch_count = _count_link_display_mismatches(html, body_text)

    combined_text_lower = (msg.get("Subject", "") + " " + body_text).lower()
    urgency_count = sum(1 for kw in URGENCY_KEYWORDS if kw in combined_text_lower)
    subject_has_urgency = int(any(kw in msg.get("Subject", "").lower() for kw in URGENCY_KEYWORDS))

    has_attachment = 0
    risky_attachment = 0
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_disposition() == "attachment":
                has_attachment = 1
                filename = part.get_filename() or ""
                ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
                if ext in RISKY_ATTACHMENT_EXTS:
                    risky_attachment = 1

    to_field = msg.get("To", "") or ""
    num_recipients = max(1, len([a for a in to_field.split(",") if a.strip()]))

    return EmailMetadataFeatures(
        spf_pass=auth["spf"],
        dkim_pass=auth["dkim"],
        dmarc_pass=auth["dmarc"],
        sender_domain_trusted=sender_domain_trusted,
        sender_domain_high_risk_tld=sender_domain_high_risk_tld,
        reply_to_mismatch=reply_to_mismatch,
        num_links=num_links,
        link_display_mismatch_count=mismatch_count,
        urgency_keyword_count=urgency_count,
        has_attachment=has_attachment,
        attachment_is_executable_or_archive=risky_attachment,
        subject_has_urgency=subject_has_urgency,
        num_recipients=num_recipients,
        display_name_domain_mismatch=display_name_domain_mismatch,
    )


def extract_body_text(raw_email: str | bytes) -> str:
    """Convenience helper so the text branch can reuse the same MIME-walking logic."""
    msg = message_from_bytes(raw_email) if isinstance(raw_email, bytes) else message_from_string(raw_email)
    return _get_body_text(msg).strip()
