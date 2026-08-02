"""
data/generate_synthetic_data.py

Design doc, Section 10 (Dataset Sources) notes for Email:
    "Public phishing email corpora (e.g., Nazario, SpamAssassin) + legitimate email samples.
    Combine multiple sources for header diversity."

This repository does not ship real corpora (they are external downloads, some require
attribution/agreements). This script instead generates a small, clearly-synthetic labeled
dataset with the same structure (raw .eml text + label) so that `train.py` and `predict.py`
can be exercised end-to-end offline. Swap this out for a real loader over Nazario/
SpamAssassin/Enron-ham in production — nothing else in the pipeline needs to change, since
downstream code only depends on getting raw RFC822 email text + a 0/1 label.

Usage:
    python data/generate_synthetic_data.py            # writes data/processed/emails.csv
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

random.seed(42)

LEGITIMATE_SENDERS = [
    ("Amazon", "auto-confirm@amazon.com"),
    ("GitHub", "notifications@github.com"),
    ("LinkedIn", "messages-noreply@linkedin.com"),
    ("Google", "no-reply@google.com"),
    ("Team Lead", "priya.sharma@microsoft.com"),
    ("HR Dept", "hr@company.com"),
]

PHISHING_SENDERS = [
    ("Amazon Security", "security-update@amaz0n-verify.top"),
    ("PayPal Support", "support@paypal-secure-verify.xyz"),
    ("IT Helpdesk", "helpdesk@corp-it-alerts.click"),
    ("Bank Alert", "alert@yourbank-online.kim"),
    ("Netflix Billing", "billing@netflix-account-update.zip"),
]

LEGIT_SUBJECTS_BODIES = [
    ("Your order has shipped",
     "Hi there,\n\nYour recent order #4471 has shipped and is on its way. "
     "You can track the package from your account order history page.\n\nThanks,\nThe Team"),
    ("Weekly team sync notes",
     "Hi all,\n\nAttaching this week's sync notes. Let me know if I missed anything "
     "from the roadmap discussion. See you at the next standup.\n\nBest,\nPriya"),
    ("New comment on your pull request",
     "Someone commented on your pull request #182. View the discussion and reply "
     "on the repository page whenever you get a chance."),
    ("Your monthly statement is ready",
     "Your statement for this billing cycle is now available in your online account. "
     "No action is required unless you have questions about a specific charge."),
]

PHISH_SUBJECTS_BODIES = [
    ("URGENT: Your account will be suspended",
     "Dear Customer,\n\nWe detected unusual activity on your account. Your account will be "
     "suspended within 24 hours unless you verify your information immediately. "
     "Click here to confirm your identity and avoid permanent restriction.\n\n"
     "Failure to act now will result in loss of access."),
    ("Action Required: Password expires today",
     "Your password expires today. To avoid being locked out, please verify your account "
     "immediately by clicking the secure link below. This is your final notice."),
    ("Unusual sign-in activity detected",
     "We noticed an unauthorized login attempt. For your security, please confirm your "
     "account details urgently. Immediate action is required to restore access."),
    ("Payment failed - update required immediately",
     "Your last payment could not be processed. To avoid service suspension, please "
     "update your billing information immediately by verifying your account now."),
]


def _build_eml(display_name: str, from_addr: str, to_addr: str, subject: str, body: str,
               spf: str, dkim: str, dmarc: str, reply_to: str | None,
               html_link: tuple[str, str] | None, attachment: str | None) -> str:
    headers = [
        f"From: {display_name} <{from_addr}>",
        f"To: {to_addr}",
        f"Subject: {subject}",
        f"Authentication-Results: mx.example.com; spf={spf}; dkim={dkim}; dmarc={dmarc}",
    ]
    if reply_to:
        headers.append(f"Reply-To: {reply_to}")

    if html_link or attachment:
        boundary = "BOUNDARY123"
        headers.append("MIME-Version: 1.0")
        headers.append(f'Content-Type: multipart/mixed; boundary="{boundary}"')
        parts = [f"--{boundary}", "Content-Type: text/plain; charset=utf-8", "", body, ""]
        if html_link:
            link_text, href = html_link
            html_body = f"<html><body><p>{body}</p><a href='{href}'>{link_text}</a></body></html>"
            parts += [f"--{boundary}", "Content-Type: text/html; charset=utf-8", "", html_body, ""]
        if attachment:
            parts += [
                f"--{boundary}",
                f'Content-Type: application/octet-stream; name="{attachment}"',
                "Content-Disposition: attachment; filename=\"" + attachment + "\"",
                "Content-Transfer-Encoding: base64", "", "QUJDREVGRw==", "",
            ]
        parts.append(f"--{boundary}--")
        return "\n".join(headers) + "\n\n" + "\n".join(parts)
    else:
        return "\n".join(headers) + "\n\n" + body


def generate(n_per_class: int = 60) -> list[dict]:
    rows = []
    for i in range(n_per_class):
        name, addr = random.choice(LEGITIMATE_SENDERS)
        subject, body = random.choice(LEGIT_SUBJECTS_BODIES)
        eml = _build_eml(
            display_name=name, from_addr=addr, to_addr="user@example.com",
            subject=subject, body=body,
            spf="pass", dkim="pass", dmarc="pass", reply_to=None,
            html_link=("View order", f"https://{addr.split('@')[1]}/orders/4471") if i % 2 == 0 else None,
            attachment=None,
        )
        rows.append({"raw_email": eml, "label": 0})

    for i in range(n_per_class):
        name, addr = random.choice(PHISHING_SENDERS)
        subject, body = random.choice(PHISH_SUBJECTS_BODIES)
        fake_domain = addr.split("@")[1]
        eml = _build_eml(
            display_name=name, from_addr=addr, to_addr="user@example.com",
            subject=subject, body=body,
            spf="fail", dkim="fail", dmarc="fail",
            reply_to=f"reply@{'attacker-collect.ru' if i % 2 == 0 else fake_domain}",
            html_link=("www.paypal.com/verify" if i % 2 == 0 else "amazon.com/account",
                       f"http://{fake_domain}/phish/{i}"),
            attachment="invoice.zip" if i % 3 == 0 else None,
        )
        rows.append({"raw_email": eml, "label": 1})

    random.shuffle(rows)
    return rows


def main():
    out_dir = Path(__file__).parent / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = generate(n_per_class=60)

    out_path = out_dir / "emails.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["raw_email", "label"])
        writer.writeheader()
        writer.writerows(rows)

    n_phish = sum(r["label"] for r in rows)
    print(f"Wrote {len(rows)} synthetic emails ({n_phish} phishing / {len(rows) - n_phish} legitimate) to {out_path}")


if __name__ == "__main__":
    main()
