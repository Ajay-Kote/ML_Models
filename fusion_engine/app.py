"""
app.py — Streamlit demo frontend for Adaptive Risk Fusion.

Run this AFTER the fusion_engine API is running (uvicorn api:app --reload
on http://127.0.0.1:8000).

To run:
    pip install streamlit requests plotly pandas
    streamlit run app.py
"""

import math

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"
API_ROOT = "http://127.0.0.1:8000/"

st.set_page_config(
    page_title="Adaptive Risk Fusion — Fraud & Phishing Detector",
    page_icon="🛡️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Theme: "security console" — deep ink-navy, teal/coral risk spectrum,
# Space Grotesk + IBM Plex Sans/Mono. Deliberately not the generic
# cream+terracotta or near-black+neon AI-template look.
# ---------------------------------------------------------------------------
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --ink: #0B1220;
    --panel: #131B2E;
    --panel-border: #2A3550;
    --text: #E7ECF5;
    --text-muted: #8C9AB8;
    --safe: #3DDC97;
    --warn: #FFB84D;
    --risk: #FF6B4A;
}

html, body, [class*="css"]  {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background: var(--ink);
    color: var(--text);
}

section[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--panel-border);
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.01em;
}

/* Masthead */
.console-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.25rem;
}
.console-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.1rem;
    line-height: 1.15;
    margin: 0 0 0.35rem 0;
    color: var(--text);
}
.console-subtitle {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}

/* Buttons */
.stButton > button {
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--panel-border);
    border-radius: 6px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
}
.stButton > button:hover {
    border-color: var(--safe);
    color: var(--safe);
}
.stFormSubmitButton > button {
    background: var(--safe) !important;
    color: #04140D !important;
    border: none !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.02em;
}
.stFormSubmitButton > button:hover {
    filter: brightness(1.08);
}

/* Inputs */
.stTextInput input, .stTextArea textarea {
    background: var(--panel) !important;
    color: var(--text) !important;
    border: 1px solid var(--panel-border) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* Expanders (module breakdown) */
div[data-testid="stExpander"] {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 8px;
}

/* Verdict card */
.verdict-card {
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin: 1.25rem 0 1rem 0;
    border: 1px solid var(--panel-border);
    background: var(--panel);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
}
.verdict-card.verdict-safe { border-left: 5px solid var(--safe); }
.verdict-card.verdict-suspicious { border-left: 5px solid var(--warn); }
.verdict-card.verdict-high_risk { border-left: 5px solid var(--risk); }

.verdict-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.2rem;
}
.verdict-text {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.7rem;
}
.verdict-safe .verdict-text { color: var(--safe); }
.verdict-suspicious .verdict-text { color: var(--warn); }
.verdict-high_risk .verdict-text { color: var(--risk); }

.verdict-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: var(--text-muted);
}

.override-note {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    background: rgba(255, 184, 77, 0.08);
    border: 1px solid rgba(255, 184, 77, 0.35);
    color: var(--warn);
    padding: 0.6rem 0.9rem;
    border-radius: 6px;
    margin-bottom: 0.9rem;
}

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 1.4rem 0 0.5rem 0;
    border-top: 1px solid var(--panel-border);
    padding-top: 1rem;
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Signature element: hand-built radial risk dial (SVG), styled like a
# security-console instrument rather than a default chart-library gauge.
# ---------------------------------------------------------------------------
def render_gauge_svg(score: float) -> str:
    score = max(0.0, min(1.0, score))
    cx, cy, r = 150, 150, 118

    if score < 0.4:
        color = "#3DDC97"
    elif score < 0.7:
        color = "#FFB84D"
    else:
        color = "#FF6B4A"

    circumference = math.pi * r
    filled = circumference * score

    # Needle angle: pi (pointing left, score=0) -> 0 (pointing right, score=1)
    angle = math.pi * (1 - score)
    needle_len = r - 18
    nx = cx + needle_len * math.cos(angle)
    ny = cy - needle_len * math.sin(angle)

    # Tick marks every 10%
        # Tick marks every 10% (longer tick every 50%)
    ticks = []
    for i in range(11):
        t = i / 10
        a = math.pi * (1 - t)
        tick_len = 10 if i % 5 == 0 else 4
        x1 = cx + (r + 6) * math.cos(a)
        y1 = cy - (r + 6) * math.sin(a)
        x2 = cx + (r + 6 - tick_len) * math.cos(a)
        y2 = cy - (r + 6 - tick_len) * math.sin(a)
        ticks.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                      f'stroke="#2A3550" stroke-width="2"/>')

    pct = f"{score * 100:.1f}"

    return f"""
    <svg viewBox="0 0 300 190" xmlns="http://www.w3.org/2000/svg" style="width:100%; max-width:340px; display:block; margin:0 auto;">
        <path d="M {cx - r} {cy} A {r} {r} 0 0 1 {cx + r} {cy}"
              fill="none" stroke="#1C2740" stroke-width="14" stroke-linecap="round"/>
        <path d="M {cx - r} {cy} A {r} {r} 0 0 1 {cx + r} {cy}"
              fill="none" stroke="{color}" stroke-width="14" stroke-linecap="round"
              stroke-dasharray="{filled:.1f} {circumference:.1f}"/>
        {''.join(ticks)}
        <line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}"
              stroke="#E7ECF5" stroke-width="3" stroke-linecap="round"/>
        <circle cx="{cx}" cy="{cy}" r="7" fill="#E7ECF5"/>
        <text x="{cx}" y="{cy - 34}" text-anchor="middle"
              font-family="IBM Plex Mono, monospace" font-size="30" font-weight="600" fill="{color}">{pct}%</text>
        <text x="{cx}" y="{cy - 12}" text-anchor="middle"
              font-family="IBM Plex Mono, monospace" font-size="11" letter-spacing="2" fill="#8C9AB8">RISK SCORE</text>
    </svg>
    """


# ---------------------------------------------------------------------------
# Session state defaults (so quick-example buttons can pre-fill the form)
# ---------------------------------------------------------------------------
for key, default in {"url_val": "", "sms_val": "", "email_val": ""}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def load_phishing_example():
    st.session_state.url_val = "http://paypal-secure-verify.account-update.tk/login"
    st.session_state.sms_val = (
        "Congratulations! You have won a $1000 Amazon gift card. "
        "Click here to claim now: bit.ly/claim-prize-xyz"
    )
    st.session_state.email_val = (
        "From: security@amaz0n-support.com\n"
        "To: user@example.com\n"
        "Subject: Urgent: Verify your account now\n\n"
        "Your account has been suspended. Click here immediately to verify: "
        "http://amaz0n-verify.tk/login"
    )


def load_safe_example():
    st.session_state.url_val = "https://www.wikipedia.org/"
    st.session_state.sms_val = "Your OTP is 4521, do not share with anyone. Valid for 5 mins."
    st.session_state.email_val = (
        "From: notifications@github.com\n"
        "To: user@example.com\n"
        "Subject: Your weekly digest\n\n"
        "Here's what happened in your repositories this week."
    )


# ---------------------------------------------------------------------------
# Sidebar — About
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="console-eyebrow">System overview</div>', unsafe_allow_html=True)
    st.header("About This Project")
    st.markdown(
        """
**Adaptive Risk Fusion** is a multi-modal fraud & phishing detection
system combining 5 independently trained detection modules:

| Module | Model |
|---|---|
| URL | LightGBM (lexical + host features) |
| SMS | Fine-tuned DistilBERT |
| QR Code | LightGBM (pixel + metadata + texture) |
| Payment Screenshot | PaddleOCR + EfficientNet-B0 + LightGBM |
| Email | DistilBERT + LightGBM + MLP fusion |

Their outputs are combined by a **Logistic Regression meta-learner**,
trained on each module's real held-out test performance — so the fusion
model *learns* how much to trust each channel, rather than using
fixed/hand-picked weights. A rule-based safety override also fires if
any single module is ≥90% confident, so a strong signal is never diluted.
        """
    )
    st.markdown("---")

    # API health indicator
    try:
        r = requests.get(API_ROOT, timeout=2)
        if r.status_code == 200:
            st.success("Backend API: Connected")
        else:
            st.warning("Backend API: Unexpected response")
    except requests.exceptions.RequestException:
        st.error("Backend API: Not reachable")
        st.caption("Start it with:\n```\ncd fusion_engine\npython -m uvicorn api:app --reload\n```")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown('<div class="console-eyebrow">Adaptive Risk Fusion / Detection Console</div>', unsafe_allow_html=True)
st.markdown('<div class="console-title">Multi-Channel Fraud & Phishing Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="console-subtitle">URL · SMS · QR · Payment Screenshot · Email — '
    'fused into a single risk verdict.</div>',
    unsafe_allow_html=True,
)

st.markdown("Fill in whichever inputs you have (you don't need all five), or try a quick example:")

ex_col1, ex_col2 = st.columns(2)
with ex_col1:
    st.button("Try Phishing Example", on_click=load_phishing_example, use_container_width=True)
with ex_col2:
    st.button("Try Safe Example", on_click=load_safe_example, use_container_width=True)

with st.form("fusion_form"):
    col1, col2 = st.columns(2)

    with col1:
        url = st.text_input("URL", key="url_val", placeholder="http://example.com/login")
        sms_text = st.text_area("SMS Text", key="sms_val", placeholder="Your OTP is 4521...", height=100)
        email_raw = st.text_area(
            "Raw Email (headers + body)",
            key="email_val",
            placeholder="From: sender@example.com\nSubject: ...\n\nBody text...",
            height=150,
        )

    with col2:
        qr_image = st.file_uploader("QR Code Image", type=["png", "jpg", "jpeg"])
        payment_image = st.file_uploader("Payment Screenshot", type=["png", "jpg", "jpeg"])

    submitted = st.form_submit_button("Analyze", use_container_width=True, type="primary")


# ---------------------------------------------------------------------------
# Call the API and render results
# ---------------------------------------------------------------------------
def call_fusion_api():
    data = {}
    if url.strip():
        data["url"] = url.strip()
    if sms_text.strip():
        data["sms_text"] = sms_text.strip()
    if email_raw.strip():
        data["email_raw"] = email_raw.strip()

    files = {}
    if qr_image is not None:
        files["qr_image"] = (qr_image.name, qr_image.getvalue(), qr_image.type)
    if payment_image is not None:
        files["payment_image"] = (payment_image.name, payment_image.getvalue(), payment_image.type)

    if not data and not files:
        st.warning("Please provide at least one input before analyzing.")
        return None

    try:
        response = requests.post(API_URL, data=data, files=files, timeout=120)
    except requests.exceptions.ConnectionError:
        st.error(
            "Could not reach the Fusion API. Make sure it's running:\n\n"
            "```\ncd fusion_engine\npython -m uvicorn api:app --reload\n```"
        )
        return None

    if response.status_code != 200:
        st.error(f"API error ({response.status_code}): {response.json().get('detail', response.text)}")
        return None

    return response.json()


VERDICT_LABEL = {
    "safe": "Safe",
    "suspicious": "Suspicious",
    "high_risk": "High Risk",
}


def render_verdict_card(verdict: str, score: float):
    label = VERDICT_LABEL.get(verdict, verdict.title())
    st.markdown(
        f"""
        <div class="verdict-card verdict-{verdict}">
            <div>
                <div class="verdict-label">Fusion verdict</div>
                <div class="verdict-text">{label}</div>
            </div>
            <div class="verdict-score">combined risk&nbsp;·&nbsp;{score:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chart(contributing_modules: list[dict]):
    df = pd.DataFrame(contributing_modules)
    df["risk_pct"] = df["risk_probability"] * 100
    colors = ["#FF6B4A" if lbl == "malicious" else "#3DDC97" for lbl in df["label"]]

    fig = go.Figure(
        go.Bar(
            x=df["risk_pct"],
            y=df["module"].str.upper(),
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in df["risk_pct"]],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono, monospace", color="#E7ECF5"),
        )
    )
    fig.update_layout(
        title=dict(text="Per-Module Risk Score", font=dict(family="Space Grotesk, sans-serif", color="#E7ECF5", size=16)),
        xaxis=dict(title="Risk Probability (%)", range=[0, 105], color="#8C9AB8", gridcolor="#2A3550"),
        yaxis=dict(title="", color="#E7ECF5"),
        height=90 + 60 * len(df),
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color="#E7ECF5"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_result(result: dict):
    verdict = result["verdict"]
    score = result["combined_risk_probability"]

    st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)

    gauge_col, card_col = st.columns([1, 1.3])
    with gauge_col:
        st.markdown(render_gauge_svg(score), unsafe_allow_html=True)
    with card_col:
        render_verdict_card(verdict, score)
        if result.get("override_triggered"):
            st.markdown(
                f'<div class="override-note">⚡ Safety override triggered by '
                f'<strong>{result["override_module"]}</strong> — a single module was '
                f'≥90% confident, so it drove the verdict directly.</div>',
                unsafe_allow_html=True,
            )
        st.write(result["explanation"])

    render_chart(result["contributing_modules"])

    st.markdown('<div class="section-label">Module Breakdown</div>', unsafe_allow_html=True)
    for m in result["contributing_modules"]:
        tag = "FLAGGED" if m["label"] == "malicious" else "CLEAR"
        with st.expander(
            f"{m['module'].upper()} — {tag} "
            f"({m['risk_probability']:.1%} risk, model weight: {m['learned_weight']:+.2f})"
        ):
            st.write(m.get("explanation", "No explanation available."))

    with st.expander("Raw JSON response"):
        st.json(result)


if submitted:
    with st.spinner("Running detection modules and fusing results..."):
        result = call_fusion_api()
    if result:
        render_result(result)

st.markdown("---")
st.caption(
    "Adaptive Risk Fusion — 5 independently trained detection modules "
    "(URL · SMS · QR · Payment Screenshot · Email) combined via a Logistic "
    "Regression meta-learner trained on each module's real held-out performance."
)