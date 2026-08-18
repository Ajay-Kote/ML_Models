"""
app.py — Streamlit demo frontend for Adaptive Risk Fusion.

Run this AFTER the fusion_engine API is running (uvicorn api:app --reload
on http://127.0.0.1:8000).

To run:
    pip install streamlit requests plotly pandas
    streamlit run app.py
"""

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
    st.header("ℹ️ About This Project")
    st.markdown(
        """
**Adaptive Risk Fusion** is a multi-modal fraud & phishing detection
system combining 5 independently trained detection modules:

| Module | Model |
|---|---|
| 🔗 URL | LightGBM (lexical + host features) |
| 💬 SMS | Fine-tuned DistilBERT |
| 📷 QR Code | LightGBM (pixel + metadata + texture) |
| 💳 Payment Screenshot | PaddleOCR + EfficientNet-B0 + LightGBM |
| ✉️ Email | DistilBERT + LightGBM + MLP fusion |

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
            st.success("🟢 Backend API: Connected")
        else:
            st.warning("🟡 Backend API: Unexpected response")
    except requests.exceptions.RequestException:
        st.error("🔴 Backend API: Not reachable")
        st.caption("Start it with:\n```\ncd fusion_engine\npython -m uvicorn api:app --reload\n```")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("🛡️ Adaptive Risk Fusion")
st.caption("Multi-modal phishing & fraud detection — URL, SMS, QR, Payment Screenshot, Email")

st.markdown("Fill in whichever inputs you have (you don't need all five), or try a quick example:")

ex_col1, ex_col2 = st.columns(2)
with ex_col1:
    st.button("🚩 Try Phishing Example", on_click=load_phishing_example, use_container_width=True)
with ex_col2:
    st.button("✅ Try Safe Example", on_click=load_safe_example, use_container_width=True)

with st.form("fusion_form"):
    col1, col2 = st.columns(2)

    with col1:
        url = st.text_input("🔗 URL", key="url_val", placeholder="http://example.com/login")
        sms_text = st.text_area("💬 SMS Text", key="sms_val", placeholder="Your OTP is 4521...", height=100)
        email_raw = st.text_area(
            "✉️ Raw Email (headers + body)",
            key="email_val",
            placeholder="From: sender@example.com\nSubject: ...\n\nBody text...",
            height=150,
        )

    with col2:
        qr_image = st.file_uploader("📷 QR Code Image", type=["png", "jpg", "jpeg"])
        payment_image = st.file_uploader("💳 Payment Screenshot", type=["png", "jpg", "jpeg"])

    submitted = st.form_submit_button("🔍 Analyze", use_container_width=True, type="primary")


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


VERDICT_STYLE = {
    "safe": ("🟢", "green"),
    "suspicious": ("🟡", "orange"),
    "high_risk": ("🔴", "red"),
}


def render_chart(contributing_modules: list[dict]):
    df = pd.DataFrame(contributing_modules)
    df["risk_pct"] = df["risk_probability"] * 100
    colors = ["#e74c3c" if lbl == "malicious" else "#2ecc71" for lbl in df["label"]]

    fig = go.Figure(
        go.Bar(
            x=df["risk_pct"],
            y=df["module"].str.upper(),
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in df["risk_pct"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Per-Module Risk Score",
        xaxis_title="Risk Probability (%)",
        yaxis_title="",
        xaxis_range=[0, 105],
        height=90 + 60 * len(df),
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_result(result: dict):
    verdict = result["verdict"]
    icon, color = VERDICT_STYLE.get(verdict, ("⚪", "gray"))
    score = result["combined_risk_probability"]

    st.markdown("---")
    st.markdown(f"## {icon} Verdict: :{color}[{verdict.upper().replace('_', ' ')}]")

    st.progress(min(score, 1.0), text=f"Combined risk score: {score:.1%}")

    if result.get("override_triggered"):
        st.info(
            f"⚡ **Safety override triggered** by the `{result['override_module']}` module — "
            f"a single module was highly confident (≥90%), so it drove the verdict directly."
        )

    st.write(result["explanation"])

    render_chart(result["contributing_modules"])

    st.markdown("### Module Breakdown")
    for m in result["contributing_modules"]:
        label_icon = "🚩" if m["label"] == "malicious" else "✅"
        with st.expander(
            f"{label_icon} **{m['module'].upper()}** — {m['label']} "
            f"({m['risk_probability']:.1%} risk, model weight: {m['learned_weight']:+.2f})"
        ):
            st.write(m.get("explanation", "No explanation available."))

    with st.expander("🔧 Raw JSON response"):
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