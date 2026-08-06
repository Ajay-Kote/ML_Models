"""
fusion_engine/api.py

FastAPI orchestrator: accepts any combination of inputs (URL, SMS text,
QR image, payment screenshot, raw email), calls only the relevant module(s),
normalizes each output via adapters.py, and combines everything via fuse.py.

Run:
    cd fusion_engine
    python -m uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from adapters import adapt_url, adapt_sms, adapt_qr, adapt_image, adapt_email
from fuse import fuse, FusionInputError

# ---------------------------------------------------------------------------
# MODULE_PATHS -- confirmed against actual folder layout
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent  # the 5modules/ folder

URL_MODULE_DIR = ROOT / "module1_url" / "models"
SMS_MODULE_DIR = ROOT / "module2_sms" / "model"
QR_PREDICT_DIR = ROOT / "module3_qr" / "models"
IMAGE_MODULE_DIR = ROOT / "module4_image" / "models"
EMAIL_PREDICT_DIR = ROOT / "module5_email"


def _load_module_from_path(module_name: str, file_path: Path):
    """Import a predict.py file directly by path, regardless of package structure."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path}. Check MODULE_PATHS at the top of api.py."
        )
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Several modules define same-named packages (feature_extraction,
# explainability, fusion, etc.). Once Python imports "feature_extraction"
# from module1_url, it caches it in sys.modules -- so when module3_qr later
# does `from feature_extraction... import ...`, Python hands back URL's
# cached version instead of QR's, causing ModuleNotFoundError /
# AttributeError. _purge_generic_modules() clears these shared names out of
# the cache right before loading each module, so every module gets its own.
# ---------------------------------------------------------------------------
_GENERIC_PACKAGE_NAMES = [
    "feature_extraction", "explainability", "fusion", "ocr", "vision",
    "metadata_branch", "text_branch",
]


def _purge_generic_modules():
    for name in list(sys.modules.keys()):
        top = name.split(".")[0]
        if top in _GENERIC_PACKAGE_NAMES:
            del sys.modules[name]


# --- URL module ---
_purge_generic_modules()
sys.path.insert(0, str(URL_MODULE_DIR.parent))
sys.path.insert(0, str(URL_MODULE_DIR))
url_predict = _load_module_from_path("url_predict", URL_MODULE_DIR / "predict.py")

# --- SMS module ---
_purge_generic_modules()
sys.path.insert(0, str(SMS_MODULE_DIR))
sms_predict = _load_module_from_path("sms_predict", SMS_MODULE_DIR / "predict.py")
_sms_detector = sms_predict.SmishingDetector()

# --- QR module ---
_purge_generic_modules()
sys.path.insert(0, str(QR_PREDICT_DIR.parent))
sys.path.insert(0, str(QR_PREDICT_DIR))
qr_predict = _load_module_from_path("qr_predict", QR_PREDICT_DIR / "predict.py")

# --- Payment image module ---
_purge_generic_modules()
sys.path.insert(0, str(IMAGE_MODULE_DIR.parent))
sys.path.insert(0, str(IMAGE_MODULE_DIR))
image_predict = _load_module_from_path("image_predict", IMAGE_MODULE_DIR / "predict.py")
_image_detector = image_predict.PaymentImageDetector(
    artifacts_dir=str(IMAGE_MODULE_DIR / "artifacts")
)

# --- Email module ---
_purge_generic_modules()
sys.path.insert(0, str(EMAIL_PREDICT_DIR.parent))
sys.path.insert(0, str(EMAIL_PREDICT_DIR))
email_predict = _load_module_from_path("email_predict", EMAIL_PREDICT_DIR / "predict.py")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Multi-Modal Fraud & Phishing Detection - Fusion API")


class FusionRequest(BaseModel):
    url: Optional[str] = None
    sms_text: Optional[str] = None
    qr_image_path: Optional[str] = None
    payment_image_path: Optional[str] = None
    email_raw: Optional[str] = None


@app.get("/")
def root():
    return {"status": "Fusion API running"}


@app.post("/predict")
def predict(request: FusionRequest):
    module_outputs = {}

    if request.url:
        raw = url_predict.predict_url(request.url)
        module_outputs["url"] = adapt_url(raw)

    if request.sms_text:
        raw = _sms_detector.predict(request.sms_text)
        module_outputs["sms"] = adapt_sms(raw)

    if request.qr_image_path:
        raw = qr_predict.predict(request.qr_image_path)
        module_outputs["qr"] = adapt_qr(raw)

    if request.payment_image_path:
        raw = _image_detector.predict(request.payment_image_path)
        module_outputs["image"] = adapt_image(raw)

    if request.email_raw:
        raw = email_predict.predict(request.email_raw)
        module_outputs["email"] = adapt_email(raw)

    if not module_outputs:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one of: url, sms_text, qr_image_path, "
                   "payment_image_path, email_raw.",
        )

    try:
        result = fuse(module_outputs)
    except FusionInputError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return result