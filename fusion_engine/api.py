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
import shutil
import sys
import tempfile
from time import perf_counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from adapters import adapt_url, adapt_sms, adapt_qr, adapt_image, adapt_email
from fuse import fuse, FusionInputError
from fuse_ml import fuse_ml, model_health

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


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Multi-Modal Fraud & Phishing Detection - Fusion API")

# CORS: allows a browser-based frontend (React/Streamlit dev server etc.,
# usually on a different port like localhost:3000 or localhost:8501) to call
# this API. "*" is fine for local development/demo; for a real deployment,
# replace with the specific frontend origin(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Uploaded QR/payment-screenshot images are written here temporarily so the
# existing predict() functions (which expect a file PATH, not raw bytes)
# can be reused unchanged. Files are deleted again right after prediction.
UPLOAD_TMP_DIR = Path(tempfile.gettempdir()) / "fusion_engine_uploads"
UPLOAD_TMP_DIR.mkdir(exist_ok=True)
_prediction_count = 0
_prediction_latency_ms = 0.0


def _save_upload_to_temp(upload: UploadFile) -> Path:
    suffix = Path(upload.filename).suffix or ".png"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=UPLOAD_TMP_DIR)
    with os.fdopen(fd, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return Path(tmp_path)


@app.get("/")
def root():
    return {"status": "Fusion API running"}


@app.get("/model-health")
def get_model_health():
    """Expose runtime health and existing model metadata without retraining."""
    global _prediction_count, _prediction_latency_ms
    try:
        health = model_health()
        model_status = "nominal"
    except (FileNotFoundError, KeyError, AttributeError) as error:
        health = {"model_loaded": False, "model_file": "meta_learner.joblib", "modules": []}
        model_status = "degraded"
        health["error"] = str(error)

    average_latency_ms = (
        round(_prediction_latency_ms / _prediction_count, 2)
        if _prediction_count else None
    )
    return {
        "status": model_status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "prediction_count": _prediction_count,
        "average_latency_ms": average_latency_ms,
        "evaluation_metrics": {
            "accuracy": None,
            "f1": None,
            "roc_auc": None,
        },
        **health,
    }


@app.post("/predict")
def predict(
    url: Optional[str] = Form(None),
    sms_text: Optional[str] = Form(None),
    email_raw: Optional[str] = Form(None),
    qr_image: Optional[UploadFile] = File(None),
    payment_image: Optional[UploadFile] = File(None),
):
    global _prediction_count, _prediction_latency_ms
    started_at = perf_counter()
    module_outputs = {}
    temp_files_to_clean = []

    try:
        if url:
            raw = url_predict.predict_url(url)
            module_outputs["url"] = adapt_url(raw)

        if sms_text:
            raw = _sms_detector.predict(sms_text)
            module_outputs["sms"] = adapt_sms(raw)

        if qr_image is not None:
            qr_path = _save_upload_to_temp(qr_image)
            temp_files_to_clean.append(qr_path)
            raw = qr_predict.predict(str(qr_path))
            module_outputs["qr"] = adapt_qr(raw)

        if payment_image is not None:
            image_path = _save_upload_to_temp(payment_image)
            temp_files_to_clean.append(image_path)
            raw = _image_detector.predict(str(image_path))
            module_outputs["image"] = adapt_image(raw)

        if email_raw:
            raw = email_predict.predict(email_raw)
            module_outputs["email"] = adapt_email(raw)

        if not module_outputs:
            raise HTTPException(
                status_code=422,
                detail="Provide at least one of: url, sms_text, qr_image, "
                       "payment_image, email_raw.",
            )

        try:
            result = fuse_ml(module_outputs)  # ML meta-learner fusion (primary)
        except FileNotFoundError:
            # meta_learner.joblib missing/corrupted -> fall back to rule-based
            # weighted-average fusion so the API never goes down
            try:
                result = fuse(module_outputs)
            except FusionInputError as e:
                raise HTTPException(status_code=422, detail=str(e))
        except FusionInputError as e:
            raise HTTPException(status_code=422, detail=str(e))

        _prediction_count += 1
        _prediction_latency_ms += (perf_counter() - started_at) * 1000
        return result

    finally:
        # Always clean up temp files, even if prediction raised an error.
        for p in temp_files_to_clean:
            p.unlink(missing_ok=True)