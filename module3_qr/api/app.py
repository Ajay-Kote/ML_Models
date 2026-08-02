"""
app.py
------
Standalone FastAPI service for the QR Code Analysis Module. Exposes a single
POST /analyze-qr endpoint that accepts an uploaded QR image and returns the
malicious-QR probability, decoded URL, and SHAP-based explanation.

Run:
    uvicorn api.app:app --reload --port 8003
"""
import io
import os
import sys
import tempfile

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.predict import predict

app = FastAPI(title="QR Code Analysis Module", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok", "module": "qr_analysis"}


@app.post("/analyze-qr")
async def analyze_qr(file: UploadFile = File(...)):
    contents = await file.read()
    suffix = os.path.splitext(file.filename or "qr.png")[1] or ".png"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = predict(tmp_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    finally:
        os.remove(tmp_path)

    return result
