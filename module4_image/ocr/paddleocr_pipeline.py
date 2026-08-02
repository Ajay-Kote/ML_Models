"""
Module 4 - Payment Image Detection
OCR Pipeline: PaddleOCR text extraction + regex-based structured field parsing
             + OCR quality signal computation (Section 5.4 of the design doc).

Output of extract() feeds directly into fusion/feature_fusion.py
"""

import re
import statistics
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from paddleocr import PaddleOCR

# ---------------------------------------------------------------------------
# Regex patterns for common UPI / payment-app screenshot fields
# (Google Pay, PhonePe, Paytm style layouts). Extend as new formats are seen.
# ---------------------------------------------------------------------------
AMOUNT_RE = re.compile(r"(?:₹|Rs\.?|INR)\s?([\d,]+\.?\d{0,2})", re.IGNORECASE)
UPI_ID_RE = re.compile(r"[a-zA-Z0-9.\-_]{2,}@[a-zA-Z]{2,}")
TXN_ID_RE = re.compile(
    r"(?:txn|transaction|ref(?:erence)?|utr)\s*(?:id|no\.?)?[:\s]*([A-Z0-9]{10,35})",
    re.IGNORECASE,
)
TIMESTAMP_RE = re.compile(
    r"(\d{1,2}[:.]\d{2}\s?(?:AM|PM)?,?\s?\d{1,2}\s?[A-Za-z]{3,9}\s?\d{2,4})"
    r"|(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s?\d{0,2}[:.]?\d{0,2})",
    re.IGNORECASE,
)
STATUS_RE = re.compile(
    r"\b(success(?:ful)?|completed|failed|pending|declined)\b", re.IGNORECASE
)
KNOWN_APPS = [
    "google pay", "gpay", "phonepe", "paytm", "amazon pay",
    "bhim", "whatsapp pay", "cred", "mobikwik",
]


@dataclass
class OCRStructuredFields:
    amount: Optional[float] = None
    upi_id: Optional[str] = None
    transaction_id: Optional[str] = None
    timestamp: Optional[str] = None
    bank_or_app_name: Optional[str] = None
    status_text: Optional[str] = None


@dataclass
class OCRQualitySignals:
    avg_confidence: float = 0.0
    min_confidence: float = 0.0
    num_text_boxes: int = 0
    font_size_std: float = 0.0          # spacing/font anomaly proxy
    line_spacing_std: float = 0.0
    low_confidence_ratio: float = 0.0   # fraction of boxes below 0.6 conf


@dataclass
class OCRResult:
    raw_text: str
    structured: OCRStructuredFields = field(default_factory=OCRStructuredFields)
    quality: OCRQualitySignals = field(default_factory=OCRQualitySignals)


class PaymentOCRPipeline:
    """Wraps PaddleOCR and turns raw detections into the structured
    fields + quality signals consumed by the fusion layer."""

    def __init__(self, lang: str = "en", device: str = "auto"):
        # PaddleOCR 3.x renamed/removed several constructor args:
        #   use_angle_cls -> use_textline_orientation
        #   use_gpu       -> device ("cpu" / "gpu:0")
        #   show_log      -> removed
        #
        # NOTE on enable_mkldnn: PaddlePaddle 3.3.0/3.3.1 has a known CPU bug
        # where MKL-DNN crashes with "ConvertPirAttribute2RuntimeAttribute not
        # support [pir::ArrayAttribute...]" (Paddle issue #77340). We pin
        # paddlepaddle==3.2.2 in requirements.txt specifically to avoid that
        # bug, so MKL-DNN can stay ON here (its default). Leaving it enabled
        # matters a lot: with it OFF, CPU inference falls back to an
        # unoptimized path that is roughly 30-40x slower (minutes vs hours
        # for a few hundred images). If you ever upgrade paddlepaddle past
        # 3.2.x, re-test this -- you may need enable_mkldnn=False again, at
        # the cost of that same slowdown.
        if device == "auto":
            device = self._detect_device()

        self.device = device
        print(f"[PaymentOCRPipeline] Using device: {device}")

        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            lang=lang,
            device=device,
        )

    @staticmethod
    def _detect_device() -> str:
        """
        Auto-detect GPU availability for PaddleOCR specifically.
        IMPORTANT: this requires the `paddlepaddle-gpu` pip package, not the
        plain `paddlepaddle` package -- having an NVIDIA GPU in the machine
        is not enough on its own. If paddle was installed CPU-only, this
        will correctly fall back to "cpu" even with a GPU present.
        """
        try:
            import paddle
            if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
                return "gpu:0"
        except Exception:
            pass
        return "cpu"

    # ------------------------------------------------------------------
    def extract(self, image_path: str) -> OCRResult:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        # PaddleOCR 3.x: .ocr() is gone, use .predict() which returns a
        # list of result objects (one per page/image) with dict-style access.
        results = self.ocr.predict(image_path)
        boxes, texts, confidences = self._flatten(results)

        raw_text = " ".join(texts)
        structured = self._parse_fields(raw_text)
        quality = self._quality_signals(boxes, confidences)

        return OCRResult(raw_text=raw_text, structured=structured, quality=quality)

    # ------------------------------------------------------------------
    @staticmethod
    def _flatten(results):
        """PaddleOCR 3.x result object exposes rec_texts / rec_scores /
        rec_boxes (each box = [x_min, y_min, x_max, y_max]).
        rec_boxes comes back as a numpy array, which raises "ambiguous
        truth value" if you try `x or []` on it — check `is None` instead."""
        if not results:
            return [], [], []
        res = results[0]

        raw_texts = res.get("rec_texts")
        texts = list(raw_texts) if raw_texts is not None else []

        raw_scores = res.get("rec_scores")
        confidences = list(raw_scores) if raw_scores is not None else []

        raw_boxes = res.get("rec_boxes")
        boxes = list(raw_boxes) if raw_boxes is not None else []

        return boxes, texts, confidences

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_fields(raw_text: str) -> OCRStructuredFields:
        amount_match = AMOUNT_RE.search(raw_text)
        amount = None
        if amount_match:
            try:
                amount = float(amount_match.group(1).replace(",", ""))
            except ValueError:
                amount = None

        upi_match = UPI_ID_RE.search(raw_text)
        # Find all txn matches and pick the longest one that looks like a real ID
        txn_matches = TXN_ID_RE.findall(raw_text)
        txn_id = None
        if txn_matches:
            # Sort by length descending and pick the first one that isn't just a common word
            valid_ids = [m for m in txn_matches if not m.lower() in ["successful", "failed", "pending"]]
            if valid_ids:
                txn_id = max(valid_ids, key=len)
        ts_match = TIMESTAMP_RE.search(raw_text)
        status_match = STATUS_RE.search(raw_text)

        app_name = None
        lowered = raw_text.lower()
        for app in KNOWN_APPS:
            if app in lowered:
                app_name = app
                break

        return OCRStructuredFields(
            amount=amount,
            upi_id=upi_match.group(0) if upi_match else None,
            transaction_id=txn_id,
            timestamp=(ts_match.group(0) if ts_match else None),
            bank_or_app_name=app_name,
            status_text=status_match.group(0).lower() if status_match else None,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _quality_signals(boxes, confidences) -> OCRQualitySignals:
        if not confidences:
            return OCRQualitySignals()

        avg_conf = statistics.mean(confidences)
        min_conf = min(confidences)
        low_conf_ratio = sum(1 for c in confidences if c < 0.6) / len(confidences)

        # Font-size proxy: box height variance across detected text lines.
        # boxes are [x_min, y_min, x_max, y_max] rectangles (PaddleOCR 3.x rec_boxes).
        heights = []
        y_centers = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            heights.append(float(y_max - y_min))
            y_centers.append(float((y_min + y_max) / 2))

        font_size_std = float(np.std(heights)) if len(heights) > 1 else 0.0

        # Line-spacing proxy: variance of consecutive y-center gaps.
        y_centers_sorted = sorted(y_centers)
        gaps = [
            y_centers_sorted[i + 1] - y_centers_sorted[i]
            for i in range(len(y_centers_sorted) - 1)
        ]
        line_spacing_std = float(np.std(gaps)) if len(gaps) > 1 else 0.0

        return OCRQualitySignals(
            avg_confidence=avg_conf,
            min_confidence=min_conf,
            num_text_boxes=len(boxes),
            font_size_std=font_size_std,
            line_spacing_std=line_spacing_std,
            low_confidence_ratio=low_conf_ratio,
        )


if __name__ == "__main__":
    import sys

    pipeline = PaymentOCRPipeline()
    res = pipeline.extract(sys.argv[1])
    print("Structured fields:", res.structured)
    print("Quality signals:", res.quality)
