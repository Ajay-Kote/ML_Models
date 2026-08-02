"""
metadata.py
-----------
QR structural metadata feature extraction (Section 5.3 "QR metadata" row).

Two operating modes:
  1. `from_manifest_row` -- when generation-time ground truth is known
     (training pipeline, since we synthesize the dataset ourselves).
  2. `from_image` -- best-effort estimation directly from a QR PNG at
     inference time, when only the raw image is available (production path).
     We re-detect the finder/module grid with OpenCV's QRCodeDetector and
     derive version, approximate module density, and quiet-zone size from
     pixel geometry. Error-correction level cannot be recovered from a
     decoded image without also parsing the format-information bits, so a
     conservative "unknown" placeholder (mapped to the dataset's neutral
     class) is used with a low-confidence flag the fusion layer can weigh
     accordingly.
"""
import cv2
import numpy as np

EC_MAP = {"L": 0, "M": 1, "Q": 2, "H": 3, "UNKNOWN": 1}  # ordinal, low->high robustness
FEATURE_NAMES = [
    "qr_version",
    "error_correction_level",   # ordinal 0=L .. 3=H
    "module_density",           # modules-per-side estimate normalized
    "quiet_zone_size",          # border size in modules
    "metadata_confidence",      # 1.0 if exact (from generation), <1.0 if estimated
]


def from_manifest_row(row):
    version = row["qr_version"]
    ec = EC_MAP.get(row["error_correction"], 1)
    modules_per_side = 21 + 4 * (version - 1)  # QR spec: version N -> (17+4N) modules... see note below
    density = modules_per_side / 177.0  # normalize against max (version 40 = 177 modules/side)
    quiet_zone = row["border"]
    return np.array([version, ec, density, quiet_zone, 1.0], dtype=np.float32)


def from_image(path_or_array):
    """Best-effort metadata extraction from a bare QR image (no generation params known)."""
    if isinstance(path_or_array, np.ndarray):
        img = path_or_array
    else:
        img = cv2.imread(str(path_or_array), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path_or_array}")

    detector = cv2.QRCodeDetector()
    ok, points = detector.detect(img)

    h, w = img.shape[:2]
    if ok and points is not None:
        pts = points.reshape(-1, 2)
        side_px = np.mean([
            np.linalg.norm(pts[0] - pts[1]),
            np.linalg.norm(pts[1] - pts[2]),
        ])
        # Estimate module count from finder-pattern spacing heuristics; fall back
        # to a mid-range default when detection is too imprecise to trust.
        approx_modules = np.clip(side_px / max(w, h) * 177, 21, 177)
        version = max(1, round((approx_modules - 21) / 4) + 1)
        density = approx_modules / 177.0
        quiet_zone = 4  # cannot be measured post-hoc reliably; use QR-spec recommended default
        confidence = 0.6
    else:
        version, density, quiet_zone, confidence = 10, 0.35, 4, 0.3

    ec = EC_MAP["UNKNOWN"]
    return np.array([version, ec, density, quiet_zone, confidence], dtype=np.float32)


def extract_metadata_features(row_or_path, from_generation=False):
    if from_generation:
        return from_manifest_row(row_or_path)
    return from_image(row_or_path)
