"""
Module 4 - Payment Image Detection
Error Level Analysis (ELA): a classic pixel-domain forensic technique for
detecting JPEG recompression / splice tampering that a generic
ImageNet-pretrained embedding (see vision/efficientnet_extractor.py) does
not capture, because that embedding is global-average-pooled and trained
for natural-image classification, not forensic artifact detection.

How it works: re-save the image at a known JPEG quality, then diff it
against the original pixel-for-pixel. Regions that were already
JPEG-compressed at a *different* quality (e.g. a pasted/spliced patch, or
a screenshot edited and re-exported) settle into a different "error level"
than untouched regions, showing up as localized brightness anomalies in
the diff map. We don't do pixel-level localization here (Grad-CAM already
covers visual localization) -- we summarize the diff map into a handful of
global statistics that get added to the LightGBM fusion feature vector.

No new dependency: uses only Pillow + numpy, both already required.
"""
import io
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image


@dataclass
class ELAFeatures:
    ela_mean: float = 0.0
    ela_std: float = 0.0
    ela_max: float = 0.0
    ela_p95: float = 0.0
    # fraction of pixels whose error level is a strong outlier vs. the rest
    # of the image -- uniform/global recompression raises ela_mean broadly,
    # while a *localized* splice/edit shows up as a small high-outlier patch,
    # so this ratio helps distinguish "whole image re-saved" from "patch
    # tampered" without needing full pixel-level localization here.
    ela_hotspot_ratio: float = 0.0


def compute_ela_features(image_path: str, quality: int = 90) -> ELAFeatures:
    """Computes ELA summary statistics for a single image.

    Returns default-zero features (rather than raising) on any decode
    failure, so a single bad image doesn't take down a whole training run --
    matching the fault-tolerance style already used in ocr/paddleocr_pipeline.py.
    """
    try:
        original = Image.open(image_path).convert("RGB")

        buf = io.BytesIO()
        original.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        resaved = Image.open(buf).convert("RGB")

        orig_arr = np.asarray(original, dtype=np.int16)
        resaved_arr = np.asarray(resaved, dtype=np.int16)

        # per-pixel error level = max channel-wise absolute difference
        diff = np.abs(orig_arr - resaved_arr).max(axis=2).astype(np.float32)

        mean = float(diff.mean())
        std = float(diff.std())
        outlier_threshold = mean + 2.0 * std
        hotspot_ratio = float((diff > outlier_threshold).mean()) if std > 0 else 0.0

        return ELAFeatures(
            ela_mean=mean,
            ela_std=std,
            ela_max=float(diff.max()),
            ela_p95=float(np.percentile(diff, 95)),
            ela_hotspot_ratio=hotspot_ratio,
        )
    except Exception as e:
        print(f"[ELA] Failed on {image_path}: {e}")
        return ELAFeatures()


def ela_features_to_row(features: ELAFeatures) -> dict:
    return asdict(features)
