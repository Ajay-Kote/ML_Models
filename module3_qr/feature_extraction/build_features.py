"""
build_features.py
------------------
Concatenates pixel + QR-metadata + texture features into the single feature
vector consumed by the LightGBM classifier, per Section 5.3 ("Model: LightGBM
trained on concatenated pixel + metadata + texture features").
"""
import numpy as np

from feature_extraction.pixel import extract_pixel_features, pixel_feature_names
from feature_extraction.metadata import extract_metadata_features, FEATURE_NAMES as META_NAMES
from feature_extraction.texture import extract_texture_features, FEATURE_NAMES as TEX_NAMES


def feature_names():
    return pixel_feature_names() + META_NAMES + TEX_NAMES


def build_feature_vector(image_path, manifest_row=None):
    """
    image_path: path (or ndarray) to the QR image
    manifest_row: optional dict with generation-time ground-truth metadata
                  (used during training). If None, metadata is estimated
                  directly from the image (inference path).
    """
    px = extract_pixel_features(image_path)
    if manifest_row is not None:
        meta = extract_metadata_features(manifest_row, from_generation=True)
    else:
        meta = extract_metadata_features(image_path, from_generation=False)
    tex = extract_texture_features(image_path)
    return np.concatenate([px, meta, tex]).astype(np.float32)
