"""
texture.py
----------
Visual texture feature extraction for the QR Code Analysis Module
(Section 5.3 "Visual texture" row): Local Binary Patterns, edge density,
and contrast/entropy measures.
"""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.filters import sobel

LBP_POINTS = 8
LBP_RADIUS = 1
LBP_BINS = LBP_POINTS + 2  # uniform LBP bin count

FEATURE_NAMES = [f"lbp_bin_{i}" for i in range(LBP_BINS)] + [
    "edge_density", "contrast", "entropy",
]


def _load_gray(path_or_array):
    if isinstance(path_or_array, np.ndarray):
        img = path_or_array
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.imread(str(path_or_array), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path_or_array}")
    return img


def extract_texture_features(path_or_array, resize_to=128):
    img = _load_gray(path_or_array)
    img = cv2.resize(img, (resize_to, resize_to), interpolation=cv2.INTER_AREA)

    # Local Binary Pattern histogram (uniform patterns) — captures the
    # fine-grained black/white module structure and any printing/compression
    # artifacts that differ between clean-generated and screenshotted/forged codes.
    lbp = local_binary_pattern(img, P=LBP_POINTS, R=LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(0, LBP_BINS + 1), density=True)

    # Edge density via Sobel gradient magnitude threshold
    edges = sobel(img.astype(np.float32) / 255.0)
    edge_density = float((edges > edges.mean()).mean())

    # Contrast (std of pixel intensities) and Shannon entropy of intensity histogram
    contrast = float(img.std() / 255.0)
    hist_counts, _ = np.histogram(img, bins=256, range=(0, 255), density=True)
    hist_counts = hist_counts[hist_counts > 0]
    entropy = float(-np.sum(hist_counts * np.log2(hist_counts)))
    entropy_norm = entropy / 8.0  # normalize against max entropy for 8-bit image

    return np.concatenate([hist.astype(np.float32), [edge_density, contrast, entropy_norm]]).astype(np.float32)
