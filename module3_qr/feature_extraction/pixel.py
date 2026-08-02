"""
pixel.py
--------
Pixel-level feature extraction for the QR Code Analysis Module (Section 5.3).

Produces a fixed-length, flattened grayscale pixel vector so that images of
varying native resolution (different QR versions / box sizes) can all be fed
into the same LightGBM feature vector.
"""
import cv2
import numpy as np

PIXEL_SIZE = 32  # resize target -> 32*32 = 1024 pixel features


def load_grayscale(path_or_array):
    if isinstance(path_or_array, np.ndarray):
        img = path_or_array
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.imread(str(path_or_array), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path_or_array}")
    return img


def extract_pixel_features(path_or_array, size=PIXEL_SIZE):
    """
    Returns a flattened, normalized (0-1) grayscale pixel matrix of shape
    (size*size,) resized with area interpolation (best for downscaling
    high-contrast binary QR patterns without aliasing artifacts).
    """
    img = load_grayscale(path_or_array)
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    flat = (resized.astype(np.float32) / 255.0).flatten()
    return flat


def pixel_feature_names(size=PIXEL_SIZE):
    return [f"px_{i:04d}" for i in range(size * size)]
