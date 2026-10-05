"""SLIC superpixels and superpixel averaging of class probabilities.

Superpixels are computed on the NIR-R-G false-color image
clip(band / 4000, 0, 1) * 255 (B08, B04, B03), the image the annotators
labeled. They are the annotation unit of Pan-ASGM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.segmentation import slic

N_SEGMENTS = 1350
COMPACTNESS = 10.0
SIGMA = 1.0
BANDS = ["B08", "B04", "B03"]
FALSE_COLOR_DIV = 4000.0


def false_color(raw: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 false color from raw B08/B04/B03 reflectance (3, H, W)."""
    hwc = np.transpose(raw.astype(np.float32), (1, 2, 0))
    return (np.clip(hwc / FALSE_COLOR_DIV, 0, 1) * 255).astype(np.uint8)


def compute_superpixels(raw: np.ndarray, n_segments: int = N_SEGMENTS) -> np.ndarray:
    """(H, W) int32 labels 0..n-1 from raw B08/B04/B03 reflectance (3, H, W)."""
    image = false_color(raw).astype(np.float32) / 255.0
    seg = slic(image, n_segments=n_segments, compactness=COMPACTNESS, sigma=SIGMA,
               channel_axis=-1, start_label=0)
    unique, inverse = np.unique(seg, return_inverse=True)
    if not np.array_equal(unique, np.arange(len(unique))):
        seg = inverse.reshape(seg.shape)
    return seg.astype(np.int32)


def superpixel_path(sp_root: Path, site: str, stem: str) -> Path:
    return Path(sp_root) / site / f"{stem}.npz"


def load_superpixels(sp_root: Path, site: str, stem: str) -> np.ndarray:
    with np.load(superpixel_path(sp_root, site, stem)) as d:
        return np.asarray(d["segments"], dtype=np.int32)


def average_over_superpixels(probs: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Replace each pixel's class probabilities (C, H, W) by its superpixel mean."""
    C, H, W = probs.shape
    if segments.shape != (H, W):
        raise ValueError(f"segments {segments.shape} != probs {(H, W)}")
    flat = segments.astype(np.int64).reshape(-1)
    n_sp = int(flat.max()) + 1
    count = np.maximum(np.bincount(flat, minlength=n_sp).astype(np.float64), 1.0)
    mean = np.empty((C, n_sp), dtype=np.float64)
    for c in range(C):
        mean[c] = np.bincount(flat, weights=probs[c].reshape(-1).astype(np.float64),
                              minlength=n_sp) / count
    return mean[:, flat].reshape(C, H, W).astype(np.float32)
