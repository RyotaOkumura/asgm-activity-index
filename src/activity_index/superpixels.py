"""SLIC superpixels and superpixel averaging of class probabilities.

Two superpixel sets are used, as in the paper:

* ``reflectance``: SLIC on B08/B04/B03 reflectance (/ 10000, clipped to [0, 1],
  then min-max scaled per image). Used for the segmentation benchmark.
* ``false_color``: SLIC on the NIR-R-G false-color image
  ``clip(band / 4000, 0, 1) * 255`` (B08/B04/B03, uint8). Used for the
  Activity Index on every year.

Both use n_segments=1350, compactness=10, sigma=1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.segmentation import slic

N_SEGMENTS = 1350
COMPACTNESS = 10.0
SIGMA = 1.0

REFLECTANCE_BANDS = ["B08", "B04", "B03"]
FALSE_COLOR_BANDS = ["B08", "B04", "B03"]
FALSE_COLOR_DIV = 4000.0
VARIANTS = ("reflectance", "false_color")


def _contiguous(seg: np.ndarray) -> np.ndarray:
    """Relabel to 0..n-1."""
    unique, inverse = np.unique(seg, return_inverse=True)
    if not np.array_equal(unique, np.arange(len(unique))):
        seg = inverse.reshape(seg.shape)
    return seg.astype(np.int32)


def slic_reflectance(raw: np.ndarray, n_segments: int = N_SEGMENTS) -> np.ndarray:
    """SLIC labels (H, W) from raw B08/B04/B03 reflectance (3, H, W)."""
    image = np.clip(raw.astype(np.float32) / 10000.0, 0.0, 1.0)
    hwc = np.transpose(image, (1, 2, 0)).astype(np.float64)
    lo, hi = float(hwc.min()), float(hwc.max())
    if hi > lo:
        hwc = (hwc - lo) / (hi - lo)
    seg = slic(hwc, n_segments=n_segments, compactness=COMPACTNESS, sigma=SIGMA,
               channel_axis=-1, start_label=0)
    return _contiguous(seg)


def false_color(raw: np.ndarray) -> np.ndarray:
    """NIR-R-G false color (H, W, 3) uint8 from raw B08/B04/B03 (3, H, W)."""
    hwc = np.transpose(raw.astype(np.float32), (1, 2, 0))
    return (np.clip(hwc / FALSE_COLOR_DIV, 0, 1) * 255).astype(np.uint8)


def slic_false_color(raw: np.ndarray, n_segments: int = N_SEGMENTS) -> np.ndarray:
    """SLIC labels (H, W) from raw B08/B04/B03 reflectance (3, H, W)."""
    image = false_color(raw).astype(np.float32) / 255.0
    seg = slic(image, n_segments=n_segments, compactness=COMPACTNESS, sigma=SIGMA,
               channel_axis=-1, start_label=0)
    return _contiguous(seg)


def superpixel_path(sp_root: Path, variant: str, site: str, stem: str) -> Path:
    return Path(sp_root) / variant / site / f"{stem}.npz"


def load_superpixels(sp_root: Path, variant: str, site: str, stem: str) -> np.ndarray:
    with np.load(superpixel_path(sp_root, variant, site, stem)) as d:
        return np.asarray(d["segments"], dtype=np.int32)


def average_over_superpixels(probs: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Replace each pixel's class probabilities by its superpixel mean.

    probs: (C, H, W), segments: (H, W) with ids 0..n-1. Returns (C, H, W) float32.
    """
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
