"""Per-tile segmentation metrics: class IoU, mIoU, Brier score, ECE.

The benchmark table reports the mean of these per-tile values over all
held-out tiles, folds and seeds.
"""

from __future__ import annotations

import numpy as np

from activity_index.schema import IGNORE_INDEX, NUM_CLASSES


def class_iou(pred: np.ndarray, target: np.ndarray, num_classes: int = NUM_CLASSES,
              ignore_index: int = IGNORE_INDEX) -> tuple[np.ndarray, float]:
    """Per-class IoU (NaN when a class is absent from both) and their nanmean."""
    valid = target != ignore_index
    pred, target = pred[valid].ravel(), target[valid].ravel()
    ious = np.full(num_classes, np.nan)
    for c in range(num_classes):
        p, t = pred == c, target == c
        union = int((p | t).sum())
        if union > 0:
            ious[c] = int((p & t).sum()) / union
    finite = ious[~np.isnan(ious)]
    return ious, float(finite.mean()) if finite.size else 0.0


def brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int = NUM_CLASSES,
                ignore_index: int = IGNORE_INDEX) -> float:
    """Mean over pixels of sum_c (p_c - y_c)^2. probs: (N, C), labels: (N,)."""
    keep = labels != ignore_index
    p = probs[keep].astype(np.float64)
    y = labels[keep]
    if p.size == 0:
        return float("nan")
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10,
                               ignore_index: int = IGNORE_INDEX) -> float:
    """Top-1 ECE with equal-width bins. probs: (N, C), labels: (N,)."""
    keep = labels != ignore_index
    if not keep.any():
        return float("nan")
    p, y = probs[keep], labels[keep]
    pred = p.argmax(axis=1)
    conf = p[np.arange(len(pred)), pred].astype(np.float64)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(conf, edges, right=False) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = ids == b
        if in_bin.any():
            ece += in_bin.sum() / len(conf) * abs(conf[in_bin].mean() - correct[in_bin].mean())
    return float(ece)


def tile_metrics(probs: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """All benchmark metrics for one tile. probs: (C, H, W), mask: (H, W)."""
    ious, miou = class_iou(probs.argmax(0), mask)
    flat_p = probs.reshape(probs.shape[0], -1).T
    flat_y = mask.reshape(-1).astype(np.int64)
    row = {
        "miou": miou,
        "brier": brier_score(flat_p, flat_y),
        "ece": expected_calibration_error(flat_p, flat_y),
    }
    for c, v in enumerate(ious):
        row[f"iou_{c}"] = float(v)
    return row
