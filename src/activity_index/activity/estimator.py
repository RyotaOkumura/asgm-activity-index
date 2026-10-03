"""The Activity Index estimator  A = M o T o A_smooth.

Input per tile: yearly ``active_pond`` probabilities p (T, H, W) after
superpixel averaging, and a cloud mask (T, H, W).

* A (smoothing): cloud-weighted Gaussian (normalized convolution), sigma = 9 px.
  Cloudy pixels borrow evidence from cloud-free neighbors.
* T (trend): ordinary least squares over a trailing window of w = 3 years.
  Every year whose smoothed value is finite gets weight 1. Returns the
  slope beta and its standard error SE for each window end year.
* M (reliability filter): keep beta where |beta| > z * SE, z = 2.

The smoothed series is rounded through float16, as in the cached
intermediates that produced the paper numbers.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

SIGMA_A = 9.0
TRUNCATE = 3.0
TRAILING_W = 3
N_MIN = 3  # minimum valid years in a window for a finite slope
Z = 2.0


def gaussian_mean(arr: np.ndarray, valid: np.ndarray, sigma: float = SIGMA_A,
                  truncate: float = TRUNCATE) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware Gaussian smoothing per frame.

    mean = G * (arr * valid) / G * valid, NaN where the denominator is ~0.
    Returns (mean, kernel support G * valid), both (T, H, W) float32.
    """
    mean = np.empty(arr.shape, dtype=np.float32)
    support = np.empty(arr.shape, dtype=np.float32)
    for t in range(arr.shape[0]):
        v = valid[t].astype(np.float32)
        num = gaussian_filter(arr[t].astype(np.float32) * v, sigma=sigma, truncate=truncate, mode="reflect")
        den = gaussian_filter(v, sigma=sigma, truncate=truncate, mode="reflect")
        with np.errstate(invalid="ignore", divide="ignore"):
            mean[t] = np.where(den > 1e-6, num / den, np.nan)
        support[t] = den
    return mean, support


def smooth(p: np.ndarray, cloud_mask: np.ndarray, sigma: float = SIGMA_A,
           truncate: float = TRUNCATE) -> tuple[np.ndarray, np.ndarray]:
    """Operator A. Returns q (T, H, W) and the mean kernel support n_eff (H, W)."""
    q, support = gaussian_mean(p.astype(np.float32), ~cloud_mask, sigma, truncate)
    n_eff = np.nanmean(support, axis=0).astype(np.float32)
    return q.astype(np.float16).astype(np.float32), n_eff


def weighted_ols(years: np.ndarray, ts: np.ndarray, weights: np.ndarray):
    """Weighted least-squares line per row.

    years: (T,), ts and weights: (N, T). Returns slope (N,), residual sum of
    squares (N,), and the number of observations with weight > 0 (N,).
    Slope and RSS are NaN with fewer than 2 observations.
    """
    eps = 1e-9
    w_sum = weights.sum(axis=1)
    yr_mean = (weights * years[None, :]).sum(axis=1) / np.maximum(w_sum, eps)
    ts_mean = (weights * ts).sum(axis=1) / np.maximum(w_sum, eps)
    dy = years[None, :] - yr_mean[:, None]
    dt = ts - ts_mean[:, None]
    den = (weights * dy * dy).sum(axis=1)
    slope = (weights * dy * dt).sum(axis=1) / np.maximum(den, eps)
    rss = (weights * (ts - (ts_mean[:, None] + slope[:, None] * dy)) ** 2).sum(axis=1)
    n_obs = (weights > 0).sum(axis=1)
    bad = (den < eps) | (n_obs < 2)
    slope[bad] = np.nan
    rss[bad] = np.nan
    return slope, rss, n_obs


def trailing_ols(years: np.ndarray, q: np.ndarray, weights: np.ndarray,
                 w: int = TRAILING_W, n_min: int = N_MIN):
    """Operator T. q and weights: (T, N).

    Returns slopes (T-w+1, N), standard errors (T-w+1, N), and the end year
    of each window (T-w+1,).
    """
    n_win = q.shape[0] - w + 1
    slopes = np.full((n_win, q.shape[1]), np.nan, dtype=np.float32)
    se = np.full_like(slopes, np.nan)
    ends = np.zeros(n_win, dtype=np.float32)
    for i in range(n_win):
        ys = years[i:i + w]
        ts_w, w_w = q[i:i + w].T, weights[i:i + w].T
        slope, rss, n_obs = weighted_ols(ys, ts_w, w_w)
        with np.errstate(invalid="ignore", divide="ignore"):
            yr_mean = (w_w * ys[None, :]).sum(axis=1) / np.maximum(w_w.sum(axis=1), 1e-9)
            dy = ys[None, :] - yr_mean[:, None]
            den = (w_w * dy * dy).sum(axis=1)
            se_i = np.sqrt(rss / np.maximum(n_obs - 2, 1) / np.maximum(den, 1e-9))
        ok = n_obs >= n_min
        slopes[i] = np.where(ok, slope, np.nan)
        se[i] = np.where(ok, se_i, np.nan)
        ends[i] = ys[-1]
    return slopes, se, ends


def activity_index(years: np.ndarray, p: np.ndarray, cloud_mask: np.ndarray,
                   sigma: float = SIGMA_A, w: int = TRAILING_W, truncate: float = TRUNCATE):
    """A then T for one tile.

    Returns beta (n, H, W), SE (n, H, W), window end years (n,), and the mean
    kernel support n_eff (H, W). Apply ``reliable`` for operator M.
    """
    q, n_eff = smooth(p, cloud_mask, sigma, truncate)
    T, H, W = q.shape
    q_tn = q.reshape(T, -1)
    weights = np.isfinite(q_tn).astype(np.float32)
    q_tn = np.where(np.isfinite(q_tn), q_tn, 0.0).astype(np.float32)
    beta, se, ends = trailing_ols(years, q_tn, weights, w=w)
    n = beta.shape[0]
    return beta.reshape(n, H, W), se.reshape(n, H, W), ends, n_eff


def reliable(beta: np.ndarray, se: np.ndarray, z: float = Z) -> np.ndarray:
    """Operator M: True where the slope clears the noise floor |beta| > z * SE."""
    return np.isfinite(beta) & np.isfinite(se) & (np.abs(beta) > z * se)


def windows_inside(interval: tuple[int, int], w: int = TRAILING_W) -> tuple[float, ...]:
    """End years of the trailing windows that lie fully inside ``interval``."""
    start, end = interval
    return tuple(float(t) for t in range(start + w - 1, end + 1))


def interval_stat(values: np.ndarray, ends: np.ndarray, selected: tuple[float, ...],
                  stat: str) -> np.ndarray:
    """Per-pixel mean ("mean", for beta) or root mean square ("rms", for SE)
    over the selected windows."""
    sel = np.isin(ends, np.asarray(selected, dtype=ends.dtype))
    if not sel.any():
        return np.full(values.shape[1:], np.nan, dtype=np.float32)
    if stat == "mean":
        return np.nanmean(values[sel], axis=0).astype(np.float32)
    if stat == "rms":
        return np.sqrt(np.nanmean(values[sel] ** 2, axis=0)).astype(np.float32)
    raise ValueError(stat)
