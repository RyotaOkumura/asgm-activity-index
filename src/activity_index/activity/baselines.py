"""Temporal baselines of Table 1, reduced to the core changepoint decision.

* Hard-label stacking: difference of the per-year hard labels at the
  interval end points (computed in ``evaluate.py``).
* Hard labels + our estimator: the Activity Index estimator fed with hard
  labels (computed in ``evaluate.py``).
* BFAST-equiv.: a Chow structural-break test; direction from the post-break slope.
* LandTrendr-equiv.: a single-vertex piecewise-linear fit (ruptures Binseg);
  direction from the dominant segment.

Both breakpoint baselines run on the same smoothed series q as the Activity
Index, with cloudy years given weight 0.
"""

from __future__ import annotations

import numpy as np
import ruptures as rpt

from activity_index.activity.estimator import weighted_ols

# BFAST-equiv.
BREAK_YEARS = (2017, 2018, 2019, 2020, 2021, 2022, 2023)  # >= 3 points per segment
CHOW_F_THRESHOLD = 5.14  # F(2, 6) at alpha = 0.05
MIN_SEGMENT = 3
# LandTrendr-equiv.
SLOPE_THRESHOLD = 0.005
LANDTRENDR_MAX_PIXELS = 5000  # per-tile pixel sample (fixed RNG seed 0)

INTERVAL_1_YEARS = {2017, 2018, 2019, 2020}
INTERVAL_2_YEARS = {2021, 2022, 2023, 2024}


def chow_breakpoint(years: np.ndarray, ts: np.ndarray, weights: np.ndarray,
                    f_threshold: float = CHOW_F_THRESHOLD) -> dict[str, np.ndarray]:
    """Per-pixel Chow test over candidate break years. ts, weights: (N, T).

    Returns the significant break year (0 = none), the full-period slope, and
    the post-break slope (the full slope where no break is significant).
    """
    N = ts.shape[0]
    slope_full, rss_full, n_obs = weighted_ols(years, ts, weights)
    max_f = np.full(N, np.nan, dtype=np.float32)
    best = np.zeros(N, dtype=np.int16)
    for by in BREAK_YEARS:
        pre = (years <= by).astype(np.float32)[None, :]
        w_pre, w_post = weights * pre, weights * (1.0 - pre)
        ok = ((w_pre > 0).sum(axis=1) >= MIN_SEGMENT) & ((w_post > 0).sum(axis=1) >= MIN_SEGMENT)
        if not ok.any():
            continue
        _, rss_pre, _ = weighted_ols(years, ts, w_pre)
        _, rss_post, _ = weighted_ols(years, ts, w_post)
        rss_split = rss_pre + rss_post
        denom = rss_split / np.maximum(n_obs - 4, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            F = np.where((denom > 1e-9) & ok, (rss_full - rss_split) / 2.0 / denom, np.nan)
        update = np.isfinite(F) & (np.isnan(max_f) | (F > max_f))
        max_f = np.where(update, F, max_f)
        best = np.where(update, by, best)

    significant = np.isfinite(max_f) & (max_f > f_threshold)
    break_year = np.where(significant, best, 0).astype(np.int16)
    slope_post = slope_full.copy()
    for by in BREAK_YEARS:
        sel = significant & (break_year == by)
        if sel.any():
            w_after = weights[sel] * (years > by).astype(np.float32)[None, :]
            slope_post[sel], _, _ = weighted_ols(years, ts[sel], w_after)
    return {"break_year": break_year, "slope_full": slope_full.astype(np.float32),
            "slope_post": slope_post.astype(np.float32)}


def piecewise_linear(years: np.ndarray, ts: np.ndarray, weights: np.ndarray,
                     slope_threshold: float = SLOPE_THRESHOLD) -> dict[str, np.ndarray]:
    """Per-pixel single-vertex piecewise-linear fit. ts, weights: (N, T).

    Cloudy years are filled by linear interpolation before the fit. Returns
    the vertex year (0 = none), the full-period slope, and the slope of the
    segment with the larger |slope|.
    """
    N, T = ts.shape
    slope_full, _, _ = weighted_ols(years, ts, weights)
    vertex = np.zeros(N, dtype=np.int16)
    slope_dom = slope_full.copy()
    for i in range(N):
        w = weights[i]
        if w.sum() < 6:
            continue
        ts_i = ts[i].copy()
        missing = w == 0
        if missing.any():
            obs = np.where(~missing)[0]
            if len(obs) < 2:
                continue
            ts_i[missing] = np.interp(np.where(missing)[0], obs, ts_i[obs])
        try:
            v = rpt.Binseg(model="linear", jump=1, min_size=3).fit(ts_i.reshape(-1, 1)).predict(n_bkps=1)[0]
        except Exception:
            continue
        if v < 3 or v > T - 3:
            continue
        seg_slopes = []
        for sl in (slice(0, v), slice(v, T)):
            w_seg = w[sl] if w[sl].sum() > 0 else np.ones(sl.stop - sl.start)
            s, _, _ = weighted_ols(years[sl], ts_i[sl].reshape(1, -1), w_seg.reshape(1, -1))
            seg_slopes.append(float(s[0]) if np.isfinite(s[0]) else 0.0)
        pre, post = seg_slopes
        if max(abs(pre), abs(post)) < slope_threshold:
            continue
        vertex[i] = int(years[v - 1])
        slope_dom[i] = pre if abs(pre) >= abs(post) else post
    return {"vertex_year": vertex, "slope_full": slope_full.astype(np.float32),
            "slope_dom": slope_dom.astype(np.float32)}


def interval_slopes(event_year: np.ndarray, slope_event: np.ndarray,
                    slope_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-interval slopes of a breakpoint baseline.

    The interval that contains the detected break (or vertex) takes the
    event slope. The other interval, and pixels without a break, take the
    full-period slope.
    """
    in1 = np.isin(event_year, list(INTERVAL_1_YEARS))
    in2 = np.isin(event_year, list(INTERVAL_2_YEARS))
    beta1 = np.where(in1, slope_event, slope_full).astype(np.float32)
    beta2 = np.where(in2, slope_event, slope_full).astype(np.float32)
    return beta1, beta2
