import numpy as np
import pytest

from panasgm.activity import baselines
from panasgm.activity.estimator import (
    activity_index, gaussian_mean, interval_stat, reliable, trailing_ols, weighted_ols, windows_inside,
)
from panasgm.activity.evaluate import TileSeries, border_mask, direction_agreement, evaluate_run, gt_changes

YEARS = np.arange(2016, 2026, dtype=np.float32)


def test_weighted_ols_recovers_line_and_skips_zero_weights():
    ts = np.array([2.0 * YEARS + 1.0])
    w = np.ones_like(ts)
    slope, rss, n = weighted_ols(YEARS, ts, w)
    assert slope[0] == pytest.approx(2.0) and rss[0] == pytest.approx(0.0, abs=1e-6) and n[0] == 10
    ts_bad = ts.copy()
    ts_bad[0, 3] = 100.0
    w[0, 3] = 0.0
    assert weighted_ols(YEARS, ts_bad, w)[0][0] == pytest.approx(2.0)


def test_trailing_ols_window_ends_and_min_obs():
    q = np.tile(0.1 * (YEARS - 2016), (2, 1)).T  # (T, 2)
    w = np.ones_like(q)
    w[4:6, 1] = 0.0  # pixel 1 misses 2020 and 2021
    slopes, se, ends = trailing_ols(YEARS, q, w, w=3)
    assert list(ends) == list(range(2018, 2026))
    assert np.allclose(slopes[:, 0], 0.1, atol=1e-6)
    assert np.isnan(slopes[ends == 2021, 1]).all()  # window 2019-2021 has 1 valid year


def test_gaussian_mean_fills_cloudy_pixel_from_neighbors():
    arr = np.full((1, 21, 21), 0.5, dtype=np.float32)
    valid = np.ones_like(arr, dtype=bool)
    valid[0, 10, 10] = False
    arr[0, 10, 10] = 99.0
    mean, support = gaussian_mean(arr, valid, sigma=2.0)
    assert mean[0, 10, 10] == pytest.approx(0.5)
    assert support[0, 10, 10] < support[0, 0, 0] or support[0, 10, 10] < 1.0


def test_activity_index_sign_and_reliability():
    T, H, W = len(YEARS), 60, 60
    p = np.zeros((T, H, W), dtype=np.float32)
    p[:, :, :30] = np.linspace(0.0, 0.9, T)[:, None, None]  # left half expands
    cloud = np.zeros_like(p, dtype=bool)
    beta, se, ends, n_eff = activity_index(YEARS, p, cloud)
    assert beta.shape == (8, H, W) and n_eff.shape == (H, W)
    assert (beta[:, 30, 5] > 0).all()
    assert np.allclose(beta[:, 30, 55], 0.0, atol=1e-3)
    assert reliable(np.array([0.1, 0.1]), np.array([0.01, 0.1])).tolist() == [True, False]


def test_interval_windows_and_stats():
    assert windows_inside((2016, 2020)) == (2018.0, 2019.0, 2020.0)
    ends = np.array([2018, 2019, 2020, 2021], dtype=np.float32)
    vals = np.array([[1.0], [2.0], [3.0], [10.0]], dtype=np.float32)
    assert interval_stat(vals, ends, (2018.0, 2019.0, 2020.0), "mean")[0] == pytest.approx(2.0)
    assert interval_stat(vals, ends, (2018.0, 2019.0, 2020.0), "rms")[0] == pytest.approx(np.sqrt(14 / 3))


def test_gt_changes_and_direction():
    m = {2016: np.array([[2, 0]]), 2020: np.array([[0, 2]]), 2024: np.array([[0, 255]])}
    gt = gt_changes(m)
    assert gt["sign1"].tolist() == [[-1, 1]]
    assert gt["unlabeled"].tolist() == [[False, True]]
    pool = {"beta1": np.array([-0.1, 0.2]), "beta2": np.array([0.0, 0.0]),
            "sign1": np.array([-1, 1]), "sign2": np.array([0, 0]), "excluded": np.array([False, False])}
    assert direction_agreement(pool) == (1.0, 2)


def test_border_mask_width():
    m = border_mask((60, 60))
    assert m[26, 30] and not m[27, 30] and m[33, 30] and not m[32, 30]


def test_chow_detects_a_break():
    ts = np.where(YEARS <= 2020, 0.0, 0.3 * (YEARS - 2020)).astype(np.float32)[None, :]
    ts = ts + np.random.default_rng(0).normal(0, 0.005, ts.shape).astype(np.float32)
    out = baselines.chow_breakpoint(YEARS, ts, np.ones_like(ts))
    assert out["break_year"][0] in (2020, 2021)
    assert out["slope_post"][0] > 0.2


def test_evaluate_run_on_synthetic_tile():
    T, H, W = len(YEARS), 64, 64
    p = np.zeros((T, H, W), dtype=np.float32)
    p[YEARS >= 2019, 20:44, 20:44] = 0.9  # a pond appears after 2018
    argmax = np.where(p > 0.5, 2, 0).astype(np.uint8)
    series = TileSeries(YEARS, p, np.zeros_like(p, dtype=bool), argmax)
    masks = {y: argmax[list(YEARS).index(y)] for y in (2016, 2020, 2024)}
    res = evaluate_run([(series, masks)])
    assert res["activity_index"]["dir"] == pytest.approx(1.0)
    assert res["hard_label_stacking"]["dir"] == pytest.approx(1.0)
