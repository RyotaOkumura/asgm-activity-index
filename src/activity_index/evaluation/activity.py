"""Direction-of-change evaluation of the Activity Index and the baselines (Table 1).

Ground truth comes from the labeled years 2016, 2020, and 2024. For each
pixel, the active-pond state change in [2016, 2020] and [2020, 2024] gives a
sign in {-1, 0, +1}.

Metrics, pooled over all pixels of the held-out site's tiles:

* ``dir``: share of (pixel, interval) pairs with a GT change where the sign
  of the method's interval slope matches the GT sign. Both intervals are pooled.
* ``sign_cum``: the same test on the cumulative 2016 -> 2024 change.
* ``dir_reliable`` (Activity Index only): ``dir`` restricted to pairs that
  pass the reliability filter |beta| > 2 SE.

Pixels are excluded when they lie within 27 px (3 sigma) of the tile border,
are unlabeled in any of the three years, have a non-finite interval slope,
or have mean kernel support n_eff < 0.05.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from activity_index.activity import baselines
from activity_index.activity.estimator import (
    SIGMA_A, TRAILING_W, TRUNCATE, Z, activity_index, interval_stat, smooth, windows_inside,
)
from activity_index.data import ACTIVE_POND_IDX, ANNOTATED_YEARS, IGNORE_INDEX

INTERVALS = ((2016, 2020), (2020, 2024))
N_EFF_MIN = 0.05
METHODS = ("hard_label_stacking", "hard_labels_our_estimator", "bfast_equiv",
           "landtrendr_equiv", "activity_index")


@dataclass
class TileSeries:
    """Yearly outputs of the segmentation front-end for one tile."""

    years: np.ndarray       # (T,) float32
    p_active: np.ndarray    # (T, H, W) float32, superpixel-averaged
    cloud_mask: np.ndarray  # (T, H, W) bool
    argmax: np.ndarray      # (T, H, W) uint8, superpixel-averaged class



def gt_changes(masks: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    """GT signs from the three labeled years (masks remapped, 255 = unlabeled)."""
    state = [masks[y] == ACTIVE_POND_IDX for y in ANNOTATED_YEARS]
    s0, s1, s2 = (s.astype(np.int8) for s in state)
    unlabeled = np.zeros(state[0].shape, dtype=bool)
    for y in ANNOTATED_YEARS:
        unlabeled |= masks[y] == IGNORE_INDEX
    return {"sign1": s1 - s0, "sign2": s2 - s1, "cum": s2 - s0, "unlabeled": unlabeled}


def border_mask(shape: tuple[int, int], sigma: float = SIGMA_A, truncate: float = TRUNCATE) -> np.ndarray:
    """True within ceil(truncate * sigma) px of the tile border."""
    m = int(math.ceil(truncate * sigma))
    mask = np.zeros(shape, dtype=bool)
    mask[:m, :] = mask[-m:, :] = True
    mask[:, :m] = mask[:, -m:] = True
    return mask


def excluded(eligible: np.ndarray, beta1: np.ndarray, beta2: np.ndarray, n_eff: np.ndarray) -> np.ndarray:
    nonfinite = ~np.isfinite(beta1) | ~np.isfinite(beta2)
    return (~eligible) | nonfinite | (n_eff < N_EFF_MIN) | ~np.isfinite(n_eff)



def _agreement(beta: np.ndarray, sign: np.ndarray, keep: np.ndarray) -> np.ndarray:
    sel = keep & (sign != 0) & np.isfinite(beta)
    return np.sign(beta[sel]).astype(np.int8) == sign[sel].astype(np.int8)


def direction_agreement(pool: dict[str, np.ndarray]) -> tuple[float, int]:
    keep = ~pool["excluded"]
    agree = np.concatenate([_agreement(pool["beta1"], pool["sign1"], keep),
                            _agreement(pool["beta2"], pool["sign2"], keep)])
    return (float(agree.mean()) if agree.size else float("nan")), int(agree.size)


def cumulative_sign_agreement(pool: dict[str, np.ndarray]) -> float:
    keep = ~pool["excluded"] & np.isfinite(pool["beta_cum"])
    agree = _agreement(pool["beta_cum"], np.sign(pool["cum"]), keep)
    return float(agree.mean()) if agree.size else float("nan")


def reliable_direction_agreement(pool: dict[str, np.ndarray], z: float = Z) -> tuple[float, float]:
    """``dir`` on pairs passing |beta| > z SE, and the share of changed pairs kept."""
    keep = ~pool["excluded"]
    agree, n_changed = [], 0
    for k in (1, 2):
        beta, se, sign = pool[f"beta{k}"], pool[f"se{k}"], pool[f"sign{k}"]
        changed = keep & (sign != 0) & np.isfinite(beta)
        gate = np.isfinite(beta) & np.isfinite(se) & (np.abs(beta) > z * se)
        n_changed += int(changed.sum())
        sel = changed & gate
        agree.append(np.sign(beta[sel]).astype(np.int8) == sign[sel].astype(np.int8))
    agree = np.concatenate(agree)
    rate = float(agree.mean()) if agree.size else float("nan")
    return rate, (agree.size / n_changed if n_changed else float("nan"))



def evaluate_run(tiles: list[tuple[TileSeries, dict[int, np.ndarray]]],
                 w: int = TRAILING_W) -> dict[str, dict[str, float]]:
    """Scores of all methods on the held-out tiles of one run.

    ``tiles`` must be sorted by tile id: the LandTrendr-equiv. pixel sample
    draws from one RNG across tiles in that order.
    """
    pools: dict[str, dict[str, list[np.ndarray]]] = {m: {} for m in METHODS}

    def add(method: str, **arrays: np.ndarray) -> None:
        for k, v in arrays.items():
            pools[method].setdefault(k, []).append(np.ravel(v))

    windows = [windows_inside(iv, w) for iv in INTERVALS]
    rng = np.random.default_rng(0)
    for series, masks in tiles:
        years = series.years
        H, W = series.p_active.shape[1:]
        gt = gt_changes(masks)
        eligible = ~border_mask((H, W)) & ~gt["unlabeled"]
        gt_arrays = {"sign1": gt["sign1"], "sign2": gt["sign2"], "cum": gt["cum"]}

        inputs = {
            "activity_index": series.p_active,
            "hard_labels_our_estimator": (series.argmax == ACTIVE_POND_IDX).astype(np.float32),
        }
        n_eff = None
        for method, p in inputs.items():
            beta_ts, se_ts, ends, n_eff = activity_index(years, p, series.cloud_mask, w=w)
            beta1 = interval_stat(beta_ts, ends, windows[0], "mean")
            beta2 = interval_stat(beta_ts, ends, windows[1], "mean")
            add(method, beta1=beta1, beta2=beta2,
                se1=interval_stat(se_ts, ends, windows[0], "rms"),
                se2=interval_stat(se_ts, ends, windows[1], "rms"),
                beta_cum=np.nansum(beta_ts, axis=0),
                excluded=excluded(eligible, beta1, beta2, n_eff), **gt_arrays)

        year_idx = {int(y): i for i, y in enumerate(years.astype(int))}
        hard = {y: series.argmax[year_idx[y]] == ACTIVE_POND_IDX for y in ANNOTATED_YEARS}
        d = lambda a, b: (hard[b].astype(np.int8) - hard[a].astype(np.int8)).astype(np.float32)  # noqa: E731
        beta1, beta2 = d(2016, 2020), d(2020, 2024)
        add("hard_label_stacking", beta1=beta1, beta2=beta2, beta_cum=d(2016, 2024),
            excluded=excluded(eligible, beta1, beta2, n_eff), **gt_arrays)

        q, n_eff_q = smooth(series.p_active, series.cloud_mask)
        ts = q.transpose(1, 2, 0).reshape(-1, len(years))
        weights = (~series.cloud_mask).transpose(1, 2, 0).reshape(-1, len(years)).astype(np.float32)

        out = baselines.chow_breakpoint(years, ts, weights)
        beta1, beta2 = baselines.interval_slopes(out["break_year"], out["slope_post"], out["slope_full"])
        add("bfast_equiv", beta1=beta1, beta2=beta2, beta_cum=out["slope_full"],
            excluded=excluded(eligible.ravel(), out["slope_full"], out["slope_full"], n_eff_q.ravel()),
            **gt_arrays)

        idx = np.where(eligible.ravel())[0]
        if len(idx) > baselines.LANDTRENDR_MAX_PIXELS:
            idx = rng.choice(idx, baselines.LANDTRENDR_MAX_PIXELS, replace=False)
        out = baselines.piecewise_linear(years, ts[idx], weights[idx])
        event = np.zeros(H * W, dtype=np.int16)
        slope_full = np.full(H * W, np.nan, dtype=np.float32)
        slope_dom = np.full(H * W, np.nan, dtype=np.float32)
        event[idx], slope_full[idx], slope_dom[idx] = out["vertex_year"], out["slope_full"], out["slope_dom"]
        sampled = np.zeros(H * W, dtype=bool)
        sampled[idx] = True
        beta1, beta2 = baselines.interval_slopes(event, slope_dom, slope_full)
        add("landtrendr_equiv", beta1=beta1, beta2=beta2, beta_cum=slope_full,
            excluded=excluded(sampled & eligible.ravel(), slope_full, slope_full, n_eff_q.ravel()),
            **gt_arrays)

    results = {}
    for method in METHODS:
        pool = {k: np.concatenate(v) for k, v in pools[method].items()}
        dir_rate, n_changed = direction_agreement(pool)
        res = {"dir": dir_rate, "sign_cum": cumulative_sign_agreement(pool), "n_changed": n_changed}
        if method == "activity_index":
            res["dir_reliable"], res["reliable_kept"] = reliable_direction_agreement(pool)
        results[method] = res
    return results
