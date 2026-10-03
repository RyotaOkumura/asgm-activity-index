"""Table 1: direction of change for the Activity Index and the temporal baselines.

Reads the outputs of ``infer_timeseries.py`` and the labels of the held-out
site of each fold. Reports, per method, the mean over runs (folds x seeds)
of ``dir`` and ``sign_cum``, and ``worst`` = the lowest per-site mean of
``dir``. For the Activity Index it also reports ``dir`` behind the
reliability filter.

Usage:
  uv run python scripts/eval_activity.py --data-root data/pan_asgm \
      --timeseries results/timeseries --out results/activity
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from panasgm.activity.evaluate import METHODS, TileSeries, evaluate_run
from panasgm.data import ANNOTATED_YEARS, TileKey, load_folds, load_mask


def load_run(data_root: Path, ts_dir: Path, site: str):
    tiles = []
    for path in sorted(ts_dir.glob("*.npz")):
        with np.load(path) as d:
            series = TileSeries(years=d["years"].astype(np.float32),
                                p_active=d["p_active"].astype(np.float32),
                                cloud_mask=d["cloud_mask"].astype(bool),
                                argmax=d["argmax"].astype(np.uint8))
        masks = {y: load_mask(data_root, TileKey(site, y, path.stem)) for y in ANNOTATED_YEARS}
        tiles.append((series, masks))
    return tiles


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--timeseries", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    folds_map = load_folds(args.data_root)
    folds = args.folds if args.folds is not None else sorted(folds_map)
    rows = []
    for fold in folds:
        site = folds_map[fold]
        for seed in args.seeds:
            tiles = load_run(args.data_root, args.timeseries / f"fold{fold}" / f"seed{seed}" / site, site)
            for method, res in evaluate_run(tiles).items():
                rows.append({"fold": fold, "seed": seed, "site": site, "method": method, **res})
            print(f"fold {fold} seed {seed}: AI dir = {rows[-1]['dir']:.3f}")
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "per_run.csv", index=False)

    table = df.groupby("method")[["dir", "sign_cum", "dir_reliable"]].mean()
    table["worst"] = df.groupby(["method", "site"])["dir"].mean().groupby("method").min()
    table = (100 * table[["dir", "worst", "sign_cum", "dir_reliable"]]).reindex(list(METHODS))
    table.to_csv(args.out / "table1.csv")
    print(table.round(1).to_string())


if __name__ == "__main__":
    main()
