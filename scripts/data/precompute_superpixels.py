"""Compute SLIC superpixels for every image (written to <data-root>/superpixels).

Usage:
  uv run python scripts/data/precompute_superpixels.py --data-root data/pan_asgm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from activity_index.data.pan_asgm import TileKey, list_tiles, list_years, load_folds, read_bands
from activity_index.features.superpixels import BANDS, compute_superpixels, superpixel_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="default: <data-root>/superpixels")
    args = ap.parse_args()
    out = args.out or args.data_root / "superpixels"

    keys = []
    for site in sorted(set(load_folds(args.data_root).values())):
        for tile in list_tiles(args.data_root, site):
            keys += [TileKey(site, y, tile) for y in list_years(args.data_root, site, tile)]

    for key in tqdm(keys):
        path = superpixel_path(out, key.site, key.stem)
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, segments=compute_superpixels(read_bands(args.data_root, key, BANDS)))


if __name__ == "__main__":
    main()
