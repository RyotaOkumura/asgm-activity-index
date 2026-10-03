"""A tiny synthetic dataset in the Pan-ASGM layout."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import tifffile
import yaml

SITES = ["SiteA", "SiteB", "SiteC"]
TILES = ["r0c0", "r0c1"]
SIZE = 32


@pytest.fixture
def tiny_root(tmp_path):
    rng = np.random.default_rng(0)
    for site in SITES:
        for tile in TILES:
            for year in range(2016, 2026):
                stem = f"{site}_{year}_{tile}"
                img = tmp_path / "images" / site / f"{stem}.tif"
                img.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(img, rng.integers(0, 5000, (13, SIZE, SIZE), dtype=np.uint16))
                if year in (2016, 2020, 2024):
                    m = tmp_path / "masks" / site / f"{stem}.png"
                    m.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(m), rng.integers(0, 8, (SIZE, SIZE), dtype=np.uint8))
    (tmp_path / "splits").mkdir()
    folds = [{"fold": i, "test_site": s} for i, s in enumerate(SITES)]
    (tmp_path / "splits" / "loso.yaml").write_text(yaml.safe_dump({"sites": SITES, "folds": folds}))
    return tmp_path
