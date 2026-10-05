"""Pan-ASGM data access: file layout, band preprocessing, masks, and LOSO splits.

Dataset layout (``data_root``)::

    images/{site}/{site}_{year}_{tile}.tif   13-band Sentinel-2 L1C, uint16, 200x200
    masks/{site}/{site}_{year}_{tile}.png    labels for 2016 / 2020 / 2024
    splits/loso.yaml                         fold -> held-out (test) site

``tile`` is ``r{row}c{col}`` with row, col in {0, 1, 2}: the 3x3 grid of
2 km tiles that covers each 6 km site.

Mask PNG values are 0 = unlabeled and 1..7 = the classes below. They are
remapped on load to 0..6, with unlabeled pixels set to ``IGNORE_INDEX``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
import yaml
from torch.utils.data import Dataset

CLASSES = (
    "vegetation",
    "bare_ground",
    "active_pond",
    "inactive_pond",
    "artifact",
    "cloud",
    "cloud_shadow",
)
NUM_CLASSES = len(CLASSES)
ACTIVE_POND_IDX = CLASSES.index("active_pond")
CLOUD_IDX = CLASSES.index("cloud")
CLOUD_SHADOW_IDX = CLASSES.index("cloud_shadow")
IGNORE_INDEX = 255

ANNOTATED_YEARS = (2016, 2020, 2024)

# Band order inside each image GeoTIFF.
ALL_BANDS = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
)

# Per-band mean / std applied after scaling reflectance to [0, 1]
# (normalize_mode="imagenet"). RGB uses the ImageNet statistics.
IMAGENET_NORM = {
    "B04": (0.485, 0.229),
    "B03": (0.456, 0.224),
    "B02": (0.406, 0.225),
}
DEFAULT_NORM = (0.35, 0.20)



@dataclass(frozen=True, order=True)
class TileKey:
    """One tile of one site in one year."""

    site: str
    year: int
    tile: str  # "r{row}c{col}"

    @property
    def stem(self) -> str:
        return f"{self.site}_{self.year}_{self.tile}"


def image_path(data_root: Path, key: TileKey) -> Path:
    return Path(data_root) / "images" / key.site / f"{key.stem}.tif"


def mask_path(data_root: Path, key: TileKey) -> Path:
    return Path(data_root) / "masks" / key.site / f"{key.stem}.png"


def list_tiles(data_root: Path, site: str) -> list[str]:
    """Sorted tile ids ("r0c0", ...) of a site."""
    names = {p.stem.rsplit("_", 1)[1] for p in (Path(data_root) / "images" / site).glob("*.tif")}
    return sorted(names)


def list_years(data_root: Path, site: str, tile: str) -> list[int]:
    """Sorted years with an image for (site, tile)."""
    years = []
    for p in (Path(data_root) / "images" / site).glob(f"{site}_*_{tile}.tif"):
        years.append(int(p.stem[len(site) + 1:].split("_")[0]))
    return sorted(years)



def read_bands(data_root: Path, key: TileKey, bands: list[str]) -> np.ndarray:
    """Raw reflectance of the selected bands, (C, H, W) uint16."""
    stack = tifffile.imread(image_path(data_root, key))  # (13, H, W)
    if stack.ndim != 3 or stack.shape[0] != len(ALL_BANDS):
        raise ValueError(f"expected (13, H, W) image, got {stack.shape}: {key.stem}")
    idx = [ALL_BANDS.index(b) for b in bands]
    return stack[idx]


def preprocess(
    raw: np.ndarray, bands: list[str], normalize_mode: str, image_size: int,
) -> np.ndarray:
    """Model input from raw reflectance, (C, H, W) float32.

    Reflectance is divided by 10000 and clipped to [0, 1]. "imagenet" then
    standardizes each band, "sentinel" stops there. The image is resized
    bilinearly when ``image_size`` differs from the tile size.
    """
    image = raw.astype(np.float32) / 10000.0
    image = np.clip(image, 0.0, 1.0)
    if normalize_mode == "imagenet":
        mean = np.array([IMAGENET_NORM.get(b, DEFAULT_NORM)[0] for b in bands],
                        dtype=np.float32).reshape(-1, 1, 1)
        std = np.array([IMAGENET_NORM.get(b, DEFAULT_NORM)[1] for b in bands],
                       dtype=np.float32).reshape(-1, 1, 1)
        image = (image - mean) / std
    elif normalize_mode != "sentinel":
        raise ValueError(f"unknown normalize_mode: {normalize_mode}")
    h, w = image.shape[1:]
    if (h, w) != (image_size, image_size):
        image = np.stack([
            cv2.resize(c, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            for c in image
        ])
    return image


def load_mask(data_root: Path, key: TileKey) -> np.ndarray:
    """Label map (H, W) uint8: 0..6 = class, 255 = unlabeled."""
    png = cv2.imread(str(mask_path(data_root, key)), cv2.IMREAD_GRAYSCALE)
    if png is None:
        raise FileNotFoundError(mask_path(data_root, key))
    return np.where(png == 0, IGNORE_INDEX, png - 1).astype(np.uint8)



def load_folds(data_root: Path) -> dict[int, str]:
    """fold index -> held-out (test) site."""
    with open(Path(data_root) / "splits" / "loso.yaml") as f:
        spec = yaml.safe_load(f)
    return {int(f["fold"]): f["test_site"] for f in spec["folds"]}


def annotated_tiles(data_root: Path, sites: list[str]) -> list[TileKey]:
    """All labeled (site, year, tile) of the given sites, sorted."""
    keys = []
    for site in sites:
        for tile in list_tiles(data_root, site):
            for year in ANNOTATED_YEARS:
                key = TileKey(site, year, tile)
                if mask_path(data_root, key).exists():
                    keys.append(key)
    return sorted(keys)


def loso_split(data_root: Path, fold: int) -> tuple[list[TileKey], list[TileKey]]:
    """(train tiles, test tiles) of a fold. The test site is never in train."""
    folds = load_folds(data_root)
    test_site = folds[fold]
    train_sites = sorted(s for s in folds.values() if s != test_site)
    return annotated_tiles(data_root, train_sites), annotated_tiles(data_root, [test_site])



class SegmentationDataset(Dataset):
    """Labeled tiles for training and evaluation."""

    def __init__(
        self,
        data_root: Path,
        keys: list[TileKey],
        bands: list[str],
        image_size: int,
        normalize_mode: str,
        augment: bool = False,
    ):
        self.data_root = Path(data_root)
        self.keys = list(keys)
        self.bands = list(bands)
        self.image_size = image_size
        self.normalize_mode = normalize_mode
        self.augment = augment

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> dict:
        key = self.keys[idx]
        image = preprocess(read_bands(self.data_root, key, self.bands),
                           self.bands, self.normalize_mode, self.image_size)
        mask = load_mask(self.data_root, key)
        if mask.shape != (self.image_size, self.image_size):
            mask = cv2.resize(mask, (self.image_size, self.image_size),
                              interpolation=cv2.INTER_NEAREST)
        if self.augment:
            image, mask = random_flip_rotate(image, mask)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)).long(),
            "stem": key.stem,
        }


def random_flip_rotate(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Random horizontal / vertical flip and 90-degree rotation."""
    if np.random.random() > 0.5:
        image, mask = image[:, :, ::-1], mask[:, ::-1]
    if np.random.random() > 0.5:
        image, mask = image[:, ::-1, :], mask[::-1, :]
    k = np.random.randint(0, 4)
    if k > 0:
        image, mask = np.rot90(image, k, axes=(1, 2)), np.rot90(mask, k, axes=(0, 1))
    return image.copy(), mask.copy()
