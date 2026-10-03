import numpy as np
import torch
import torch.nn as nn

import activity_index.data as data
from activity_index.data import (
    IGNORE_INDEX, SegmentationDataset, TileKey, list_tiles, list_years, load_mask, loso_split, preprocess,
)
from activity_index.train import train


def test_loso_split_never_puts_test_site_in_train(tiny_root):
    for fold, test_site in data.load_folds(tiny_root).items():
        train_keys, test_keys = loso_split(tiny_root, fold)
        assert {k.site for k in test_keys} == {test_site}
        assert test_site not in {k.site for k in train_keys}
        assert len(test_keys) == 2 * 3  # 2 tiles x 3 labeled years
        assert len(train_keys) == 2 * 2 * 3


def test_listing(tiny_root):
    assert list_tiles(tiny_root, "SiteA") == ["r0c0", "r0c1"]
    assert list_years(tiny_root, "SiteA", "r0c0") == list(range(2016, 2026))


def test_mask_remap(tiny_root):
    key = TileKey("SiteA", 2016, "r0c0")
    raw = __import__("cv2").imread(str(data.mask_path(tiny_root, key)), 0)
    mask = load_mask(tiny_root, key)
    assert np.all(mask[raw == 0] == IGNORE_INDEX)
    assert np.array_equal(mask[raw > 0], raw[raw > 0] - 1)


def test_preprocess_imagenet_and_resize():
    raw = np.full((4, 10, 10), 5000, dtype=np.uint16)
    out = preprocess(raw, ["B04", "B03", "B02", "B08"], "imagenet", 10)
    assert np.allclose(out[0], (0.5 - 0.485) / 0.229)
    assert np.allclose(out[3], (0.5 - 0.35) / 0.20)
    assert preprocess(raw, ["B04"] * 4, "sentinel", 16).shape == (4, 16, 16)


def test_training_never_reads_the_test_site(tiny_root, monkeypatch, tmp_path):
    """Run a short training and record every tile the data loader opens."""
    opened = []
    real_read = data.read_bands

    def spy(root, key, bands):
        opened.append(key)
        return real_read(root, key, bands)

    monkeypatch.setattr(data, "read_bands", spy)
    cfg = {
        "data": {"bands": ["B04", "B03", "B02", "B08"], "image_size": 32, "normalize_mode": "imagenet"},
        "training": {"epochs": 2, "batch_size": 2, "lr": 1e-3, "weight_decay": 0.0,
                     "warmup_epochs": 1, "num_workers": 0},
        "loss": {"ce_weight": 1.0, "dice_weight": 1.0},
    }
    fold = 1
    train_keys, _ = loso_split(tiny_root, fold)
    path = train(cfg, tiny_root, train_keys, tmp_path / "run", seed=1, device="cpu",
                 model=nn.Conv2d(4, 7, 1))
    test_site = data.load_folds(tiny_root)[fold]
    assert opened and all(k.site != test_site for k in opened)
    assert torch.load(path, weights_only=False)["epoch"] == 2


def test_dataset_item_shapes(tiny_root):
    keys, _ = loso_split(tiny_root, 0)
    ds = SegmentationDataset(tiny_root, keys, ["B02", "B03", "B04", "B8A", "B11", "B12"], 40, "sentinel")
    item = ds[0]
    assert item["image"].shape == (6, 40, 40)
    assert item["mask"].shape == (40, 40)
