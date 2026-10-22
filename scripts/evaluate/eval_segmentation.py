"""Segmentation benchmark (appendix table): per-tile metrics after superpixel averaging.

For each fold and seed, the model predicts the 27 held-out tiles. Class
probabilities are averaged over superpixels, then IoU,
Brier, and ECE are computed per tile. The table is the mean over all tiles,
folds, and seeds (per-class IoU ignores tiles where the class is absent).

Usage:
  uv run python scripts/eval_segmentation.py --config configs/segformer_b2.yaml \
      --data-root data/pan_asgm --checkpoint "runs/segformer_b2/fold{fold}_seed{seed}/final_model.pth" \
      --out results/segmentation/segformer_b2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from activity_index.data import CLASSES, SegmentationDataset, load_folds, loso_split
from activity_index.metrics import tile_metrics
from activity_index.models import build_model, load_checkpoint
from activity_index.superpixels import average_over_superpixels, load_superpixels


@torch.no_grad()
def evaluate_checkpoint(cfg: dict, data_root: Path, sp_root: Path, fold: int,
                        checkpoint: str, device: str) -> list[dict]:
    _, test_keys = loso_split(data_root, fold)
    d = cfg["data"]
    dataset = SegmentationDataset(data_root, test_keys, d["bands"], d["image_size"], d["normalize_mode"])
    model = load_checkpoint(build_model(cfg["model"], len(d["bands"])), checkpoint, device)
    model = model.to(device).eval()
    rows = []
    for key, item in zip(test_keys, dataset):
        logits = model(item["image"][None].to(device))
        mask = item["mask"]
        segments = load_superpixels(sp_root, key.site, key.stem)
        if logits.shape[-2:] != segments.shape:
            logits = F.interpolate(logits, size=segments.shape, mode="bilinear", align_corners=False)
            mask = F.interpolate(mask[None, None].float(), size=segments.shape, mode="nearest")[0, 0].long()
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        probs = average_over_superpixels(probs, segments)
        rows.append({"site": key.site, "year": key.year, "tile": key.tile,
                     **tile_metrics(probs, mask.numpy().astype(np.int64))})
    return rows


def summarize(df: pd.DataFrame) -> pd.Series:
    cols = [f"iou_{c}" for c in range(len(CLASSES))] + ["miou", "brier", "ece"]
    out = df[cols].mean()
    out.index = [f"iou_{CLASSES[int(c[4:])]}" if c.startswith("iou_") else c for c in out.index]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--checkpoint", required=True, help="path template with {fold} and {seed}")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sp-root", type=Path, default=None, help="default: <data-root>/superpixels")
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    sp_root = args.sp_root or args.data_root / "superpixels"
    folds = args.folds if args.folds is not None else sorted(load_folds(args.data_root))
    rows = []
    for fold in tqdm(folds, desc="folds"):
        for seed in args.seeds:
            ckpt = args.checkpoint.format(fold=fold, seed=seed)
            for r in evaluate_checkpoint(cfg, args.data_root, sp_root, fold, ckpt, args.device):
                rows.append({"fold": fold, "seed": seed, **r})
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "per_tile.csv", index=False)
    table = summarize(df)
    table.to_csv(args.out / "summary.csv", header=["mean"])
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
