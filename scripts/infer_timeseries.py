"""Yearly front-end outputs (2016-2025) for the Activity Index.

For each tile and year, the segmentation model's class probabilities are
averaged over superpixels. Saved per tile:

  years       (T,)        float32
  p_active    (T, H, W)   float16  active_pond probability
  cloud_mask  (T, H, W)   bool     superpixel class is cloud or cloud_shadow
  argmax      (T, H, W)   uint8    superpixel class

Usage (held-out site of each fold, as in the paper):
  uv run python scripts/infer_timeseries.py --config configs/segformer_b2.yaml \
      --data-root data/pan_asgm --checkpoint "runs/segformer_b2/fold{fold}_seed{seed}/final_model.pth" \
      --out results/timeseries
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from activity_index.data import (
    ACTIVE_POND_IDX, CLOUD_IDX, CLOUD_SHADOW_IDX, TileKey, list_tiles, list_years, load_folds,
    preprocess, read_bands,
)
from activity_index.models import build_model, load_checkpoint
from activity_index.superpixels import average_over_superpixels, load_superpixels


@torch.no_grad()
def infer_tile(model, cfg: dict, data_root: Path, sp_root: Path, site: str, tile: str,
               device: str) -> dict[str, np.ndarray]:
    d = cfg["data"]
    years = list_years(data_root, site, tile)
    p_active, cloud, argmax = [], [], []
    for year in years:
        key = TileKey(site, year, tile)
        image = preprocess(read_bands(data_root, key, d["bands"]), d["bands"],
                           d["normalize_mode"], d["image_size"])
        probs = torch.softmax(model(torch.from_numpy(image)[None].to(device)), dim=1)[0].cpu().numpy()
        probs = average_over_superpixels(probs, load_superpixels(sp_root, site, key.stem))
        cls = probs.argmax(axis=0).astype(np.uint8)
        p_active.append(probs[ACTIVE_POND_IDX].astype(np.float16))
        argmax.append(cls)
        cloud.append((cls == CLOUD_IDX) | (cls == CLOUD_SHADOW_IDX))
    return {"years": np.array(years, dtype=np.float32), "p_active": np.stack(p_active),
            "cloud_mask": np.stack(cloud), "argmax": np.stack(argmax)}


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
    folds_map = load_folds(args.data_root)
    folds = args.folds if args.folds is not None else sorted(folds_map)
    for fold in folds:
        site = folds_map[fold]
        for seed in args.seeds:
            model = build_model(cfg["model"], len(cfg["data"]["bands"]))
            model = load_checkpoint(model, args.checkpoint.format(fold=fold, seed=seed), args.device)
            model = model.to(args.device).eval()
            out_dir = args.out / f"fold{fold}" / f"seed{seed}" / site
            out_dir.mkdir(parents=True, exist_ok=True)
            for tile in tqdm(list_tiles(args.data_root, site), desc=f"fold{fold} seed{seed}"):
                np.savez_compressed(out_dir / f"{tile}.npz",
                                    **infer_tile(model, cfg, args.data_root, sp_root, site, tile, args.device))


if __name__ == "__main__":
    main()
