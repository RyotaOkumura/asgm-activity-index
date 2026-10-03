"""Train one model on one LOSO fold.

The held-out site of the fold is excluded from training and is not read.

Usage:
  uv run python scripts/train.py --config configs/segformer_b2.yaml \
      --data-root data/pan_asgm --fold 0 --seed 1 --out runs/segformer_b2/fold0_seed1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from panasgm.data import load_folds, loso_split
from panasgm.train import train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = yaml.safe_load(args.config.read_text())
    train_keys, _ = loso_split(args.data_root, args.fold)
    test_site = load_folds(args.data_root)[args.fold]
    assert all(k.site != test_site for k in train_keys)
    logging.info("fold %d: test site %s, %d training tiles", args.fold, test_site, len(train_keys))
    path = train(cfg, args.data_root, train_keys, args.out, seed=args.seed, device=args.device)
    logging.info("saved %s", path)


if __name__ == "__main__":
    main()
