# Tracking Gold Mining Activity Behind Tropical Deforestation with Satellite Time Series (NeurIPS 2026 Workshop)
[![Paper](https://img.shields.io/badge/Paper-OpenReview-8c1b13.svg)](https://openreview.net/forum?id=dBRczK795f)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-ffcc4d.svg)](https://huggingface.co/datasets/oryota/pan-asgm)

This repository is the official code for the paper:
> [**Tracking Gold Mining Activity Behind Tropical Deforestation with Satellite Time Series**](https://openreview.net/forum?id=dBRczK795f)  
> Ryota Okumura, Kazuyo Hirose, Binze Wang, Kazuyo Matsubae, Yoko Yamakata  
> NeurIPS 2026 Workshop on Tackling Climate Change with Machine Learning

## Introduction
The Activity Index turns yearly Sentinel-2 imagery into a per-pixel record of
artisanal and small-scale gold mining activity, with a measure of confidence.
We evaluate it on Pan-ASGM, a multi-temporal segmentation dataset of nine
mining sites in five countries.

## Installation

```bash
uv sync
```

## Data

```bash
uv run hf download oryota/pan-asgm --repo-type dataset --local-dir data/pan_asgm
```

See the [dataset card](https://huggingface.co/datasets/oryota/pan-asgm) for the file format.

## Usage

Configs for the four segmentation models are in `configs/`
(`unet`, `segformer_b2`, `mask2former`, `prithvi`). The commands below use SegFormer-B2.

**1. Compute superpixels**

```bash
uv run python scripts/precompute_superpixels.py --data-root data/pan_asgm
```

**2. Train a segmentation model** (one fold and seed; the held-out site is not used)

```bash
uv run python scripts/train.py --config configs/segformer_b2.yaml \
    --data-root data/pan_asgm --fold 0 --seed 1 --out runs/segformer_b2/fold0_seed1
```

**3. Evaluate the segmentation model** (appendix table)

```bash
uv run python scripts/eval_segmentation.py --config configs/segformer_b2.yaml \
    --data-root data/pan_asgm --checkpoint "runs/segformer_b2/fold{fold}_seed{seed}/final_model.pth" \
    --out results/segmentation/segformer_b2
```

**4. Compute the Activity Index and compare it with the baselines** (Table 1)

```bash
uv run python scripts/infer_timeseries.py --config configs/segformer_b2.yaml \
    --data-root data/pan_asgm --checkpoint "runs/segformer_b2/fold{fold}_seed{seed}/final_model.pth" \
    --out results/timeseries
uv run python scripts/eval_activity.py --data-root data/pan_asgm \
    --timeseries results/timeseries --out results/activity
```

`bash scripts/run_loso.sh segformer_b2 data/pan_asgm` runs steps 2-4 for all 9 folds and 3 seeds.

## Citation

```bibtex
@inproceedings{okumura2026tracking,
  title     = {Tracking Gold Mining Activity Behind Tropical Deforestation with Satellite Time Series},
  author    = {Okumura, Ryota and Hirose, Kazuyo and Wang, Binze and Matsubae, Kazuyo and Yamakata, Yoko},
  booktitle = {NeurIPS 2026 Workshop on Tackling Climate Change with Machine Learning},
  year      = {2026}
}
```

## License

The code and the Pan-ASGM labels, splits, and metadata are released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
The Sentinel-2 imagery contains modified Copernicus Sentinel data (2016-2025)
and is subject to the Copernicus data terms.
