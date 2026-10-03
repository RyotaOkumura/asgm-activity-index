# Pan-ASGM and the Activity Index

Code for *Tracking Gold Mining Activity Behind Tropical Deforestation with
Satellite Time Series* (NeurIPS 2026 Workshop on Tackling Climate Change with
Machine Learning).

The **Activity Index** turns yearly Sentinel-2 imagery into a per-pixel record
of artisanal and small-scale gold mining (ASGM) activity. It reads the yearly
probability of *active ponds* (turbid ore-washing ponds) from a segmentation
model and reports whether activity is expanding or declining, with a standard
error and a reliability filter.

**Pan-ASGM** is the multi-temporal dataset used to evaluate it: nine sites in
five countries, a 6 km x 6 km area per site, yearly composites for 2016-2025,
and seven-class labels for 2016, 2020, and 2024.

This repository reproduces the experiments in the paper:

| Experiment | Script |
|---|---|
| Segmentation benchmark of U-Net, SegFormer-B2, Mask2Former, and Prithvi-EO 2.0 under leave-one-site-out (LOSO) evaluation (appendix table) | `train.py`, `eval_segmentation.py` |
| Activity Index vs. hard-label stacking, hard labels + our estimator, BFAST-equiv., and LandTrendr-equiv. (Table 1), and direction agreement behind the reliability filter | `infer_timeseries.py`, `eval_activity.py` |

## Installation

```bash
uv sync
```

Python 3.10. Package versions are pinned to those that produced the paper
numbers. Pretrained weights for the backbones are downloaded from the Hugging
Face Hub on first use.

## Data

<!-- TODO: Hugging Face dataset link -->

Download Pan-ASGM and place it as follows (`data/pan_asgm` below):

```
data/pan_asgm/
├── images/{site}/{site}_{year}_{tile}.tif   13-band Sentinel-2 L1C, uint16, 200 x 200 px (10 m)
├── masks/{site}/{site}_{year}_{tile}.png    labels for 2016, 2020, 2024
├── splits/loso.yaml                         fold -> held-out site
└── metadata.csv                             country and acquisition month per image
```

`tile` is `r{row}c{col}` in the 3 x 3 grid of 2 km tiles of a site. Band order
in each image is B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12.
Mask values are 0 = unlabeled and 1-7 = `vegetation`, `bare_ground`,
`active_pond`, `inactive_pond`, `artifact`, `cloud`, `cloud_shadow`.

## Usage

**1. Superpixels.** Both evaluations average class probabilities over SLIC
superpixels. This writes `data/pan_asgm/superpixels/`.

```bash
uv run python scripts/precompute_superpixels.py --data-root data/pan_asgm
```

**2. Training.** One model per fold (9 folds, held-out site in
`splits/loso.yaml`) and seed (1, 2, 3). Training runs for a fixed 50 epochs
and keeps the final weights. The held-out site is not read during training.

```bash
uv run python scripts/train.py --config configs/segformer_b2.yaml \
    --data-root data/pan_asgm --fold 0 --seed 1 --out runs/segformer_b2/fold0_seed1
```

**3. Segmentation benchmark.** Per-tile class IoU, mIoU, Brier score, and ECE
on the held-out tiles, averaged over tiles, folds, and seeds.

```bash
uv run python scripts/eval_segmentation.py --config configs/segformer_b2.yaml \
    --data-root data/pan_asgm --checkpoint "runs/segformer_b2/fold{fold}_seed{seed}/final_model.pth" \
    --out results/segmentation/segformer_b2
```

**4. Activity Index.** Yearly SegFormer-B2 outputs for 2016-2025 on the
held-out site of each fold, then Table 1.

```bash
uv run python scripts/infer_timeseries.py --config configs/segformer_b2.yaml \
    --data-root data/pan_asgm --checkpoint "runs/segformer_b2/fold{fold}_seed{seed}/final_model.pth" \
    --out results/timeseries
uv run python scripts/eval_activity.py --data-root data/pan_asgm \
    --timeseries results/timeseries --out results/activity
```

`scripts/run_loso.sh <model> data/pan_asgm` runs steps 2-4 for all folds and seeds.

## The Activity Index in short

`src/activity_index/activity/estimator.py` implements the estimator
A = M ∘ T ∘ A_smooth on the per-pixel series of superpixel-averaged
`active_pond` probabilities:

- **A_smooth**: Gaussian smoothing (sigma = 9 px) weighted by cloud-free
  pixels, so cloudy pixels borrow evidence from clear neighbors.
- **T**: least-squares slope beta and its standard error SE over a trailing
  three-year window. The value for year t uses no later years.
- **M**: the reliability filter keeps beta where |beta| > 2 SE.

Evaluation details (eligible pixels, interval aggregation, and baseline
settings) are documented in `src/activity_index/activity/evaluate.py` and
`baselines.py`.

## Tests

```bash
uv run pytest -m "not weights"   # fast, synthetic data only
uv run pytest -m weights         # builds the four models (downloads weights)
```

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

Code: MIT. Sentinel-2 imagery in Pan-ASGM contains modified Copernicus
Sentinel data (2016-2025).
