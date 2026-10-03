#!/usr/bin/env bash
# Full LOSO run for one model: 9 folds x 3 seeds, then evaluation.
# Usage: bash scripts/run_loso.sh segformer_b2 data/pan_asgm
set -euo pipefail

MODEL=${1:?model config name, e.g. segformer_b2}
DATA=${2:?data root}
CKPT="runs/${MODEL}/fold{fold}_seed{seed}/final_model.pth"

for FOLD in 0 1 2 3 4 5 6 7 8; do
  for SEED in 1 2 3; do
    uv run python scripts/train.py --config "configs/${MODEL}.yaml" --data-root "$DATA" \
      --fold "$FOLD" --seed "$SEED" --out "runs/${MODEL}/fold${FOLD}_seed${SEED}"
  done
done

uv run python scripts/eval_segmentation.py --config "configs/${MODEL}.yaml" --data-root "$DATA" \
  --checkpoint "$CKPT" --out "results/segmentation/${MODEL}"

# The Activity Index uses the SegFormer-B2 front-end.
if [ "$MODEL" = "segformer_b2" ]; then
  uv run python scripts/infer_timeseries.py --config "configs/${MODEL}.yaml" --data-root "$DATA" \
    --checkpoint "$CKPT" --out results/timeseries
  uv run python scripts/eval_activity.py --data-root "$DATA" \
    --timeseries results/timeseries --out results/activity
fi
