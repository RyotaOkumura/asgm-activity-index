import numpy as np
import pytest
import torch

from activity_index.evaluation.segmentation import brier_score, class_iou, expected_calibration_error, tile_metrics
from activity_index.training.losses import CEDiceLoss, DiceLoss


def test_class_iou_nan_for_absent_class_and_ignore():
    pred = np.array([[0, 0], [1, 1]])
    target = np.array([[0, 1], [1, 255]])
    ious, miou = class_iou(pred, target, num_classes=3)
    assert ious[0] == pytest.approx(1 / 2)
    assert ious[1] == pytest.approx(1 / 2)
    assert np.isnan(ious[2])
    assert miou == pytest.approx(0.5)


def test_brier_and_ece_perfect_and_known():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    assert brier_score(probs, labels, num_classes=2) == 0.0
    assert expected_calibration_error(probs, labels) == 0.0
    # confidence 0.6 everywhere, half correct -> ECE = |0.6 - 0.5|
    probs = np.array([[0.6, 0.4]] * 4)
    labels = np.array([0, 0, 1, 1])
    assert expected_calibration_error(probs, labels) == pytest.approx(0.1)
    assert brier_score(probs, labels, num_classes=2) == pytest.approx((0.16 + 0.16 + 0.36 + 0.36) / 2)


def test_ignored_pixels_do_not_count():
    probs = np.array([[0.9, 0.1], [0.1, 0.9]])
    assert brier_score(probs, np.array([0, 255]), 2) == pytest.approx(0.02)


def test_tile_metrics_keys():
    probs = np.full((7, 4, 4), 1 / 7, dtype=np.float32)
    row = tile_metrics(probs, np.zeros((4, 4), dtype=np.int64))
    assert {"miou", "brier", "ece", "iou_0", "iou_6"} <= set(row)


def test_dice_zero_for_confident_correct_prediction():
    target = torch.tensor([[[0, 1], [1, 0]]])
    logits = torch.full((1, 2, 2, 2), -20.0)
    logits[0, 0][target[0] == 0] = 20.0
    logits[0, 1][target[0] == 1] = 20.0
    assert DiceLoss()(logits, target).item() < 1e-6
    assert CEDiceLoss()(logits, target).item() < 1e-6


def test_loss_ignores_unlabeled_pixels():
    logits = torch.randn(1, 3, 4, 4)
    t1 = torch.randint(0, 3, (1, 4, 4))
    t2 = t1.clone()
    t1[0, 0, 0], t2[0, 0, 0] = 255, 255
    t2[0, 0, 1] = t1[0, 0, 1]
    assert CEDiceLoss()(logits, t1).item() == pytest.approx(CEDiceLoss()(logits, t2).item())
