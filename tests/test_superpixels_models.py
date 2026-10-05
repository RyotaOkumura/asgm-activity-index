import numpy as np
import pytest
import torch

from activity_index.superpixels import average_over_superpixels, compute_superpixels, false_color


def test_average_over_superpixels_keeps_distribution():
    rng = np.random.default_rng(0)
    probs = rng.random((7, 20, 20)).astype(np.float32)
    probs /= probs.sum(0, keepdims=True)
    seg = np.repeat(np.arange(4), 100).reshape(20, 20)
    out = average_over_superpixels(probs, seg)
    assert np.allclose(out.sum(0), 1.0, atol=1e-5)
    assert np.allclose(out[:, 0, 0], probs[:, seg == 0].mean(axis=1), atol=1e-6)


def test_slic_outputs_contiguous_labels():
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 4000, (3, 60, 60), dtype=np.uint16)
    seg = compute_superpixels(raw, 50)
    assert seg.shape == (60, 60)
    assert np.array_equal(np.unique(seg), np.arange(seg.max() + 1))


def test_false_color_scaling():
    raw = np.array([[[0]], [[2000]], [[8000]]], dtype=np.uint16)
    assert false_color(raw)[0, 0].tolist() == [0, 127, 255]


@pytest.mark.weights
@pytest.mark.parametrize("cfg,bands,size", [
    ({"name": "unet", "encoder": "resnet50"}, 4, 200),
    ({"name": "segformer", "pretrained": "nvidia/segformer-b2-finetuned-ade-512-512"}, 4, 200),
    ({"name": "mask2former", "pretrained": "facebook/mask2former-swin-base-ade-semantic"}, 4, 200),
    ({"name": "prithvi", "backbone": "prithvi_eo_v2_300"}, 6, 224),
])
def test_models_return_logits_at_input_size(cfg, bands, size):
    from activity_index.models import build_model
    model = build_model(cfg, bands).eval()
    with torch.no_grad():
        out = model(torch.randn(1, bands, size, size))
    assert out.shape == (1, 7, size, size)
    assert torch.isfinite(out).all()
