"""Segmentation backbones compared in the paper.

All models map (B, C, H, W) inputs to (B, 7, H, W) logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from activity_index.schema import NUM_CLASSES


def build_model(model_cfg: dict, in_channels: int) -> nn.Module:
    """Build a model with its pretrained initialization.

    ``model_cfg`` is the ``model`` section of a config in ``configs/models/``.
    """
    name = model_cfg["name"]
    if name == "unet":
        from activity_index.models.unet import create_unet
        return create_unet(model_cfg.get("encoder", "resnet50"), in_channels, NUM_CLASSES)
    if name == "segformer":
        from activity_index.models.segformer import SegFormer
        return SegFormer(model_cfg["pretrained"], NUM_CLASSES, in_channels)
    if name == "mask2former":
        from activity_index.models.mask2former import Mask2Former
        return Mask2Former(model_cfg["pretrained"], NUM_CLASSES, in_channels)
    if name == "prithvi":
        from activity_index.models.prithvi import PrithviSegmentation
        return PrithviSegmentation(NUM_CLASSES, model_cfg.get("backbone", "prithvi_eo_v2_300"))
    raise ValueError(f"unknown model: {name}")


def load_checkpoint(model: nn.Module, path: str, device: str = "cpu") -> nn.Module:
    """Load weights saved by ``activity_index.training.trainer`` (or a ``state_dict`` file)."""
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model
