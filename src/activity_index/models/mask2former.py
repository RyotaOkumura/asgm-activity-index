"""Mask2Former (Hugging Face, Swin-B) as a per-pixel classifier."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation

from activity_index.models.segformer import widen_conv


class Mask2Former(nn.Module):
    """Returns log class probabilities, so softmax and cross-entropy apply as usual.

    Per-pixel class probability = sum over queries of
    softmax(class logits) * sigmoid(mask logits), with the no-object class dropped.
    """

    def __init__(self, pretrained: str, num_classes: int, in_channels: int):
        super().__init__()
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained, num_labels=num_classes, ignore_mismatched_sizes=True,
        )
        if in_channels != 3:
            embeddings = self.model.model.pixel_level_module.encoder.embeddings
            patch = embeddings.patch_embeddings
            patch.projection = widen_conv(patch.projection, in_channels)
            patch.num_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        class_probs = F.softmax(out.class_queries_logits, dim=-1)[..., :-1]  # (B, Q, C)
        mask_probs = out.masks_queries_logits.sigmoid()  # (B, Q, H', W')
        prob = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
        prob = F.interpolate(prob, size=x.shape[2:], mode="bilinear", align_corners=False)
        return torch.log(prob.clamp(min=1e-8))
