"""SegFormer (Hugging Face) with a widened first patch embedding."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


def widen_conv(old: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Copy of an RGB conv with extra input channels set to the RGB mean filter."""
    new = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=old.bias is not None)
    with torch.no_grad():
        new.weight[:, :3] = old.weight
        rgb_mean = old.weight.mean(dim=1, keepdim=True)
        for i in range(3, in_channels):
            new.weight[:, i:i + 1] = rgb_mean
        if old.bias is not None:
            new.bias.copy_(old.bias)
    return new


class SegFormer(nn.Module):
    def __init__(self, pretrained: str, num_classes: int, in_channels: int):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained, num_labels=num_classes, ignore_mismatched_sizes=True,
        )
        if in_channels != 3:
            embed = self.model.segformer.encoder.patch_embeddings[0]
            embed.proj = widen_conv(embed.proj, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=x).logits  # (B, C, H/4, W/4)
        return F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)
