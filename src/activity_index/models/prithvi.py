"""Prithvi-EO-2.0 (frozen) with a UPerNet decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from terratorch.registry import BACKBONE_REGISTRY


class PPM(nn.Module):
    """Pyramid pooling module."""

    def __init__(self, in_channels: int, pool_scales: tuple = (1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True),
            )
            for scale in pool_scales
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        h, w = x.shape[2:]
        return [x] + [
            F.interpolate(stage(x), size=(h, w), mode="bilinear", align_corners=False)
            for stage in self.stages
        ]


def conv_bn_relu(in_ch: int, out_ch: int, k: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UPerNetHead(nn.Module):
    """UPerNet decoder: PPM on the last feature map plus an FPN."""

    def __init__(self, in_channels: list[int], channels: int = 256, num_classes: int = 7,
                 pool_scales: tuple = (1, 2, 3, 6)):
        super().__init__()
        self.ppm = PPM(in_channels[-1], pool_scales)
        ppm_out = in_channels[-1] + (in_channels[-1] // 4) * len(pool_scales)
        self.bottleneck = conv_bn_relu(ppm_out, channels, 3)
        self.lateral_convs = nn.ModuleList([conv_bn_relu(ch, channels, 1) for ch in in_channels[:-1]])
        self.fpn_convs = nn.ModuleList([conv_bn_relu(channels, channels, 3) for _ in in_channels[:-1]])
        self.fpn_bottleneck = conv_bn_relu(channels * len(in_channels), channels, 3)
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        fpn = [self.bottleneck(torch.cat(self.ppm(features[-1]), dim=1))]
        for i in range(len(features) - 2, -1, -1):
            lateral = self.lateral_convs[i](features[i])
            top = F.interpolate(fpn[0], size=lateral.shape[2:], mode="bilinear", align_corners=False)
            fpn.insert(0, self.fpn_convs[i](lateral + top))
        size = fpn[0].shape[2:]
        fpn = [f if f.shape[2:] == size else
               F.interpolate(f, size=size, mode="bilinear", align_corners=False) for f in fpn]
        return self.classifier(self.fpn_bottleneck(torch.cat(fpn, dim=1)))


class PrithviSegmentation(nn.Module):
    """Frozen Prithvi-EO-2.0-300M; only the layer norms and the decoder are trained.

    Input: (B, 6, 224, 224) with bands B02, B03, B04, B8A, B11, B12.
    """

    def __init__(self, num_classes: int = 7, backbone: str = "prithvi_eo_v2_300",
                 feature_indices: tuple[int, ...] = (5, 11, 17, 23), decoder_channels: int = 256):
        super().__init__()
        self.feature_indices = feature_indices
        self.backbone = BACKBONE_REGISTRY.build(backbone, pretrained=True)
        for p in self.backbone.parameters():
            p.requires_grad = False
        embed_dim = getattr(self.backbone, "embed_dim", 1024)
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in feature_indices])
        self.decoder = UPerNetHead([embed_dim] * len(feature_indices), decoder_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks = self.backbone(x.unsqueeze(2))  # one block output per layer, (B, 1 + N, D)
        features = []
        for norm, idx in zip(self.norms, self.feature_indices):
            feat = norm(blocks[min(idx, len(blocks) - 1)][:, 1:, :])  # drop the CLS token
            B, N, D = feat.shape
            side = int(N ** 0.5)
            features.append(feat.permute(0, 2, 1).reshape(B, D, side, side))
        logits = self.decoder(features)
        return F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)
