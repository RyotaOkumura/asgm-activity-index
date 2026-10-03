"""U-Net with an ImageNet-pretrained ResNet encoder."""

import segmentation_models_pytorch as smp
import torch.nn as nn


def create_unet(encoder: str, in_channels: int, num_classes: int) -> nn.Module:
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )
