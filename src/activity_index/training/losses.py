"""Cross-entropy + Dice loss (1:1)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from activity_index.data import IGNORE_INDEX


class DiceLoss(nn.Module):
    """1 - soft Dice, averaged over classes. Unlabeled pixels are excluded."""

    def __init__(self, smooth: float = 1.0, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        valid = (target != self.ignore_index).unsqueeze(1)
        prob = F.softmax(logits, dim=1) * valid
        onehot = F.one_hot(target.clamp(0, num_classes - 1), num_classes).permute(0, 3, 1, 2).float() * valid
        dims = (0, 2, 3)
        intersection = (prob * onehot).sum(dim=dims)
        cardinality = prob.sum(dim=dims) + onehot.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return (1.0 - dice).mean()


class CEDiceLoss(nn.Module):
    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0,
                 ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(logits, target)
