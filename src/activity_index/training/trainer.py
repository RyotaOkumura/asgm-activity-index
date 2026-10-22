"""Fixed-schedule training.

Training runs for a fixed number of epochs and keeps the final weights.
There is no validation set and no early stopping, so the held-out test site
is never read during training.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from activity_index.data.pan_asgm import SegmentationDataset, TileKey
from activity_index.training.losses import CEDiceLoss
from activity_index.models import build_model

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_scheduler(optimizer, epochs: int, warmup_epochs: int, lr: float):
    """Linear warmup from 0.1 * lr, then cosine decay to 0.01 * lr (stepped per epoch)."""
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=lr * 0.01)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])


def train(
    cfg: dict,
    data_root: Path,
    train_keys: list[TileKey],
    out_dir: Path,
    seed: int,
    device: str = "cuda",
    model: torch.nn.Module | None = None,
) -> Path:
    """Train on ``train_keys`` and save ``out_dir/final_model.pth``."""
    set_seed(seed)
    data_cfg, train_cfg, loss_cfg = cfg["data"], cfg["training"], cfg["loss"]
    dataset = SegmentationDataset(
        data_root, train_keys, data_cfg["bands"], data_cfg["image_size"],
        data_cfg["normalize_mode"], augment=True,
    )
    loader = DataLoader(
        dataset, batch_size=train_cfg["batch_size"], shuffle=True, drop_last=True,
        num_workers=train_cfg.get("num_workers", 4), pin_memory=True,
    )
    if model is None:
        model = build_model(cfg["model"], in_channels=len(data_cfg["bands"]))
    model = model.to(device)
    criterion = CEDiceLoss(loss_cfg["ce_weight"], loss_cfg["dice_weight"])
    lr = train_cfg["lr"]
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=lr, weight_decay=train_cfg["weight_decay"])
    epochs = train_cfg["epochs"]
    scheduler = make_scheduler(optimizer, epochs, train_cfg["warmup_epochs"], lr)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()
        logger.info("epoch %d/%d  loss %.4f  lr %.2e", epoch + 1, epochs,
                    total / max(len(loader), 1), optimizer.param_groups[0]["lr"])

    path = out_dir / "final_model.pth"
    torch.save({"epoch": epochs, "model_state_dict": model.state_dict(), "config": cfg}, path)
    return path
