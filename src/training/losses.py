"""Segmentation losses for imbalanced thin-structure segmentation. All take logits."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits, target, smooth=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + smooth) / (denom + smooth)).mean()


def bce_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target)


def bce_dice_loss(logits, target, bce_weight=1.0, dice_weight=1.0):
    return bce_weight * bce_loss(logits, target) + dice_weight * dice_loss(logits, target)


def focal_loss(logits, target, alpha=0.75, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    at = alpha * target + (1 - alpha) * (1 - target)
    return (at * (1 - pt) ** gamma * bce).mean()


def tversky_loss(logits, target, alpha=0.3, beta=0.7, smooth=1.0):
    """alpha weights FP, beta weights FN; beta > alpha favours recall of thin cracks."""
    p = torch.sigmoid(logits)
    tp = (p * target).sum(dim=(1, 2, 3))
    fp = (p * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - p) * target).sum(dim=(1, 2, 3))
    return (1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)).mean()


def focal_tversky_loss(logits, target, gamma=0.75, **kw):
    return tversky_loss(logits, target, **kw) ** gamma


LOSSES = {"bce": bce_loss, "dice": dice_loss, "bce_dice": bce_dice_loss, "dice_bce": bce_dice_loss,
          "focal": focal_loss, "tversky": tversky_loss, "focal_tversky": focal_tversky_loss}


class SegLoss(nn.Module):
    def __init__(self, name: str = "bce_dice", **kwargs):
        super().__init__()
        self.fn = LOSSES[name]
        self.kwargs = kwargs
        self.name = name

    def forward(self, logits, target):
        return self.fn(logits, target, **self.kwargs)


def build_loss(cfg: dict | str) -> SegLoss:
    if isinstance(cfg, str):
        return SegLoss(cfg)
    cfg = dict(cfg)
    return SegLoss(cfg.pop("name", "bce_dice"), **cfg)
