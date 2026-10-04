"""Pixel-level binary segmentation metrics.

IoU, Dice, precision, recall and F1 are computed from TP/FP/TN/FN. Pixel accuracy is reported but,
because cracks typically cover ~1-3 % of an image, a model predicting "no crack" everywhere already
scores ~97-99 % accuracy; IoU/Dice are therefore the primary metrics.
"""
from __future__ import annotations

import numpy as np

METRIC_KEYS = ["iou", "dice", "precision", "recall", "f1", "accuracy", "specificity"]


def confusion_counts(pred: np.ndarray, gt: np.ndarray) -> dict:
    p = pred.astype(bool)
    g = gt.astype(bool)
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    tn = int(np.logical_and(~p, ~g).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_from_counts(c: dict, empty_value: float = 1.0) -> dict:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    if tp + fp + fn == 0:  # empty prediction on empty ground truth: perfect
        iou = dice = precision = recall = f1 = empty_value
    else:
        iou = tp / (tp + fp + fn)
        dice = 2 * tp / (2 * tp + fp + fn)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    specificity = tn / (tn + fp) if tn + fp > 0 else 1.0
    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy, "specificity": specificity}


def segmentation_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    c = confusion_counts(pred, gt)
    out = metrics_from_counts(c)
    out.update(c)
    return out


def aggregate(per_image: list[dict]) -> dict:
    """Mean of per-image metrics plus 'global' (pixel-pooled) metrics."""
    if not per_image:
        return {}
    out = {k: float(np.mean([m[k] for m in per_image])) for k in METRIC_KEYS}
    pooled = {k: int(sum(m[k] for m in per_image)) for k in ["tp", "fp", "fn", "tn"]}
    out.update({f"global_{k}": v for k, v in metrics_from_counts(pooled).items()})
    out.update(pooled)
    out["n_images"] = len(per_image)
    # metrics restricted to images that actually contain cracks (avoids the empty-vs-empty = 1 convention)
    pos = [m for m in per_image if m["tp"] + m["fn"] > 0]
    out["n_positive_images"] = len(pos)
    for k in ["iou", "dice", "precision", "recall", "f1"]:
        out[f"pos_{k}"] = float(np.mean([m[k] for m in pos])) if pos else float("nan")
    return out
