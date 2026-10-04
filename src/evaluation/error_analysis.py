"""Automated error analysis: categorise per-image failures and render representative cases."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.evaluation.metrics import segmentation_metrics
from src.postprocessing.morphology import remove_small_components
from src.visualization.plots import overlay, panel

CATEGORIES = ["false_positive", "false_negative", "boundary_error", "fragmentation", "merging", "good"]


def _n_components(mask, min_area=10):
    m = remove_small_components(mask, min_area)
    return cv2.connectedComponentsWithStats(m, connectivity=8)[0] - 1


def _dilate(m, k=7):
    return cv2.dilate((m > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def categorize(pred: np.ndarray, gt: np.ndarray) -> dict:
    met = segmentation_metrics(pred, gt)
    gt_area, pred_area = int(gt.sum()), int(pred.sum())
    n_gt, n_pred = _n_components(gt), _n_components(pred)
    tol_iou = segmentation_metrics(_dilate(pred), _dilate(gt))["iou"]
    cats = []
    if gt_area == 0 and pred_area > 20:
        cats.append("false_positive")
    if gt_area > 0 and met["recall"] < 0.2:
        cats.append("false_negative")
    if gt_area > 0 and met["iou"] < 0.5 and tol_iou > 0.6 and met["recall"] >= 0.2:
        cats.append("boundary_error")
    if n_gt >= 1 and n_pred >= 2 * n_gt + 1 and met["recall"] >= 0.2:
        cats.append("fragmentation")
    if n_gt >= 2 and n_pred <= n_gt / 2 and met["recall"] >= 0.2:
        cats.append("merging")
    if not cats:
        cats.append("good")
    return {"categories": cats, "iou": met["iou"], "dice": met["dice"], "precision": met["precision"],
            "recall": met["recall"], "tolerant_iou": tol_iou, "n_gt_components": n_gt, "n_pred_components": n_pred}


def run_error_analysis(images: list[np.ndarray], preds: list[np.ndarray], gts: list[np.ndarray], names: list[str],
                       out_dir: str | Path, max_examples: int = 4) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, p, g in zip(names, preds, gts):
        r = categorize(p, g)
        r["name"] = name
        records.append(r)
    counts = {c: sum(c in r["categories"] for r in records) for c in CATEGORIES}
    # quantitative: mean IoU by category
    by_cat = {c: float(np.mean([r["iou"] for r in records if c in r["categories"]])) if counts[c] else None for c in CATEGORIES}
    for c in CATEGORIES:
        if c == "good":
            continue
        idx = [i for i, r in enumerate(records) if c in r["categories"]]
        idx = sorted(idx, key=lambda i: records[i]["iou"])[:max_examples]
        if not idx:
            continue
        ims, titles = [], []
        for i in idx:
            ims += [images[i], gts[i], preds[i], overlay(images[i], preds[i], gts[i])]
            titles += [f"{names[i][:18]}", "ground truth", f"pred (IoU {records[i]['iou']:.2f})", "overlay TP/FP/FN"]
        panel(ims, titles, out_dir / f"failures_{c}.png", ncols=4, suptitle=f"Failure category: {c} ({counts[c]} images)")
    return {"counts": counts, "mean_iou_by_category": by_cat, "n_images": len(records), "per_image": records}
