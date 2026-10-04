"""Matplotlib figures used across the project (report quality, saved to disk, never shown)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 9})


def overlay(img: np.ndarray, pred: np.ndarray, gt: np.ndarray | None = None, alpha: float = 0.55) -> np.ndarray:
    """Green = TP, red = FP, blue = FN (if gt given); otherwise red = prediction."""
    out = img.copy().astype(np.float32)
    col = np.zeros_like(out)
    p = pred > 0
    if gt is None:
        col[p] = (255, 0, 0)
        m = p
    else:
        g = gt > 0
        col[p & g] = (0, 255, 0)
        col[p & ~g] = (255, 0, 0)
        col[~p & g] = (0, 90, 255)
        m = p | g
    out[m] = (1 - alpha) * out[m] + alpha * col[m]
    return out.astype(np.uint8)


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def panel(images: list, titles: list, path, ncols: int | None = None, suptitle: str | None = None, cmap_binary=True):
    n = len(images)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.7 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, im, t in zip(axes, images, titles):
        if im.ndim == 2:
            ax.imshow(im, cmap="gray" if cmap_binary else "magma", vmin=0, vmax=1 if im.max() <= 1 else None)
        else:
            ax.imshow(im)
        ax.set_title(t)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    _save(fig, path)


def training_curves(history: list[dict], path):
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    axes[0].plot(ep, [h["train_loss"] for h in history], label="train")
    axes[0].plot(ep, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(ep, [h["val_dice"] for h in history], label="Dice (mean per image)")
    axes[1].plot(ep, [h["val_iou"] for h in history], label="IoU (mean per image)")
    if "val_global_dice" in history[0]:
        axes[1].plot(ep, [h["val_global_dice"] for h in history], label="Dice (pixel-pooled)")
    axes[1].set_title("Validation Dice / IoU"); axes[1].legend()
    axes[2].plot(ep, [h["val_precision"] for h in history], label="Precision")
    axes[2].plot(ep, [h["val_recall"] for h in history], label="Recall")
    axes[2].set_title("Validation Precision / Recall"); axes[2].legend()
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    _save(fig, path)


def confusion_matrix_plot(counts: dict, path, title="Pixel confusion matrix (test set)"):
    m = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]], dtype=float)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(np.log1p(m), cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(m[i, j]):,}", ha="center", va="center", color="black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred bg", "pred crack"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["gt bg", "gt crack"])
    ax.set_title(title)
    _save(fig, path)


def bar_comparison(labels: list[str], series: dict[str, list[float]], path, title="", ylabel="", rotate=45):
    x = np.arange(len(labels))
    k = len(series)
    w = 0.8 / k
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(labels) * k), 3.6))
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i - (k - 1) / 2) * w, vals, w, label=name)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=rotate, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    if k > 1:
        ax.legend()
    _save(fig, path)


def threshold_curve(thresholds, metrics: dict[str, list[float]], path, title="Probability threshold sweep"):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for k, v in metrics.items():
        ax.plot(thresholds, v, marker="o", ms=3, label=k)
    ax.set_xlabel("threshold"); ax.set_title(title); ax.grid(alpha=0.3); ax.legend()
    _save(fig, path)


def pr_curve(recalls, precisions, path, title="Precision-Recall (threshold sweep)"):
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.plot(recalls, precisions, marker="o", ms=3)
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title(title); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _save(fig, path)


def characterization_figure(img, pred, gt, skel, labels, props: dict, path, title=""):
    """Original | prediction | GT | overlay | skeleton | components | measurements."""
    rng = np.random.default_rng(0)
    lut = np.vstack([[0, 0, 0], rng.integers(60, 255, size=(max(labels.max(), 1), 3))]).astype(np.uint8)
    comp_rgb = lut[labels]
    skel_rgb = img.copy(); skel_rgb[skel > 0] = (255, 255, 0)
    fig, axes = plt.subplots(1, 7, figsize=(19, 3.1))
    items = [(img, "original"), (pred, "predicted mask"), (gt, "ground truth"),
             (overlay(img, pred, gt), "overlay (TP green, FP red, FN blue)"), (skel_rgb, "skeleton"), (comp_rgb, "connected components")]
    for ax, (im, t) in zip(axes, items):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None); ax.set_title(t); ax.axis("off")
    axes[6].axis("off")
    txt = "\n".join([
        f"area: {props['crack_area_pixels']} px",
        f"length: {props['crack_length_pixels']:.1f} px",
        f"mean width: {props['mean_width_pixels']:.2f} px",
        f"max width: {props['max_width_pixels']:.2f} px",
        f"orientation: {props['orientation_degrees']:.1f} deg",
        f"components: {props['num_components']}",
        f"density: {props['crack_density']*100:.2f} %",
        f"branch points: {props['branch_points']}",
    ])
    axes[6].text(0, 0.95, txt, va="top", fontsize=9, family="monospace")
    axes[6].set_title("measurements (pixels)")
    if title:
        fig.suptitle(title, fontsize=10)
    _save(fig, path)
