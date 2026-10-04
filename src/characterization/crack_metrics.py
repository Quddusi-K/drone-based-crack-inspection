"""Quantitative crack characterisation from a binary mask.

All measurements are in pixels / image coordinates; no metric calibration is assumed.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


def skeleton_of(mask: np.ndarray) -> np.ndarray:
    return skeletonize(mask > 0).astype(np.uint8)


def skeleton_length(skel: np.ndarray) -> float:
    """Length accounting for diagonal steps (sqrt(2)) between 8-connected skeleton pixels."""
    s = skel.astype(bool)
    if s.sum() == 0:
        return 0.0
    # count horizontal/vertical and diagonal links, each link counted once
    hv = np.logical_and(s[:, :-1], s[:, 1:]).sum() + np.logical_and(s[:-1, :], s[1:, :]).sum()
    diag = np.logical_and(s[:-1, :-1], s[1:, 1:]).sum() + np.logical_and(s[:-1, 1:], s[1:, :-1]).sum()
    return float(hv + np.sqrt(2) * diag)


def branch_and_end_points(skel: np.ndarray) -> tuple[int, int]:
    s = skel.astype(np.uint8)
    if s.sum() == 0:
        return 0, 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbours = ndi.convolve(s, kernel, mode="constant") - s
    branch_px = np.logical_and(s > 0, neighbours >= 3)
    # adjacent junction pixels belong to the same junction -> count clusters, not pixels
    _, branch = ndi.label(branch_px, structure=np.ones((3, 3)))
    ends = int(np.logical_and(s > 0, neighbours == 1).sum())
    return branch, ends


def width_stats(mask: np.ndarray, skel: np.ndarray) -> tuple[float, float]:
    """Width = 2 x distance-to-background sampled along the skeleton (centre line)."""
    if skel.sum() == 0:
        return 0.0, 0.0
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    w = 2.0 * dist[skel > 0]
    return float(w.mean()), float(w.max())


def dominant_orientation(mask: np.ndarray) -> float:
    """Dominant orientation in degrees (0 = horizontal, 90 = vertical) via PCA of crack pixels."""
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 2:
        return float("nan")
    coords = np.stack([xs - xs.mean(), ys - ys.mean()], axis=1)
    cov = coords.T @ coords / len(xs)
    evals, evecs = np.linalg.eigh(cov)
    vx, vy = evecs[:, np.argmax(evals)]
    angle = np.degrees(np.arctan2(-vy, vx))  # image y axis points down
    return float(angle % 180.0)


def component_stats(mask: np.ndarray) -> list[dict]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, n):
        ys, xs = np.nonzero(labels == i)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts) if len(pts) >= 3 else ((xs.mean(), ys.mean()), (1, 1), 0)
        long_, short = max(w, h), max(min(w, h), 1.0)
        comps.append({"label": i, "area": int(stats[i, cv2.CC_STAT_AREA]),
                      "bbox": [int(v) for v in stats[i, :4]], "aspect_ratio": float(long_ / short)})
    return comps


def characterize(mask: np.ndarray) -> dict:
    m = (mask > 0).astype(np.uint8)
    skel = skeleton_of(m)
    mean_w, max_w = width_stats(m, skel)
    branch, ends = branch_and_end_points(skel)
    comps = component_stats(m)
    area = int(m.sum())
    return {
        "crack_area_pixels": area,
        "crack_length_pixels": skeleton_length(skel),
        "mean_width_pixels": mean_w,
        "max_width_pixels": max_w,
        "orientation_degrees": dominant_orientation(m),
        "num_components": len(comps),
        "crack_density": float(area / m.size),
        "branch_points": branch,
        "end_points": ends,
        "mean_component_aspect_ratio": float(np.mean([c["aspect_ratio"] for c in comps])) if comps else 0.0,
        "image_shape": list(m.shape),
    }


def angular_error(a: float, b: float) -> float:
    """Smallest difference between two orientations in [0,180) degrees."""
    if np.isnan(a) or np.isnan(b):
        return float("nan")
    d = abs(a - b) % 180.0
    return float(min(d, 180.0 - d))


def measurement_errors(pred_props: list[dict], gt_props: list[dict]) -> dict:
    """MAE / RMSE between measurements derived from predicted masks and from ground-truth masks.
    Ground-truth masks act as the 'manual measurement' reference."""
    out = {}
    for key in ["crack_length_pixels", "mean_width_pixels", "max_width_pixels", "crack_area_pixels"]:
        p = np.array([d[key] for d in pred_props], dtype=float)
        g = np.array([d[key] for d in gt_props], dtype=float)
        err = p - g
        out[f"{key}_mae"] = float(np.abs(err).mean())
        out[f"{key}_rmse"] = float(np.sqrt((err ** 2).mean()))
        if key == "crack_area_pixels":
            rel = np.abs(err) / np.maximum(g, 1)
            out["crack_area_relative_error"] = float(rel.mean())
    ang = [angular_error(p["orientation_degrees"], g["orientation_degrees"]) for p, g in zip(pred_props, gt_props)]
    ang = [a for a in ang if not np.isnan(a)]
    out["orientation_mean_abs_angular_error"] = float(np.mean(ang)) if ang else float("nan")
    out["n_images"] = len(pred_props)
    return out
