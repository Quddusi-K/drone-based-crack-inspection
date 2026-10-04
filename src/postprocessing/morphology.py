"""Morphological refinement of binary crack masks (shared by classical and deep pipelines)."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

DEFAULT_REFINE = {
    "enabled": True,
    "opening": 0,          # kernel size (0 = off)
    "closing": 5,          # closes gaps along cracks
    "fill_holes": True,
    "min_component_area": 40,
    "gap_closing": 0,      # larger-kernel closing to bridge fragmented cracks
    "dilate": 0,
    "erode": 0,
}


def _k(size: int, shape=cv2.MORPH_ELLIPSE):
    return cv2.getStructuringElement(shape, (size, size))


def morph(mask: np.ndarray, op: str, ksize: int, iterations: int = 1) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    ops = {"erode": cv2.MORPH_ERODE, "dilate": cv2.MORPH_DILATE, "open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE}
    return cv2.morphologyEx(m, ops[op], _k(ksize), iterations=iterations)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    if min_area <= 0:
        return m
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels].astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    return ndi.binary_fill_holes(mask > 0).astype(np.uint8)


def refine_mask(mask: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    """Apply the configured refinement sequence. Returns uint8 {0,1}."""
    cfg = {**DEFAULT_REFINE, **(cfg or {})}
    m = (mask > 0).astype(np.uint8)
    if not cfg.get("enabled", True):
        return m
    if cfg["opening"]:
        m = morph(m, "open", cfg["opening"])
    if cfg["closing"]:
        m = morph(m, "close", cfg["closing"])
    if cfg["gap_closing"]:
        m = morph(m, "close", cfg["gap_closing"])
    if cfg["fill_holes"]:
        m = fill_holes(m)
    if cfg["min_component_area"]:
        m = remove_small_components(m, cfg["min_component_area"])
    if cfg["dilate"]:
        m = morph(m, "dilate", cfg["dilate"])
    if cfg["erode"]:
        m = morph(m, "erode", cfg["erode"])
    return m


def probability_to_mask(prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (prob >= threshold).astype(np.uint8)
