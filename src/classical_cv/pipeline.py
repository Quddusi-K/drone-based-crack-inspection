"""Classical (non-learning) crack detection: preprocessing -> edge/threshold -> morphology -> CC filtering.

Runs on CPU only. Cracks are assumed darker than the surrounding surface, so threshold-based
detectors operate on the *inverted* luminance (or on a black-hat response).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.postprocessing.morphology import morph, remove_small_components
from src.preprocessing.ops import build_pipeline


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# ------------------------------------------------------------- edge detectors
def canny(gray, low=50, high=150, blur=3):
    g = cv2.GaussianBlur(gray, (blur, blur), 0) if blur else gray
    return (cv2.Canny(g, low, high) > 0).astype(np.uint8)


def sobel(gray, ksize=3, percentile=95.0):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    return (mag >= np.percentile(mag, percentile)).astype(np.uint8)


def laplacian(gray, ksize=3, percentile=96.0):
    lap = cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F, ksize=ksize)
    resp = np.maximum(lap, 0)  # dark ridge on bright background -> positive Laplacian
    return (resp >= np.percentile(resp, percentile)).astype(np.uint8)


# ---------------------------------------------------------------- thresholds
def global_threshold(gray, value=70):
    return (gray <= value).astype(np.uint8)


def otsu(gray):
    t, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (m > 0).astype(np.uint8)


def adaptive(gray, block=31, c=8):
    m = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, c)
    return (m > 0).astype(np.uint8)


def blackhat_otsu(gray, ksize=15):
    """Morphological black-hat enhances thin dark structures, then Otsu on the response."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    t, m = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (m > 0).astype(np.uint8)


def blackhat_fixed(gray, ksize=15, thresh=35):
    """Black-hat response with a fixed absolute threshold: does not fire on crack-free texture."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    return (bh >= thresh).astype(np.uint8)


DETECTORS = {
    "canny": canny, "sobel": sobel, "laplacian": laplacian,
    "global": global_threshold, "otsu": otsu, "adaptive": adaptive, "blackhat_otsu": blackhat_otsu, "blackhat_fixed": blackhat_fixed,
}


@dataclass
class ClassicalConfig:
    name: str
    preprocessing: str | list = "raw"
    detector: str = "canny"
    detector_kwargs: dict = field(default_factory=dict)
    morphology: list = field(default_factory=list)  # e.g. [["close", 5], ["open", 3]]
    min_component_area: int = 0


class ClassicalCrackDetector:
    def __init__(self, cfg: ClassicalConfig):
        self.cfg = cfg
        self.pre = build_pipeline(cfg.preprocessing)
        self.det = DETECTORS[cfg.detector]

    def __call__(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        gray = _gray(self.pre(img))
        mask = self.det(gray, **self.cfg.detector_kwargs)
        for op, k in self.cfg.morphology:
            mask = morph(mask, op, int(k))
        if self.cfg.min_component_area:
            mask = remove_small_components(mask, self.cfg.min_component_area)
        return mask.astype(np.uint8), time.perf_counter() - t0


# Experiments from the project brief (plus a black-hat baseline, a standard crack detector).
EXPERIMENTS: list[ClassicalConfig] = [
    ClassicalConfig("E1_raw_canny", "raw", "canny"),
    ClassicalConfig("E2_clahe_canny", "contrast", "canny"),
    ClassicalConfig("E3_illum_canny", "illumination", "canny"),
    ClassicalConfig("E4_clahe_adaptive", "contrast", "adaptive"),
    ClassicalConfig("E5_clahe_canny_morph", "contrast", "canny", morphology=[["close", 5]]),
    ClassicalConfig("E6_combined_adaptive_morph_cc", "combined", "adaptive", morphology=[["open", 3], ["close", 5]], min_component_area=80),
    ClassicalConfig("E7_raw_otsu", "raw", "otsu"),
    ClassicalConfig("E8_raw_sobel", "raw", "sobel"),
    ClassicalConfig("E9_raw_laplacian", "raw", "laplacian"),
    ClassicalConfig("E10_raw_blackhat_otsu", "raw", "blackhat_otsu"),
    ClassicalConfig("E11_clahe_blackhat_morph_cc", "contrast", "blackhat_otsu", morphology=[["close", 5]], min_component_area=80),
    ClassicalConfig("E12_combined_blackhat_morph_cc", "combined", "blackhat_otsu", morphology=[["close", 5]], min_component_area=80),
    ClassicalConfig("E13_raw_blackhat_fixed_morph_cc", "raw", "blackhat_fixed", morphology=[["close", 5]], min_component_area=80),
    ClassicalConfig("E14_illum_blackhat_fixed_morph_cc", "illumination", "blackhat_fixed", morphology=[["close", 5]], min_component_area=80),
]
