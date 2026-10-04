"""Configurable, composable image preprocessing operations.

Every op takes an RGB uint8 image (H, W, 3) and returns an RGB uint8 image, so ops can be chained
freely and the output can be fed to either the classical or the deep-learning pipeline.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

Op = Callable[[np.ndarray], np.ndarray]


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- normalisation
def minmax_normalize(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    lo, hi = f.min(), f.max()
    if hi - lo < 1e-6:
        return img.copy()
    return _to_uint8((f - lo) / (hi - lo) * 255.0)


def standardize(img: np.ndarray) -> np.ndarray:
    """Zero-mean / unit-variance per image, then re-mapped to 0..255 (mean->128, +-3 sigma)."""
    f = img.astype(np.float32)
    z = (f - f.mean()) / (f.std() + 1e-6)
    return _to_uint8(128.0 + z * (128.0 / 3.0))


def rgb_normalize(img: np.ndarray) -> np.ndarray:
    """Chromaticity normalisation r=R/(R+G+B) scaled back to intensity. Removes global colour cast."""
    f = img.astype(np.float32) + 1e-6
    s = f.sum(axis=2, keepdims=True)
    chroma = f / s
    gray = f.mean(axis=2, keepdims=True)
    return _to_uint8(chroma * 3.0 * gray)


# ---------------------------------------------------------- contrast enhancement
def hist_equalize(img: np.ndarray) -> np.ndarray:
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


def clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    c = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    ycc[..., 0] = c.apply(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


# -------------------------------------------------------------------- denoising
def gaussian_blur(img: np.ndarray, ksize: int = 5, sigma: float = 0) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def median_blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.medianBlur(img, ksize)


def bilateral(img: np.ndarray, d: int = 7, sigma_color: float = 50, sigma_space: float = 7) -> np.ndarray:
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


# ---------------------------------------------------------- illumination correction
def gamma_correction(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def auto_gamma(img: np.ndarray, target_mean: float = 0.5) -> np.ndarray:
    """Choose gamma so that the mean luminance maps to ``target_mean``."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    m = float(np.clip(gray.mean(), 1e-3, 1 - 1e-3))
    gamma = np.log(m) / np.log(target_mean)
    return gamma_correction(img, 1.0 / gamma)


def local_contrast_normalization(img: np.ndarray, ksize: int = 31) -> np.ndarray:
    """(I - local_mean) / local_std on luminance, remapped to 0..255."""
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y = ycc[..., 0]
    mean = cv2.GaussianBlur(y, (ksize, ksize), 0)
    sq = cv2.GaussianBlur(y * y, (ksize, ksize), 0)
    std = np.sqrt(np.maximum(sq - mean * mean, 1e-6))
    z = (y - mean) / (std + 1e-3)
    ycc[..., 0] = np.clip(128.0 + z * 40.0, 0, 255)
    return cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def background_illumination_correction(img: np.ndarray, ksize: int = 51) -> np.ndarray:
    """Estimate slowly varying background with a large morphological closing (cracks are dark, so
    closing removes them) and divide the image by it -> flat-field correction."""
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y = ycc[..., 0]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bg = cv2.morphologyEx(y, cv2.MORPH_CLOSE, kernel)
    bg = cv2.GaussianBlur(bg, (ksize, ksize), 0)
    corrected = y / (bg + 1e-3) * float(bg.mean())
    ycc[..., 0] = np.clip(corrected, 0, 255)
    return cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def retinex(img: np.ndarray, sigma: float = 40.0) -> np.ndarray:
    """Single-scale Retinex on luminance: log(I) - log(G_sigma * I)."""
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y = ycc[..., 0] + 1.0
    illum = cv2.GaussianBlur(y, (0, 0), sigma)
    r = np.log(y) - np.log(illum + 1.0)
    r = (r - r.min()) / (r.max() - r.min() + 1e-6) * 255.0
    ycc[..., 0] = r
    return cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2RGB)


OPS: dict[str, Op] = {
    "minmax": minmax_normalize,
    "standardize": standardize,
    "rgb_normalize": rgb_normalize,
    "hist_eq": hist_equalize,
    "clahe": clahe,
    "gaussian": gaussian_blur,
    "median": median_blur,
    "bilateral": bilateral,
    "gamma": gamma_correction,
    "auto_gamma": auto_gamma,
    "local_contrast": local_contrast_normalization,
    "background_correction": background_illumination_correction,
    "retinex": retinex,
}

# Named presets used throughout the experiments.
PRESETS: dict[str, list] = {
    "raw": [],
    "normalized": ["minmax"],
    "contrast": ["clahe"],
    "illumination": ["background_correction"],
    "retinex": ["retinex"],
    "combined": ["background_correction", {"name": "clahe", "clip_limit": 2.0}, {"name": "bilateral"}],
}


def build_pipeline(steps: list | str | None) -> Op:
    """Build a callable from a preset name or a list of op specs.

    Each step is either an op name (str) or a dict ``{"name": ..., **kwargs}``.
    """
    if steps is None:
        steps = []
    if isinstance(steps, str):
        steps = PRESETS[steps]
    fns = []
    for step in steps:
        if isinstance(step, str):
            fns.append(OPS[step])
        else:
            kw = dict(step)
            name = kw.pop("name")
            fns.append(lambda im, _f=OPS[name], _kw=kw: _f(im, **_kw))

    def run(img: np.ndarray) -> np.ndarray:
        for f in fns:
            img = f(img)
        return img

    run.steps = steps  # type: ignore[attr-defined]
    return run


def crack_separability(img: np.ndarray, mask: np.ndarray) -> dict:
    """How separable are crack pixels from background in luminance?

    Returns Michelson-style contrast and Fisher's discriminant ratio between the two
    intensity populations. Higher is better. Used to quantify the effect of preprocessing.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    crack = gray[mask > 0]
    bg = gray[mask == 0]
    if crack.size == 0 or bg.size == 0:
        return {"contrast": np.nan, "fisher": np.nan}
    mu_c, mu_b = crack.mean(), bg.mean()
    contrast = abs(mu_b - mu_c) / (mu_b + mu_c + 1e-6)
    fisher = (mu_b - mu_c) ** 2 / (crack.var() + bg.var() + 1e-6)
    return {"contrast": float(contrast), "fisher": float(fisher)}
