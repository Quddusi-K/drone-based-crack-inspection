"""Simulated photometric variation. These change appearance only; crack geometry (and therefore
the ground-truth mask) is never modified."""
from __future__ import annotations

import cv2
import numpy as np


def _u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def brightness(img, delta=60):
    return _u8(img.astype(np.float32) + delta)


def contrast(img, factor=0.5):
    f = img.astype(np.float32)
    return _u8((f - f.mean()) * factor + f.mean())


def gamma(img, g=2.2):
    lut = np.array([(i / 255.0) ** g * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def shadow(img, strength=0.5, seed=0):
    """Darken one side of the image with a soft random linear gradient (cast shadow)."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    angle = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w]
    proj = (xx * np.cos(angle) + yy * np.sin(angle))
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
    ramp = np.clip((proj - 0.4) / 0.2, 0, 1)
    factor = 1.0 - strength * ramp
    return _u8(img.astype(np.float32) * factor[..., None])


def local_illumination(img, strength=0.6, seed=0):
    """Multiplicative Gaussian spotlight / vignetting."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
    yy, xx = np.mgrid[0:h, 0:w]
    sigma = 0.35 * max(h, w)
    spot = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    factor = (1 - strength) + strength * spot
    return _u8(img.astype(np.float32) * (factor[..., None] + strength * 0.3))


def color_shift(img, shift=(20, -10, -15)):
    return _u8(img.astype(np.float32) + np.array(shift, dtype=np.float32))


def low_light_noise(img, scale=0.4, sigma=10, seed=0):
    rng = np.random.default_rng(seed)
    f = img.astype(np.float32) * scale + rng.normal(0, sigma, img.shape)
    return _u8(f)


PERTURBATIONS = {
    "original": lambda im: im,
    "bright+60": lambda im: brightness(im, 60),
    "dark-60": lambda im: brightness(im, -60),
    "low_contrast": lambda im: contrast(im, 0.5),
    "high_contrast": lambda im: contrast(im, 1.6),
    "gamma_0.5": lambda im: gamma(im, 0.5),
    "gamma_2.2": lambda im: gamma(im, 2.2),
    "shadow": lambda im: shadow(im, 0.55),
    "local_illumination": lambda im: local_illumination(im, 0.6),
    "color_shift": lambda im: color_shift(im),
    "low_light_noise": lambda im: low_light_noise(im),
}
