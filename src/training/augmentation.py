"""Configurable augmentation (numpy/OpenCV only). Geometric ops transform image AND mask
identically; photometric ops touch only the image."""
from __future__ import annotations

import cv2
import numpy as np

DEFAULT_AUG = {
    "enabled": True,
    "hflip": 0.5, "vflip": 0.5,
    "rotate": {"p": 0.5, "max_deg": 15},
    "scale_crop": {"p": 0.5, "scale": [0.8, 1.2]},
    "perspective": {"p": 0.2, "strength": 0.05},
    "brightness": {"p": 0.5, "max_delta": 40},
    "contrast": {"p": 0.5, "range": [0.7, 1.3]},
    "gamma": {"p": 0.3, "range": [0.7, 1.5]},
    "noise": {"p": 0.3, "sigma": 8},
    "blur": {"p": 0.2, "ksize": 3},
    "color_shift": {"p": 0.3, "max_delta": 15},
}


def _u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


class Augmenter:
    def __init__(self, cfg: dict | None = None, seed: int = 0):
        self.cfg = {**DEFAULT_AUG, **(cfg or {})}
        self.rng = np.random.default_rng(seed)

    def _p(self, key):
        v = self.cfg.get(key)
        p = v if isinstance(v, (int, float)) else (v or {}).get("p", 0)
        return self.rng.random() < p

    # ------------------------------------------------------------ geometric
    def _geometric(self, img, mask):
        h, w = mask.shape
        if self._p("hflip"):
            img, mask = img[:, ::-1], mask[:, ::-1]
        if self._p("vflip"):
            img, mask = img[::-1], mask[::-1]
        M = None
        if self._p("rotate"):
            deg = self.rng.uniform(-self.cfg["rotate"]["max_deg"], self.cfg["rotate"]["max_deg"])
            M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        if self._p("scale_crop"):
            s = self.rng.uniform(*self.cfg["scale_crop"]["scale"])
            S = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s)
            M = S if M is None else np.vstack([S, [0, 0, 1]]) @ np.vstack([M, [0, 0, 1]])
            M = M[:2] if M.shape[0] == 3 else M
        if M is not None:
            img = cv2.warpAffine(np.ascontiguousarray(img), M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(np.ascontiguousarray(mask), M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        if self._p("perspective"):
            st = self.cfg["perspective"]["strength"]
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = (src + self.rng.uniform(-st, st, src.shape) * np.array([w, h])).astype(np.float32)
            H = cv2.getPerspectiveTransform(src, dst)
            img = cv2.warpPerspective(np.ascontiguousarray(img), H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpPerspective(np.ascontiguousarray(mask), H, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        return np.ascontiguousarray(img), np.ascontiguousarray(mask)

    # ----------------------------------------------------------- photometric
    def _photometric(self, img):
        f = img.astype(np.float32)
        if self._p("brightness"):
            f += self.rng.uniform(-self.cfg["brightness"]["max_delta"], self.cfg["brightness"]["max_delta"])
        if self._p("contrast"):
            c = self.rng.uniform(*self.cfg["contrast"]["range"])
            f = (f - f.mean()) * c + f.mean()
        if self._p("gamma"):
            g = self.rng.uniform(*self.cfg["gamma"]["range"])
            f = (np.clip(f, 0, 255) / 255.0) ** g * 255.0
        if self._p("color_shift"):
            f += self.rng.uniform(-self.cfg["color_shift"]["max_delta"], self.cfg["color_shift"]["max_delta"], size=3)
        if self._p("noise"):
            f += self.rng.normal(0, self.cfg["noise"]["sigma"], f.shape)
        img = _u8(f)
        if self._p("blur"):
            k = self.cfg["blur"]["ksize"]
            img = cv2.GaussianBlur(img, (k, k), 0)
        return img

    def __call__(self, img, mask):
        if not self.cfg.get("enabled", True):
            return img, mask
        img, mask = self._geometric(img, mask)
        img = self._photometric(img)
        return img, mask


def build_augmenter(cfg: dict | None, seed: int = 0):
    if cfg is None or not cfg.get("enabled", False):
        return None
    return Augmenter(cfg, seed)
