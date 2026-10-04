"""Dataset discovery, loading, splitting and a PyTorch Dataset for crack segmentation.

Supports several on-disk layouts without imposing structure on the Kaggle download:

* ``<root>/Positive/Images`` + ``<root>/Positive/Masks`` and ``<root>/Negative/Images`` + ``<root>/Negative/Mask``
  (layout of the local ``crack_samples`` folder)
* ``<root>/train/images`` + ``<root>/train/masks`` (and ``test/``), as in the Kaggle crack segmentation dataset
* any directory named ``images``/``Images`` with a sibling ``masks``/``Masks``/``Mask`` directory

Masks are stored as JPEG in the source data, so they are binarised at 127 on load.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGE_DIR_NAMES = {"images", "image", "img", "imgs"}
MASK_DIR_NAMES = {"masks", "mask", "labels", "label", "gt"}


@dataclass
class Sample:
    image_path: Path
    mask_path: Path | None
    group: str  # e.g. "Positive", "Negative", "train", "test"

    @property
    def stem(self) -> str:
        return self.image_path.stem


def discover_pairs(root: str | Path) -> list[Sample]:
    """Walk ``root`` and return image/mask pairs matched by file stem."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    samples: list[Sample] = []
    for img_dir in sorted(p for p in root.rglob("*") if p.is_dir() and p.name.lower() in IMAGE_DIR_NAMES):
        mask_dir = next((s for s in img_dir.parent.iterdir() if s.is_dir() and s.name.lower() in MASK_DIR_NAMES), None)
        mask_index = {}
        if mask_dir is not None:
            mask_index = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMAGE_EXT}
        group = img_dir.parent.name
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXT:
                continue
            samples.append(Sample(img, mask_index.get(img.stem), group))
    if not samples:
        raise RuntimeError(f"No images found under {root}. Expected an 'images' directory with sibling 'masks'.")
    return samples


def load_image(path: str | Path, size: int | None = None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None and (img.shape[0] != size or img.shape[1] != size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def load_mask(path: str | Path | None, shape: tuple[int, int] | None = None, size: int | None = None) -> np.ndarray:
    """Return a binary uint8 mask in {0,1}. A missing mask (negative sample) is all zeros."""
    if path is None:
        if shape is None:
            raise ValueError("shape required when mask path is None")
        h, w = shape[:2]
        return np.zeros((size or h, size or w), dtype=np.uint8)
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"Could not read mask {path}")
    if size is not None and (m.shape[0] != size or m.shape[1] != size):
        m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def split_samples(samples: Sequence[Sample], ratios=(0.8, 0.1, 0.1), seed: int = 42) -> dict[str, list[Sample]]:
    """Stratified (by group) random train/val/test split. If the dataset already has
    a ``test`` group, that group is kept as the test set."""
    groups: dict[str, list[Sample]] = {}
    for s in samples:
        groups.setdefault(s.group.lower(), []).append(s)
    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": []}
    predefined_test = groups.pop("test", None)
    if predefined_test is not None:
        out["test"].extend(predefined_test)
        ratios = (ratios[0] / (ratios[0] + ratios[1]), ratios[1] / (ratios[0] + ratios[1]), 0.0)
    for _, items in sorted(groups.items()):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))
        out["train"].extend(items[:n_train])
        out["val"].extend(items[n_train:n_train + n_val])
        out["test"].extend(items[n_train + n_val:])
    return out


def dataset_summary(samples: Sequence[Sample]) -> dict:
    groups = {}
    for s in samples:
        g = groups.setdefault(s.group, {"images": 0, "with_mask": 0})
        g["images"] += 1
        g["with_mask"] += int(s.mask_path is not None)
    return {"total": len(samples), "groups": groups}


class CrackSegDataset:
    """Minimal torch-compatible dataset (imported lazily so the classical pipeline needs no torch)."""

    def __init__(
        self,
        samples: Sequence[Sample],
        image_size: int = 256,
        preprocess: Callable[[np.ndarray], np.ndarray] | None = None,
        augment: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    ):
        self.samples = list(samples)
        self.image_size = image_size
        self.preprocess = preprocess
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def load_pair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        s = self.samples[idx]
        img = load_image(s.image_path, self.image_size)
        mask = load_mask(s.mask_path, img.shape, self.image_size)
        return img, mask

    def __getitem__(self, idx: int):
        import torch

        img, mask = self.load_pair(idx)
        if self.preprocess is not None:
            img = self.preprocess(img)
        if self.augment is not None:
            img, mask = self.augment(img, mask)
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0
        y = torch.from_numpy(np.ascontiguousarray(mask[None].astype(np.float32)))
        return x, y, self.samples[idx].stem
