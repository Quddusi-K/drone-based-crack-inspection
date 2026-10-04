"""Shared utilities: config loading, seeding, timing, experiment directories."""
from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


@contextmanager
def timer():
    """Context manager returning elapsed seconds via ``t()`` after the block."""
    start = time.perf_counter()
    elapsed = {"s": None}
    yield lambda: elapsed["s"]
    elapsed["s"] = time.perf_counter() - start


def next_experiment_dir(results_root: str | Path, name: str) -> Path:
    """Create results/<NNN>_<name>/ with an incrementing experiment id."""
    root = ensure_dir(results_root)
    existing = [p.name for p in root.iterdir() if p.is_dir() and p.name[:3].isdigit()]
    next_id = 1 + max([int(n[:3]) for n in existing], default=0)
    return ensure_dir(root / f"{next_id:03d}_{name}")


def get_device(verbose: bool = True):
    import torch

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if verbose:
        print(f"CUDA available : {cuda}")
        print(f"GPU name       : {torch.cuda.get_device_name(0) if cuda else 'n/a'}")
        print(f"Device in use  : {device}")
    return device
