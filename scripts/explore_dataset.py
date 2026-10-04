"""Inspect the dataset: counts, mask statistics, sample grid. Laptop-friendly."""
import argparse
import numpy as np
from _common import ROOT, resolve_config, add_common_args, md_table
from src.data.dataset import discover_pairs, load_image, load_mask, dataset_summary, split_samples
from src.utils import ensure_dir, save_json
from src.visualization.plots import panel, overlay


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--data", default=None)
    a = ap.parse_args()
    cfg = resolve_config(a.config, a.experiment, a.profile, a.overrides)
    root = a.data or cfg["dataset"]["path"]
    samples = discover_pairs(root)
    summary = dataset_summary(samples)
    splits = split_samples(samples, cfg["dataset"]["split"], cfg["seed"])
    summary["splits"] = {k: len(v) for k, v in splits.items()}
    fracs, shapes = [], set()
    for s in samples:
        img = load_image(s.image_path)
        m = load_mask(s.mask_path, img.shape)
        shapes.add(img.shape)
        fracs.append(m.mean())
    fracs = np.array(fracs)
    summary["image_shapes"] = [list(s) for s in shapes]
    summary["crack_pixel_fraction"] = {"mean": float(fracs.mean()), "median": float(np.median(fracs)),
                                       "max": float(fracs.max()), "fraction_images_with_crack": float((fracs > 0).mean())}
    out = ensure_dir(ROOT / "results" / "dataset")
    save_json(summary, out / "summary.json")
    pos = [s for s in samples if s.mask_path is not None and load_mask(s.mask_path).sum() > 0][:4]
    neg = [s for s in samples if s.mask_path is None or load_mask(s.mask_path).sum() == 0][:2]
    ims, titles = [], []
    for s in pos + neg:
        img = load_image(s.image_path); m = load_mask(s.mask_path, img.shape)
        ims += [img, m, overlay(img, m)]
        titles += [f"{s.group}: {s.stem[:16]}", "mask", "overlay"]
    panel(ims, titles, out / "samples.png", ncols=6, suptitle="Dataset samples (image / mask / overlay)")
    print(md_table([{"key": k, "value": v} for k, v in summary.items()], ["key", "value"]))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
