"""Characterise a binary crack mask (or every mask in a directory).
    python scripts/characterize.py --input prediction.png [--image original.jpg] [--gt mask.png]"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from _common import ROOT, md_table
from src.characterization.crack_metrics import characterize, skeleton_of
from src.data.dataset import load_image, load_mask
from src.utils import ensure_dir, save_json
from src.visualization.plots import characterization_figure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="mask image or directory of masks")
    ap.add_argument("--image", default=None, help="original image (or directory) for visualisation")
    ap.add_argument("--gt", default=None, help="ground-truth mask (or directory)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    inp = Path(a.input)
    masks = sorted(inp.glob("*.png")) + sorted(inp.glob("*.jpg")) if inp.is_dir() else [inp]
    out = ensure_dir(a.out or ROOT / "results" / "characterization")
    rows = []
    for mp in masks:
        m = load_mask(mp)
        props = characterize(m)
        props["name"] = mp.stem
        rows.append(props)
        img = None
        if a.image:
            ip = Path(a.image)
            ip = next((p for p in ip.glob(mp.stem + ".*")), None) if ip.is_dir() else ip
            img = load_image(ip, m.shape[0]) if ip else None
        if img is None:
            img = np.stack([m * 255] * 3, axis=-1)
        gt = m
        if a.gt:
            gp = Path(a.gt)
            gp = next((p for p in gp.glob(mp.stem + ".*")), None) if gp.is_dir() else gp
            gt = load_mask(gp, m.shape, m.shape[0]) if gp else m
        labels = cv2.connectedComponents(m, connectivity=8)[1]
        characterization_figure(img, m, gt, skeleton_of(m), labels, props, out / f"{mp.stem}_characterization.png", title=mp.stem)
    save_json(rows, out / "measurements.json")
    cols = ["name", "crack_area_pixels", "crack_length_pixels", "mean_width_pixels", "max_width_pixels", "orientation_degrees", "num_components", "crack_density", "branch_points"]
    (out / "measurements.md").write_text(md_table(rows, cols, "{:.2f}") + "\n")
    print(md_table(rows, cols, "{:.2f}"))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
