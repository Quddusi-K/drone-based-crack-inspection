"""Run all classical CV experiments (Experiments A/B and the E1-E12 grid) and record a comparison table."""
import argparse
import csv
import numpy as np
from _common import ROOT, resolve_config, add_common_args, md_table
from src.classical_cv.pipeline import EXPERIMENTS, ClassicalCrackDetector
from src.data.dataset import discover_pairs, load_image, load_mask, split_samples
from src.evaluation.metrics import aggregate, segmentation_metrics
from src.utils import ensure_dir, save_json
from src.visualization.plots import panel, overlay, bar_comparison

SUMMARY = ROOT / "results" / "experiments_summary.csv"
SUMMARY_COLS = ["experiment", "method", "preprocessing", "augmentation", "loss", "morphology", "threshold",
                "iou", "dice", "precision", "recall", "f1", "pos_iou", "pos_dice", "global_iou", "global_dice", "inference_ms", "n_test_images"]


def append_summary(row: dict):
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    new = not SUMMARY.exists()
    with open(SUMMARY, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--data", default=None)
    ap.add_argument("--split", default="all", choices=["all", "test"], help="evaluate on all samples or the held-out test split")
    a = ap.parse_args()
    cfg = resolve_config(a.config, a.experiment, a.profile, a.overrides)
    samples = discover_pairs(a.data or cfg["dataset"]["path"])
    if a.split == "test":
        samples = split_samples(samples, cfg["dataset"]["split"], cfg["seed"])["test"]
    out = ensure_dir(ROOT / "results" / "classical_cv")
    data = [(s, load_image(s.image_path)) for s in samples]
    gts = [load_mask(s.mask_path, img.shape) for s, img in data]
    rows, preds_by_exp = [], {}
    for ecfg in EXPERIMENTS:
        det = ClassicalCrackDetector(ecfg)
        per, times, preds = [], [], []
        for (s, img), gt in zip(data, gts):
            pred, t = det(img)
            per.append(segmentation_metrics(pred, gt)); times.append(t); preds.append(pred)
        agg = aggregate(per)
        row = {"experiment": ecfg.name, "method": f"classical/{ecfg.detector}", "preprocessing": str(ecfg.preprocessing),
               "augmentation": "n/a", "loss": "n/a", "morphology": str(ecfg.morphology) + (f"+cc>={ecfg.min_component_area}" if ecfg.min_component_area else ""),
               "threshold": "n/a", **{k: agg[k] for k in ["iou", "dice", "precision", "recall", "f1"]},
               "global_iou": agg["global_iou"], "global_dice": agg["global_dice"], "pos_iou": agg["pos_iou"], "pos_dice": agg["pos_dice"],
               "inference_ms": float(np.mean(times) * 1000), "n_test_images": len(per)}
        rows.append(row); preds_by_exp[ecfg.name] = preds
        append_summary(row)
        print(f"{ecfg.name:36s} IoU {agg['iou']:.3f} Dice {agg['dice']:.3f} P {agg['precision']:.3f} R {agg['recall']:.3f} | pos Dice {agg['pos_dice']:.3f} | global Dice {agg['global_dice']:.3f} | {row['inference_ms']:.1f} ms")
    save_json(rows, out / "classical_results.json")
    cols = ["experiment", "preprocessing", "method", "morphology", "iou", "dice", "precision", "recall", "f1", "pos_iou", "pos_dice", "global_dice", "inference_ms"]
    (out / "classical_results.md").write_text(f"# Classical CV comparison (split={a.split}, n={len(samples)})\n\n" + md_table(rows, cols) + "\n")
    bar_comparison([r["experiment"] for r in rows], {"Dice (all)": [r["dice"] for r in rows], "Dice (crack images)": [r["pos_dice"] for r in rows], "Dice (pixel-pooled)": [r["global_dice"] for r in rows]},
                   out / "classical_comparison.png", title="Classical CV methods (mean per-image)", ylabel="score")
    # qualitative: best 4 experiments on 3 positive images
    best = sorted(rows, key=lambda r: -r["dice"])[:4]
    pos_idx = [i for i, g in enumerate(gts) if g.sum() > 0][:3]
    ims, titles = [], []
    for i in pos_idx:
        s, img = data[i]
        ims += [img, gts[i]] + [overlay(img, preds_by_exp[b["experiment"]][i], gts[i]) for b in best]
        titles += [s.stem[:14], "GT"] + [b["experiment"][:22] for b in best]
    panel(ims, titles, out / "classical_qualitative.png", ncols=2 + len(best), suptitle="Classical CV: best methods (TP green / FP red / FN blue)")
    print(f"\nSaved {out}; summary appended to {SUMMARY}")


if __name__ == "__main__":
    main()
