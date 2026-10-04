"""Robustness to photometric variation: evaluate a classical detector and (optionally) a trained
U-Net checkpoint on the test split under simulated illumination changes. Masks are unchanged."""
import argparse
import numpy as np
from _common import ROOT, resolve_config, add_common_args, md_table
from src.classical_cv.pipeline import EXPERIMENTS, ClassicalCrackDetector
from src.data.dataset import discover_pairs, load_image, load_mask, split_samples
from src.evaluation.metrics import aggregate, segmentation_metrics
from src.preprocessing.perturbations import PERTURBATIONS
from src.utils import ensure_dir, save_json
from src.visualization.plots import bar_comparison

KEYS = ["iou", "dice", "precision", "recall", "f1"]


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--data", default=None)
    ap.add_argument("--checkpoint", default=None, help="optional trained model checkpoint")
    ap.add_argument("--classical", default="E6_combined_adaptive_morph_cc")
    ap.add_argument("--split", default="test", choices=["all", "test"])
    a = ap.parse_args()
    cfg = resolve_config(a.config, a.experiment, a.profile, a.overrides)
    samples = discover_pairs(a.data or cfg["dataset"]["path"])
    if a.split == "test":
        samples = split_samples(samples, cfg["dataset"]["split"], cfg["seed"])["test"]
    out = ensure_dir(ROOT / "results" / "photometric_robustness")
    methods = {}
    for e in EXPERIMENTS:
        if e.name == a.classical:
            det = ClassicalCrackDetector(e)
            methods[f"classical:{e.name}"] = lambda img, _d=det: _d(img)[0]
    if a.checkpoint:
        from predict import make_predictor
        methods["unet:" + a.checkpoint.split("/")[-2]] = make_predictor(a.checkpoint, cfg)
    data = [(load_image(s.image_path), load_mask(s.mask_path, (448, 448))) for s in samples]
    rows = []
    for mname, fn in methods.items():
        for pname, pert in PERTURBATIONS.items():
            per = [segmentation_metrics(_resize_like(fn(pert(img)), gt), gt) for img, gt in data]
            agg = aggregate(per)
            rows.append({"method": mname, "perturbation": pname, **{k: agg[k] for k in KEYS}})
            print(f"{mname:40s} {pname:20s} IoU {agg['iou']:.3f} Dice {agg['dice']:.3f} P {agg['precision']:.3f} R {agg['recall']:.3f}")
    save_json(rows, out / "photometric_results.json")
    md = f"# Robustness to photometric variation (split={a.split}, n={len(data)})\n\n" + md_table(rows, ["method", "perturbation"] + KEYS)
    (out / "photometric_results.md").write_text(md + "\n")
    perts = list(PERTURBATIONS)
    series = {m: [next(r["dice"] for r in rows if r["method"] == m and r["perturbation"] == p) for p in perts] for m in methods}
    bar_comparison(perts, series, out / "photometric_dice.png", title="Dice under photometric perturbation", ylabel="Dice")
    print(f"\nSaved {out}")


def _resize_like(pred, gt):
    import cv2
    if pred.shape != gt.shape:
        pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    return pred


if __name__ == "__main__":
    main()
