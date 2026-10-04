"""Evaluate a checkpoint on the held-out test split.

Reports: threshold sweep, raw vs morphologically refined metrics, inference time, confusion matrix,
qualitative examples, error analysis and crack-measurement accuracy (pred-mask vs GT-mask measurements).
    python scripts/evaluate.py --checkpoint results/001_unet_raw/best.pt
"""
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from _common import ROOT, md_table
from classical_cv import append_summary
from src.characterization.crack_metrics import characterize, measurement_errors, skeleton_of
from src.data.dataset import discover_pairs, load_image, load_mask, split_samples
from src.evaluation.error_analysis import run_error_analysis
from src.evaluation.metrics import aggregate, segmentation_metrics
from src.postprocessing.morphology import probability_to_mask, refine_mask
from src.preprocessing.ops import build_pipeline
from src.training.trainer import load_checkpoint
from src.utils import ensure_dir, get_device, save_json
from src.visualization.plots import (characterization_figure, confusion_matrix_plot, overlay, panel, pr_curve,
                                     threshold_curve)

KEYS = ["iou", "dice", "precision", "recall", "f1", "accuracy", "specificity", "pos_iou", "pos_dice", "global_iou", "global_dice"]


@torch.no_grad()
def predict_probs(model, images, pre, size, device):
    probs, times = [], []
    for img in images:
        x = pre(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA) if img.shape[0] != size else img)
        t = torch.from_numpy(x.transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(device)
        t0 = time.perf_counter()
        p = torch.sigmoid(model(t))[0, 0].cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        probs.append(p)
    return probs, float(np.mean(times[1:]) * 1000 if len(times) > 1 else times[0] * 1000)


def evaluate(checkpoint: str, data: str | None = None, out_dir: str | None = None, split: str = "test") -> dict:
    device = get_device()
    model, cfg, ck = load_checkpoint(checkpoint, device)
    out_dir = Path(out_dir) if out_dir else Path(checkpoint).parent
    out_dir = ensure_dir(out_dir)
    plots, qual, pred_dir = ensure_dir(out_dir / "plots"), ensure_dir(out_dir / "qualitative_examples"), ensure_dir(out_dir / "predictions")
    samples = discover_pairs(data or cfg["dataset"]["path"])
    splits = split_samples(samples, cfg["dataset"]["split"], cfg["seed"])
    test = splits[split]
    size = cfg["dataset"]["image_size"]
    pre = build_pipeline(cfg["preprocessing"]["pipeline"])
    images = [load_image(s.image_path, size) for s in test]
    gts = [load_mask(s.mask_path, (size, size), size) for s in test]
    names = [s.stem for s in test]
    probs, infer_ms = predict_probs(model, images, pre, size, device)
    ths = cfg.get("evaluation", {}).get("thresholds", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # ---- threshold selection on the VALIDATION split (pixel-pooled Dice); test split is never used for selection
    val = splits["val"]
    val_probs, _ = predict_probs(model, [load_image(s.image_path, size) for s in val], pre, size, device)
    val_gts = [load_mask(s.mask_path, (size, size), size) for s in val]
    val_sweep = {t: aggregate([segmentation_metrics(probability_to_mask(p, t), g) for p, g in zip(val_probs, val_gts)])["global_dice"] for t in ths}
    best_t = max(ths, key=lambda t: val_sweep[t])
    # ---- threshold sweep on test (reported for analysis only)
    sweep = {}
    for t in ths:
        sweep[t] = aggregate([segmentation_metrics(probability_to_mask(p, t), g) for p, g in zip(probs, gts)])
    threshold_curve(ths, {k: [sweep[t][k] for t in ths] for k in ["dice", "iou", "precision", "recall"]}, plots / "threshold_sweep.png")
    pr_curve([sweep[t]["recall"] for t in ths], [sweep[t]["precision"] for t in ths], plots / "pr_curve.png")
    # ---- raw vs refined at 0.5 and at best threshold
    post_cfg = cfg.get("postprocessing", {})
    results = {}
    for label, t in [("t0.5", 0.5), (f"t{best_t}", best_t)]:
        raw = [probability_to_mask(p, t) for p in probs]
        t0 = time.perf_counter()
        ref = [refine_mask(m, post_cfg) for m in raw]
        ref_ms = (time.perf_counter() - t0) / len(raw) * 1000
        results[label] = {"threshold": t, "raw": aggregate([segmentation_metrics(p, g) for p, g in zip(raw, gts)]),
                          "refined": aggregate([segmentation_metrics(p, g) for p, g in zip(ref, gts)]), "refine_ms": ref_ms}
    final_raw = [probability_to_mask(p, best_t) for p in probs]
    final_ref = [refine_mask(m, post_cfg) for m in final_raw]
    for n, m in zip(names, final_ref):
        cv2.imwrite(str(pred_dir / f"{n}.png"), m * 255)
    confusion_matrix_plot(results[f"t{best_t}"]["refined"], plots / "confusion_matrix.png")
    # ---- qualitative: raw vs refined
    nq = cfg.get("evaluation", {}).get("n_qualitative", 6)
    pos = [i for i, g in enumerate(gts) if g.sum() > 0][:nq]
    ims, titles = [], []
    for i in pos:
        ims += [images[i], probs[i], gts[i], overlay(images[i], final_raw[i], gts[i]), overlay(images[i], final_ref[i], gts[i])]
        titles += [names[i][:16], "probability", "GT", f"raw (IoU {segmentation_metrics(final_raw[i], gts[i])['iou']:.2f})",
                   f"refined (IoU {segmentation_metrics(final_ref[i], gts[i])['iou']:.2f})"]
    panel(ims, titles, qual / "raw_vs_refined.png", ncols=5, suptitle=f"U-Net output before/after morphological refinement (threshold {best_t})")
    # ---- error analysis
    err = run_error_analysis(images, final_ref, gts, names, out_dir / "error_analysis")
    # ---- crack characterisation evaluation (positives only)
    pred_props, gt_props = [], []
    for i, (p, g) in enumerate(zip(final_ref, gts)):
        if g.sum() == 0:
            continue
        pp, gp = characterize(p), characterize(g)
        pred_props.append(pp); gt_props.append(gp)
        if len(pred_props) <= 4:
            labels = cv2.connectedComponents(p, connectivity=8)[1]
            characterization_figure(images[i], p, g, skeleton_of(p), labels, pp, qual / f"characterization_{names[i]}.png", title=names[i])
    meas = measurement_errors(pred_props, gt_props) if pred_props else {}
    # ---- summary
    sel = results[f"t{best_t}"]
    metrics = {"checkpoint": str(checkpoint), "experiment": cfg["experiment_name"], "split": split, "n_images": len(test),
               "best_threshold": best_t, "val_threshold_sweep_global_dice": {str(t): v for t, v in val_sweep.items()}, "threshold_sweep": {str(t): {k: sweep[t][k] for k in KEYS} for t in ths},
               "results": results, "inference_ms_per_image": infer_ms, "device": str(device),
               "error_analysis": {k: v for k, v in err.items() if k != "per_image"}, "measurement_errors": meas,
               "train_summary": {"best_epoch": ck.get("epoch"), "val_metrics": ck.get("val_metrics")}}
    save_json(metrics, out_dir / "metrics.json")
    save_json(err["per_image"], out_dir / "error_analysis" / "per_image.json")
    aug_on = cfg.get("augmentation", {}).get("enabled", False)
    loss = cfg.get("loss", "bce_dice"); loss = loss if isinstance(loss, str) else loss.get("name")
    base = {"experiment": cfg["experiment_name"], "method": f"{cfg['model']['name']}", "preprocessing": str(cfg["preprocessing"]["pipeline"]),
            "augmentation": "on" if aug_on else "off", "loss": loss, "threshold": best_t, "inference_ms": infer_ms, "n_test_images": len(test)}
    for morph, key in [("none", "raw"), ("refined", "refined")]:
        append_summary({**base, "morphology": morph, **{k: sel[key][k] for k in ["iou", "dice", "precision", "recall", "f1", "pos_iou", "pos_dice", "global_iou", "global_dice"]}})
    rows = [{"setting": f"{lab} / {k}", **{m: v[k][m] for m in KEYS}} for lab, v in results.items() for k in ["raw", "refined"]]
    md = (f"# Evaluation: {cfg['experiment_name']} ({split}, n={len(test)})\n\nthreshold {best_t} (selected on the validation split by pixel-pooled Dice); "
          f"inference {infer_ms:.1f} ms/image on {device}\n\n" + md_table(rows, ["setting"] + KEYS) +
          "\n\n## Error analysis counts\n\n" + md_table([{"category": k, "count": v} for k, v in err["counts"].items()], ["category", "count"]) +
          "\n\n## Crack measurement errors (pred mask vs GT mask, pixels)\n\n" + md_table([{"metric": k, "value": v} for k, v in meas.items()], ["metric", "value"]) + "\n")
    (out_dir / "metrics.md").write_text(md)
    print(md)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    a = ap.parse_args()
    evaluate(a.checkpoint, a.data, a.out, a.split)


if __name__ == "__main__":
    main()
