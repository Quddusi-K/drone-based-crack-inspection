"""Predict on a single image: probability map, binary mask, refined mask, overlay, measurements.
    python scripts/predict.py --input image.jpg --checkpoint checkpoints/unet_clahe_aug_best.pt [--threshold 0.5]"""
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from _common import ROOT
from src.characterization.crack_metrics import characterize, skeleton_of
from src.data.dataset import load_image
from src.postprocessing.morphology import probability_to_mask, refine_mask
from src.preprocessing.ops import build_pipeline
from src.training.trainer import load_checkpoint
from src.utils import ensure_dir, get_device, save_json
from src.visualization.plots import overlay, panel


def make_predictor(checkpoint: str, cfg_override: dict | None = None, threshold: float = 0.5, refine: bool = True):
    device = get_device(verbose=False)
    model, cfg, _ = load_checkpoint(checkpoint, device)
    size = cfg["dataset"]["image_size"]
    pre = build_pipeline(cfg["preprocessing"]["pipeline"])
    post = cfg.get("postprocessing", {})

    @torch.no_grad()
    def prob_fn(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        x = pre(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))
        t = torch.from_numpy(x.transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(device)
        p = torch.sigmoid(model(t))[0, 0].cpu().numpy()
        return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)

    def mask_fn(img: np.ndarray) -> np.ndarray:
        m = probability_to_mask(prob_fn(img), threshold)
        return refine_mask(m, post) if refine else m

    mask_fn.prob = prob_fn
    mask_fn.cfg = cfg
    return mask_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = ensure_dir(a.out or ROOT / "results" / "predictions" / Path(a.input).stem)
    predictor = make_predictor(a.checkpoint, threshold=a.threshold)
    img = load_image(a.input)
    prob = predictor.prob(img)
    raw = probability_to_mask(prob, a.threshold)
    ref = refine_mask(raw, predictor.cfg.get("postprocessing", {}))
    props = characterize(ref)
    cv2.imwrite(str(out / "input.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "probability.png"), (prob * 255).astype(np.uint8))
    cv2.imwrite(str(out / "mask_raw.png"), raw * 255)
    cv2.imwrite(str(out / "mask_refined.png"), ref * 255)
    cv2.imwrite(str(out / "overlay.png"), cv2.cvtColor(overlay(img, ref), cv2.COLOR_RGB2BGR))
    save_json(props, out / "measurements.json")
    skel_rgb = img.copy(); skel_rgb[skeleton_of(ref) > 0] = (255, 255, 0)
    panel([img, prob, raw, ref, overlay(img, ref), skel_rgb], ["input", "probability", "binary mask", "refined mask", "overlay", "skeleton"],
          out / "summary.png", suptitle=f"length {props['crack_length_pixels']:.0f}px, mean width {props['mean_width_pixels']:.1f}px, "
          f"orientation {props['orientation_degrees']:.0f} deg, {props['num_components']} components")
    print(props)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
