"""Preprocessing experiment: does preprocessing improve crack/background separability?
Compares raw / normalized / contrast / illumination / retinex / combined presets and saves
before-after figures plus a separability table."""
import argparse
import time
import numpy as np
from _common import ROOT, resolve_config, add_common_args, md_table
from src.data.dataset import discover_pairs, load_image, load_mask
from src.preprocessing.ops import PRESETS, build_pipeline, crack_separability
from src.preprocessing.perturbations import PERTURBATIONS
from src.utils import ensure_dir, save_json
from src.visualization.plots import panel, bar_comparison


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--data", default=None)
    ap.add_argument("--max-images", type=int, default=100)
    a = ap.parse_args()
    cfg = resolve_config(a.config, a.experiment, a.profile, a.overrides)
    samples = [s for s in discover_pairs(a.data or cfg["dataset"]["path"]) if s.mask_path is not None]
    samples = [s for s in samples if load_mask(s.mask_path).sum() > 0][: a.max_images]
    out = ensure_dir(ROOT / "results" / "preprocessing")
    pipes = {name: build_pipeline(name) for name in PRESETS}
    rows = []
    for name, fn in pipes.items():
        cs, fs, ts = [], [], []
        for s in samples:
            img = load_image(s.image_path); m = load_mask(s.mask_path)
            t0 = time.perf_counter(); p = fn(img); ts.append(time.perf_counter() - t0)
            r = crack_separability(p, m); cs.append(r["contrast"]); fs.append(r["fisher"])
        rows.append({"preset": name, "steps": str(PRESETS[name]), "contrast": float(np.nanmean(cs)),
                     "fisher_ratio": float(np.nanmean(fs)), "time_ms": float(np.mean(ts) * 1000)})
    # also: separability under photometric degradation, with and without preprocessing
    deg_rows = []
    for pert_name, pert in PERTURBATIONS.items():
        for name in ["raw", "contrast", "illumination", "combined"]:
            fs = []
            for s in samples[:40]:
                img = pert(load_image(s.image_path)); m = load_mask(s.mask_path)
                fs.append(crack_separability(pipes[name](img), m)["fisher"])
            deg_rows.append({"perturbation": pert_name, "preset": name, "fisher_ratio": float(np.nanmean(fs))})
    save_json({"separability": rows, "under_perturbation": deg_rows, "n_images": len(samples)}, out / "separability.json")
    table = md_table(rows, ["preset", "steps", "contrast", "fisher_ratio", "time_ms"])
    piv = {}
    for r in deg_rows:
        piv.setdefault(r["perturbation"], {"perturbation": r["perturbation"]})[r["preset"]] = r["fisher_ratio"]
    table2 = md_table(list(piv.values()), ["perturbation", "raw", "contrast", "illumination", "combined"])
    (out / "separability.md").write_text(f"# Crack/background separability (n={len(samples)})\n\n{table}\n\n"
                                         f"## Fisher ratio under photometric perturbation (first 40 images)\n\n{table2}\n")
    print(table); print(); print(table2)
    bar_comparison([r["preset"] for r in rows], {"Fisher ratio": [r["fisher_ratio"] for r in rows]},
                   out / "separability_fisher.png", title="Crack/background separability by preprocessing preset", ylabel="Fisher ratio")
    for s in samples[:3]:
        img = load_image(s.image_path)
        ims = [fn(img) for fn in pipes.values()]
        panel(ims, list(pipes), out / f"before_after_{s.stem}.png", suptitle=f"Preprocessing presets: {s.stem}")
    s = samples[0]; img = load_image(s.image_path)
    ims, titles = [], []
    for pn, pert in list(PERTURBATIONS.items()):
        ims.append(pert(img)); titles.append(pn)
    panel(ims, titles, out / "photometric_perturbations.png", ncols=6, suptitle="Simulated photometric variations (mask unchanged)")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
