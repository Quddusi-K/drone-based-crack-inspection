"""Collect results/experiments_summary.csv into markdown tables + ablation plot and refresh the README block."""
import re
from pathlib import Path
import pandas as pd
from _common import ROOT
from src.visualization.plots import bar_comparison

SUMMARY = ROOT / "results" / "experiments_summary.csv"
README = ROOT / "README.md"


def main():
    df = pd.read_csv(SUMMARY, keep_default_na=False)
    for c in ["iou", "dice", "precision", "recall", "f1", "pos_iou", "pos_dice", "global_iou", "global_dice", "inference_ms"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    unet = df[~df["experiment"].str.startswith("E")].copy()
    classical = df[df["experiment"].str.startswith("E")].copy()
    cols = ["experiment", "preprocessing", "augmentation", "loss", "morphology", "threshold", "iou", "dice", "precision", "recall", "f1",
            "pos_dice", "global_iou", "global_dice", "inference_ms"]
    fmt = {c: "{:.3f}" for c in ["iou", "dice", "precision", "recall", "f1", "pos_dice", "global_iou", "global_dice"]}
    fmt["inference_ms"] = "{:.1f}"

    def table(d):
        d = d[cols].copy()
        for c, f in fmt.items():
            d[c] = d[c].map(lambda v: f.format(v) if pd.notna(v) else "")
        d = d.rename(columns={"pos_dice": "crack-only Dice", "global_iou": "pooled IoU", "global_dice": "pooled Dice", "inference_ms": "ms/img"})
        return d.to_markdown(index=False)

    parts = []
    if len(unet):
        parts.append("**U-Net experiments (C–F + ablations), test split, mean per-image metrics unless stated.** "
                     "`morphology=none` is the raw network output (Experiments C–E); `refined` adds morphological "
                     "post-processing (Experiment F). Threshold selected on the validation split by pixel-pooled Dice. Local laptop profile: 256 px, 16-channel U-Net, 12 epochs, 159/20/20 split.\n")
        parts.append(table(unet))
        best_cls = classical.sort_values("dice", ascending=False).head(3) if len(classical) else None
        if best_cls is not None:
            parts.append("\n**Best classical methods (A/B) for reference, same metric definitions (evaluated on all 199 samples):**\n")
            parts.append(table(best_cls))
        ref = unet[unet["morphology"] == "refined"]
        raw = unet[unet["morphology"] == "none"]
        labels = list(raw["experiment"])
        series = {"raw output": list(raw["global_dice"]),
                  "+ morphology": [float(ref[ref["experiment"] == e]["global_dice"].iloc[0]) if (ref["experiment"] == e).any() else 0 for e in labels]}
        bar_comparison(labels, series, ROOT / "results" / "ablation_dice.png", title="Ablation: pixel-pooled test Dice", ylabel="Dice")
        parts.append("\n![ablation](results/ablation_dice.png)\n")
    block = "\n".join(parts)
    (ROOT / "results" / "experiments_summary.md").write_text(block + "\n")
    txt = README.read_text()
    txt = re.sub(r"<!-- DL_RESULTS_START -->.*?<!-- DL_RESULTS_END -->", f"<!-- DL_RESULTS_START -->\n{block}\n<!-- DL_RESULTS_END -->", txt, flags=re.S)
    README.write_text(txt)
    print(block)


if __name__ == "__main__":
    main()
