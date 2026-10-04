"""Generate notebooks/kaggle_training.ipynb (kept as a script so the notebook stays in sync with the code)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/<your-github-username>/drone-based-crack-inspection.git"  # <- set after creating the repo

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md("""
# UAV crack inspection – U-Net training on Kaggle (GPU)

This notebook runs the deep-learning part of the project on a Kaggle GPU. The classical CV, preprocessing,
characterisation and visualisation code lives in the same repository and is reused here unchanged.

**Setup on Kaggle**
1. *Settings → Accelerator → GPU (T4 / P100)*.
2. *Add data* → add a public concrete crack **segmentation** dataset (images + pixel masks), e.g. the
   *Crack Segmentation Dataset* (`/kaggle/input/crack-segmentation-dataset`, folders `train/images`, `train/masks`, `test/...`).
   Any layout with `images`/`masks` sibling folders is discovered automatically.
3. Set `DATASET_PATH` below and run all cells.
""")
code(f"""
import os, subprocess, sys
REPO = "{REPO}"
if not os.path.exists("/kaggle/working/drone-based-crack-inspection"):
    subprocess.run(["git", "clone", "--depth", "1", REPO, "/kaggle/working/drone-based-crack-inspection"], check=True)
os.chdir("/kaggle/working/drone-based-crack-inspection")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-image", "pyyaml"], check=True)
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
print(os.getcwd())
""")
code("""
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU name      :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")
print("Device in use :", "cuda" if torch.cuda.is_available() else "cpu")
""")
code("""
# ---- configuration -------------------------------------------------------------------
DATASET_PATH = "/kaggle/input/crack-segmentation-dataset"   # <- change to your attached dataset
IMAGE_SIZE   = 448
BATCH_SIZE   = 16
EPOCHS       = 40
EXPERIMENTS  = ["unet_raw", "unet_clahe", "unet_illum", "unet_aug", "unet_clahe_aug",
                "unet_clahe_aug_tversky", "unet_clahe_aug_focal", "unet_clahe_aug_bce"]   # C, D, ablations, E (+F via refinement)
OVERRIDES = [f"dataset.path={DATASET_PATH}", f"dataset.image_size={IMAGE_SIZE}", f"dataset.batch_size={BATCH_SIZE}", f"training.epochs={EPOCHS}"]
""")
code("""
# ---- dataset inspection ---------------------------------------------------------------
from _common import resolve_config
from src.data.dataset import discover_pairs, dataset_summary, split_samples
cfg = resolve_config(profile="kaggle", overrides=OVERRIDES)
samples = discover_pairs(cfg["dataset"]["path"])
print(dataset_summary(samples))
print({k: len(v) for k, v in split_samples(samples, cfg["dataset"]["split"], cfg["seed"]).items()})
""")
code("""
# ---- train + evaluate every experiment (each run saves config, logs, checkpoint, metrics, plots) ----
from run_experiment import resolve_config
from train import run_training
from evaluate import evaluate
for name in EXPERIMENTS:
    cfg = resolve_config(experiment=name, profile="kaggle", overrides=OVERRIDES)
    out_dir = run_training(cfg)
    evaluate(str(out_dir / "best.pt"), out_dir=out_dir)
""")
code("""
# ---- summary table (Experiments C-F + ablations); classical rows (A/B) come from scripts/classical_cv.py ----
import pandas as pd
df = pd.read_csv("/kaggle/working/results/experiments_summary.csv") if os.path.exists("/kaggle/working/results/experiments_summary.csv") else pd.read_csv("results/experiments_summary.csv")
df.sort_values("dice", ascending=False)
""")
code("""
# ---- photometric robustness of the best model + classical baseline on the test split ----
import glob
best = sorted(glob.glob("/kaggle/working/results/*unet_clahe_aug/best.pt"))[-1]
!python scripts/photometric_experiment.py --profile kaggle --checkpoint {best} --set dataset.path={DATASET_PATH}
""")
code("""
# ---- classical CV grid on the same test split (Experiments A/B) for a like-for-like comparison ----
!python scripts/classical_cv.py --profile kaggle --split test --set dataset.path={DATASET_PATH}
""")
md("""
Outputs are under `/kaggle/working/results/<id>_<experiment>/` (`config.yaml`, `metrics.json`, `training_log.csv`,
`plots/`, `qualitative_examples/`, `error_analysis/`, `predictions/`) and the best checkpoints under
`/kaggle/working/checkpoints/`. Download them and copy into the repository's `results/` folder to update the README tables.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "kaggle_training.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
