"""Train a segmentation model.  python scripts/train.py --config configs/config.yaml [--experiment unet_clahe] [--profile local|kaggle]"""
import argparse
from pathlib import Path
from _common import ROOT, resolve_config, add_common_args
from src.data.dataset import CrackSegDataset, discover_pairs, split_samples
from src.preprocessing.ops import build_pipeline
from src.training.augmentation import build_augmenter
from src.training.trainer import train
from src.utils import ensure_dir, next_experiment_dir, save_config, save_json, seed_everything
from src.visualization.plots import training_curves


def run_training(cfg: dict, out_dir: Path | None = None) -> Path:
    seed_everything(cfg["seed"])
    results_root = Path(cfg.get("results_dir", "results"))
    results_root = results_root if results_root.is_absolute() else ROOT / results_root
    out_dir = out_dir or next_experiment_dir(results_root, cfg["experiment_name"])
    save_config(cfg, out_dir / "config.yaml")
    samples = discover_pairs(cfg["dataset"]["path"])
    splits = split_samples(samples, cfg["dataset"]["split"], cfg["seed"])
    save_json({k: [s.stem for s in v] for k, v in splits.items()}, out_dir / "splits.json")
    pre = build_pipeline(cfg["preprocessing"]["pipeline"])
    aug = build_augmenter(cfg.get("augmentation"), cfg["seed"])
    size = cfg["dataset"]["image_size"]
    train_ds = CrackSegDataset(splits["train"], size, pre, aug)
    val_ds = CrackSegDataset(splits["val"], size, pre, None)
    print(f"Experiment {cfg['experiment_name']} -> {out_dir}")
    print(f"train/val/test = {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} | preprocessing={cfg['preprocessing']['pipeline']} "
          f"| augmentation={'on' if aug else 'off'} | loss={cfg.get('loss')}")
    res = train(cfg, train_ds, val_ds, out_dir)
    training_curves(res["history"], out_dir / "plots" / "training_curves.png")
    ck_dir = Path(cfg.get("checkpoints_dir", "checkpoints"))
    ck_dir = ensure_dir(ck_dir if ck_dir.is_absolute() else ROOT / ck_dir)
    import shutil
    shutil.copy(res["checkpoint"], ck_dir / f"{cfg['experiment_name']}_best.pt")
    save_json({k: v for k, v in res.items() if k != "history"}, out_dir / "train_summary.json")
    return out_dir


def main():
    a = add_common_args(argparse.ArgumentParser()).parse_args()
    run_training(resolve_config(a.config, a.experiment, a.profile, a.overrides))


if __name__ == "__main__":
    main()
