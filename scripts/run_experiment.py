"""Train + evaluate one named experiment end to end.
    python scripts/run_experiment.py --experiment unet_clahe --profile local
    python scripts/run_experiment.py --experiment all --profile kaggle"""
import argparse
from pathlib import Path
from _common import ROOT, resolve_config, add_common_args
from evaluate import evaluate
from train import run_training


def main():
    a = add_common_args(argparse.ArgumentParser()).parse_args()
    names = [p.stem for p in sorted((ROOT / "experiments").glob("*.yaml"))] if a.experiment == "all" else [a.experiment]
    for name in names:
        cfg = resolve_config(a.config, name, a.profile, a.overrides)
        out_dir = run_training(cfg)
        evaluate(str(out_dir / "best.pt"), out_dir=out_dir)


if __name__ == "__main__":
    main()
