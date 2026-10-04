"""Shared bootstrap for CLI scripts."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import deep_update, load_config  # noqa: E402


def resolve_config(config="configs/config.yaml", experiment=None, profile=None, overrides=None) -> dict:
    cfg = load_config(ROOT / config)
    if experiment:
        exp_path = Path(experiment)
        if not exp_path.exists():
            exp_path = ROOT / "experiments" / f"{experiment}.yaml"
        cfg = deep_update(cfg, load_config(exp_path))
    if profile:
        cfg = deep_update(cfg, load_config(ROOT / "configs" / f"profile_{profile}.yaml"))
    for kv in overrides or []:
        key, val = kv.split("=", 1)
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        try:
            import yaml
            val = yaml.safe_load(val)
        except Exception:
            pass
        d[parts[-1]] = val
    return cfg


def add_common_args(parser):
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--experiment", default=None, help="experiment name in experiments/ or a yaml path")
    parser.add_argument("--profile", default=None, choices=[None, "local", "kaggle"])
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="override key.path=value")
    return parser


def md_table(rows: list[dict], cols: list[str], floatfmt="{:.4f}") -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(floatfmt.format(v) if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
