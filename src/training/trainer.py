"""Training loop: AdamW/Adam, LR scheduler, early stopping, checkpointing, CSV/JSON logging."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import aggregate, segmentation_metrics
from src.models.unet import build_model
from src.training.losses import build_loss
from src.utils import ensure_dir, get_device, save_json


def build_optimizer(params, cfg: dict):
    name = cfg.get("optimizer", "adamw").lower()
    lr = float(cfg.get("learning_rate", 1e-4))
    wd = float(cfg.get("weight_decay", 1e-4))
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


def build_scheduler(opt, cfg: dict):
    name = cfg.get("scheduler", "plateau")
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.get("epochs", 50)), False
    if name == "none":
        return None, False
    return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=cfg.get("plateau_patience", 3)), True


@torch.no_grad()
def evaluate_loader(model, loader, device, loss_fn=None, threshold=0.5):
    model.eval()
    per_image, losses = [], []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if loss_fn is not None:
            losses.append(loss_fn(logits, y).item())
        pred = (torch.sigmoid(logits) >= threshold).cpu().numpy().astype(np.uint8)
        gt = y.cpu().numpy().astype(np.uint8)
        for p, g in zip(pred, gt):
            per_image.append(segmentation_metrics(p[0], g[0]))
    agg = aggregate(per_image)
    agg["loss"] = float(np.mean(losses)) if losses else float("nan")
    return agg


def train(cfg: dict, train_ds, val_ds, out_dir: str | Path) -> dict:
    out_dir = ensure_dir(out_dir)
    tcfg = cfg["training"]
    device = get_device()
    torch.manual_seed(cfg.get("seed", 42))
    model = build_model(cfg["model"].get("name", "unet"), **cfg["model"].get("params", {})).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['model'].get('name')} ({n_params/1e6:.2f} M params)")
    bs = cfg["dataset"].get("batch_size", 8)
    nw = cfg["dataset"].get("num_workers", 2)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=len(train_ds) > bs, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    loss_fn = build_loss(cfg.get("loss", "bce_dice"))
    opt = build_optimizer(model.parameters(), tcfg)
    sched, sched_on_metric = build_scheduler(opt, tcfg)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and tcfg.get("amp", True))
    monitor = tcfg.get("monitor", "global_dice")
    patience = tcfg.get("early_stopping_patience", 10)
    best, best_epoch, bad_epochs = -1.0, 0, 0
    history = []
    ckpt_path = out_dir / "best.pt"
    log_path = out_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "val_precision", "val_recall", "val_global_dice", "val_global_iou", "lr", "epoch_time_s"])
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        t0 = time.time()
        tl = []
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl.append(loss.item())
        val = evaluate_loader(model, val_loader, device, loss_fn)
        if sched is not None:
            sched.step(val[monitor]) if sched_on_metric else sched.step()
        lr = opt.param_groups[0]["lr"]
        row = {"epoch": epoch, "train_loss": float(np.mean(tl)), "val_loss": val["loss"], "val_dice": val["dice"],
               "val_iou": val["iou"], "val_precision": val["precision"], "val_recall": val["recall"],
               "val_global_dice": val["global_dice"], "val_global_iou": val["global_iou"], "lr": lr,
               "epoch_time_s": time.time() - t0}
        history.append(row)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(list(row.values()))
        improved = val[monitor] > best
        print(f"epoch {epoch:3d} | train {row['train_loss']:.4f} | val {row['val_loss']:.4f} | dice {val['dice']:.4f} | iou {val['iou']:.4f} | pooled dice {val['global_dice']:.4f} | "
              f"P {val['precision']:.3f} R {val['recall']:.3f} | lr {lr:.1e} | {row['epoch_time_s']:.0f}s{'  *' if improved else ''}")
        if improved:
            best, best_epoch, bad_epochs = val[monitor], epoch, 0
            torch.save({"model_state": model.state_dict(), "config": cfg, "epoch": epoch, "val_metrics": val}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (best {monitor}={best:.4f} @ epoch {best_epoch})")
                break
    save_json(history, out_dir / "history.json")
    return {"best_val_" + monitor: best, "best_epoch": best_epoch, "history": history, "checkpoint": str(ckpt_path),
            "n_params": n_params, "device": str(device)}


def load_checkpoint(path: str | Path, device=None):
    device = device or get_device(verbose=False)
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg["model"].get("name", "unet"), **cfg["model"].get("params", {})).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, cfg, ck
