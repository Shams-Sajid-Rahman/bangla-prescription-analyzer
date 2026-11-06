"""Train the handwritten medicine-name classifier.

Example (Colab):
    python -m src.train --data_root /content/data --out_dir /content/drive/MyDrive/rx_runs/effb0 \
        --model efficientnet_b0 --epochs 40 --img_size 224 --batch_size 64
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import LabelMap, PrescriptionWords, build_transforms, load_all
from .model import build_model, save_checkpoint


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True, help="Folder that contains Training/Validation/Testing")
    p.add_argument("--out_dir", default="runs/exp1")
    p.add_argument("--model", default="efficientnet_b0",
                   help="Any timm model: efficientnet_b0, convnext_tiny, resnet50, mobilenetv3_large_100 ...")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--patience", type=int, default=8, help="Early-stopping patience (epochs)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--max_steps_per_epoch", type=int, default=0, help="Debug only: limit batches")
    return p.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None, max_steps=0):
    model.eval()
    correct1 = correct5 = n = 0
    loss_sum = 0.0
    for step, (x, y) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
        if criterion is not None:
            loss_sum += criterion(logits.float(), y).item() * y.size(0)
        k = min(5, logits.size(1))
        topk = logits.topk(k, dim=1).indices
        correct1 += (topk[:, 0] == y).sum().item()
        correct5 += (topk == y[:, None]).any(dim=1).sum().item()
        n += y.size(0)
    return {"loss": loss_sum / max(n, 1), "top1": correct1 / max(n, 1), "top5": correct5 / max(n, 1)}


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    frames = load_all(args.data_root)
    for s, df in frames.items():
        print(f"{s:5s}: {len(df):5d} images, {df['medicine'].nunique()} classes")
    label_map = LabelMap.from_frames(frames)
    label_map.save(out / "label_map.json")
    num_classes = len(label_map.classes)

    train_ds = PrescriptionWords(frames["train"], label_map, build_transforms(args.img_size, True))
    val_ds = PrescriptionWords(frames["val"], label_map, build_transforms(args.img_size, False))
    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers,
                          pin_memory=pin, drop_last=len(train_ds) > args.batch_size)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin)

    model = build_model(args.model, num_classes, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_dl) if not args.max_steps_per_epoch else min(len(train_dl), args.max_steps_per_epoch)
    total, warm = args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch

    def lr_lambda(step):  # linear warm-up then cosine decay
        if step < warm:
            return (step + 1) / max(warm, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    cfg = vars(args) | {"num_classes": num_classes}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    history, best, bad_epochs = [], -1.0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, run_correct, seen = time.time(), 0.0, 0, 0
        for step, (x, y) in enumerate(train_dl):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            run_loss += loss.item() * y.size(0)
            run_correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)

        val = evaluate(model, val_dl, device, criterion, args.max_steps_per_epoch)
        row = {"epoch": epoch, "train_loss": run_loss / seen, "train_acc": run_correct / seen,
               "val_loss": val["loss"], "val_top1": val["top1"], "val_top5": val["top5"],
               "lr": optimizer.param_groups[0]["lr"], "sec": round(time.time() - t0, 1)}
        history.append(row)
        print(f"ep {epoch:3d} | train loss {row['train_loss']:.3f} acc {row['train_acc']:.3f} | "
              f"val loss {row['val_loss']:.3f} top1 {row['val_top1']:.3f} top5 {row['val_top5']:.3f} | "
              f"{row['sec']}s")

        if val["top1"] > best:
            best, bad_epochs = val["top1"], 0
            save_checkpoint(out / "best.pt", model, cfg, row)
            print(f"  ✓ new best val top1 = {best:.4f} -> saved {out / 'best.pt'}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs).")
                break
        (out / "history.json").write_text(json.dumps(history, indent=2))

    save_checkpoint(out / "last.pt", model, cfg, history[-1])
    print(f"Done. Best val top1 = {best:.4f}. Artifacts in {out}")
    return best


if __name__ == "__main__":
    main()
