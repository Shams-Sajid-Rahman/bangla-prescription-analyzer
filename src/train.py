"""Train the handwritten medicine-name classifier."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import LabelMap, PrescriptionWords, build_transforms, load_all
from .model import build_model, save_checkpoint


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", default="runs/exp1")
    p.add_argument("--model", default="efficientnet_b0")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--input_hw", type=int, nargs=2, default=None, metavar=("H", "W"))
    p.add_argument("--aug", choices=["basic", "strong"], default="basic")
    p.add_argument("--mixup", action="store_true")
    p.add_argument("--ema", type=float, default=0.0)
    p.add_argument("--select", choices=["acc", "loss"], default="acc")
    p.add_argument("--trainval", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--max_steps_per_epoch", type=int, default=0)
    return p.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class EMA:
    def __init__(self, model, decay: float):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])


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
        topk = logits.topk(min(5, logits.size(1)), dim=1).indices
        correct1 += (topk[:, 0] == y).sum().item()
        correct5 += (topk == y[:, None]).any(dim=1).sum().item()
        n += y.size(0)
    return {"loss": loss_sum / max(n, 1), "top1": correct1 / max(n, 1), "top5": correct5 / max(n, 1)}


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    frames = load_all(args.data_root)
    for s, df in frames.items():
        print(f"{s:5s}: {len(df):5d} images, {df['medicine'].nunique()} classes")
    label_map = LabelMap.from_frames(frames)
    label_map.save(out / "label_map.json")
    num_classes = len(label_map.classes)

    hw = tuple(args.input_hw) if args.input_hw else None
    train_df = pd.concat([frames["train"], frames["val"]]) if args.trainval else frames["train"]
    train_ds = PrescriptionWords(train_df, label_map,
                                 build_transforms(args.img_size, True, args.aug == "strong", hw))
    val_ds = PrescriptionWords(frames["val"], label_map, build_transforms(args.img_size, False, input_hw=hw))
    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers,
                          pin_memory=pin, drop_last=True)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin)

    model = build_model(args.model, num_classes, pretrained=not args.no_pretrained).to(device)
    ema = EMA(model, args.ema) if args.ema > 0 else None

    eval_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    mixup_fn = None
    if args.mixup:
        from timm.data import Mixup
        from timm.loss import SoftTargetCrossEntropy
        mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5, switch_prob=0.5,
                         label_smoothing=args.label_smoothing, num_classes=num_classes)
        train_criterion = SoftTargetCrossEntropy()
    else:
        train_criterion = eval_criterion

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = len(train_dl) if not args.max_steps_per_epoch else min(len(train_dl), args.max_steps_per_epoch)
    total, warm = args.epochs * steps, args.warmup_epochs * steps

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(warm, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    cfg = vars(args) | {"num_classes": num_classes}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    history, best, bad = [], None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, run_correct, seen = time.time(), 0.0, 0, 0
        for step, (x, y) in enumerate(train_dl):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            target = y
            if mixup_fn is not None:
                x, target = mixup_fn(x, y)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = train_criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            if ema is not None:
                ema.update(model)
            run_loss += loss.item() * y.size(0)
            run_correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)

        eval_model = ema.model if ema is not None else model
        val = evaluate(eval_model, val_dl, device, eval_criterion, args.max_steps_per_epoch)
        row = {"epoch": epoch, "train_loss": run_loss / seen, "train_acc": run_correct / seen,
               "val_loss": val["loss"], "val_top1": val["top1"], "val_top5": val["top5"],
               "sec": round(time.time() - t0, 1)}
        history.append(row)
        print(f"ep {epoch:3d} | train loss {row['train_loss']:.3f} acc {row['train_acc']:.3f} | "
              f"val loss {row['val_loss']:.3f} top1 {row['val_top1']:.3f} top5 {row['val_top5']:.3f} | {row['sec']}s")
        (out / "history.json").write_text(json.dumps(history, indent=2))

        if args.trainval:
            save_checkpoint(out / "best.pt", eval_model, cfg, row)
            continue
        score = -val["loss"] if args.select == "loss" else val["top1"]
        if best is None or score > best:
            best, bad = score, 0
            save_checkpoint(out / "best.pt", eval_model, cfg, row)
            print(f"  ✓ new best ({args.select}) -> saved best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs).")
                break

    save_checkpoint(out / "last.pt", ema.model if ema is not None else model, cfg, history[-1])
    print(f"Done. Artifacts in {out}")


if __name__ == "__main__":
    main()