"""Model factory (any timm backbone, ImageNet-pretrained, new classifier head)."""
from __future__ import annotations

from pathlib import Path

import timm
import torch


def build_model(name: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.3):
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes,
                             drop_rate=drop_rate)


def save_checkpoint(path: str | Path, model, cfg: dict, metrics: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "metrics": metrics}, path)


def load_checkpoint(path: str | Path, num_classes: int, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model(cfg["model"], num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), cfg
