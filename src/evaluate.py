"""Evaluate a trained checkpoint on the Testing split.

    python -m src.evaluate --data_root DATA --run_dir RUN

Writes to run_dir: test_metrics.json, classification_report.csv,
confusion_matrix.png, misclassified.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .data import LabelMap, PrescriptionWords, load_all, transforms_from_cfg
from .model import load_checkpoint


@torch.no_grad()
def predict_all(model, loader, device):
    probs, ys = [], []
    for x, y in loader:
        logits = model(x.to(device))
        probs.append(logits.softmax(1).cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args(argv)

    run = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = LabelMap.load(run / "label_map.json")
    model, cfg = load_checkpoint(run / args.ckpt, len(label_map.classes), str(device))

    df = load_all(args.data_root)[args.split]
    ds = PrescriptionWords(df, label_map, transforms_from_cfg(cfg))
    probs, y = predict_all(model, DataLoader(ds, args.batch_size, num_workers=2), device)
    pred = probs.argmax(1)
    top5 = np.argsort(-probs, axis=1)[:, :5]

    # generic-level accuracy: wrong brand but same active ingredient is a "safe" mistake
    gen = np.array([label_map.generic_of.get(c, "") for c in label_map.classes])
    metrics = {
        "split": args.split,
        "n": int(len(y)),
        "top1_acc": float((pred == y).mean()),
        "top5_acc": float((top5 == y[:, None]).any(1).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "generic_acc": float((gen[pred] == gen[y]).mean()),
    }
    print(json.dumps(metrics, indent=2))
    (run / f"{args.split}_metrics.json").write_text(json.dumps(metrics, indent=2))

    labels = list(range(len(label_map.classes)))
    rep = classification_report(y, pred, labels=labels, target_names=label_map.classes,
                                output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(run / "classification_report.csv")

    cm = confusion_matrix(y, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(18, 18))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(labels, label_map.classes, rotation=90, fontsize=6)
    ax.set_yticks(labels, label_map.classes, fontsize=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix ({args.split}) — top1 {metrics['top1_acc']:.3f}")
    fig.tight_layout(); fig.savefig(run / "confusion_matrix.png", dpi=150); plt.close(fig)

    wrong = df.assign(pred=[label_map.classes[i] for i in pred],
                      conf=probs.max(1)).loc[pred != y]
    wrong.to_csv(run / "misclassified.csv", index=False)
    print(f"{len(wrong)} misclassified images listed in {run / 'misclassified.csv'}")
    return metrics


if __name__ == "__main__":
    main()