"""Inference: handwritten medicine-word image -> medicine name + Bangla description.

    python -m src.analyzer --run_dir runs/exp1 --image path/to/word.png
    python -m src.analyzer --run_dir runs/s42 runs/s7 --image path/to/word.png   # 2-model ensemble
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import LabelMap, load_image, transforms_from_cfg
from .knowledge_base import KnowledgeBase
from .model import load_checkpoint

DISCLAIMER_BN = (
    "⚠️ সতর্কতা: এটি একটি শিক্ষামূলক/গবেষণা প্রকল্পের স্বয়ংক্রিয় ফলাফল এবং ভুল হতে পারে। "
    "ওষুধ সেবনের আগে অবশ্যই চিকিৎসক বা ফার্মাসিস্টের সাথে মিলিয়ে নিন। "
    "এখানে কোনো ওষুধের মাত্রা (ডোজ) দেওয়া হয়নি — প্রেসক্রিপশনে লেখা মাত্রাই অনুসরণ করুন।"
)


# Test-time augmentation used in the final test: small horizontal / vertical shifts.
TTA_SHIFTS = (0, -6, 6)


class PrescriptionAnalyzer:
    """Loads one or more trained runs. With several runs the probabilities are averaged (ensemble)."""

    def __init__(self, run_dir, ckpt: str = "best.pt", device: str | None = None,
                 low_conf: float = 0.60, tta: bool = True):
        runs = [Path(run_dir)] if isinstance(run_dir, (str, Path)) else [Path(r) for r in run_dir]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = LabelMap.load(runs[0] / "label_map.json")
        self.models, self.tfs = [], []
        for run in runs:
            lm = LabelMap.load(run / "label_map.json")
            if lm.classes != self.labels.classes:
                raise ValueError(f"{run} has a different class list than {runs[0]}; cannot ensemble.")
            model, cfg = load_checkpoint(run / ckpt, len(lm.classes), self.device)
            self.models.append(model)
            self.tfs.append(transforms_from_cfg(cfg))
        self.kb = KnowledgeBase()
        self.low_conf = low_conf
        self.tta = tta

    def _model_probs(self, model, x: torch.Tensor) -> torch.Tensor:
        if not self.tta:
            return model(x).softmax(1)
        p = 0
        for s in TTA_SHIFTS:
            for dims in ((3,), (2,)):
                if s == 0 and dims == (2,):
                    continue
                p = p + model(torch.roll(x, s, dims=dims) if s else x).softmax(1)
        return p / p.sum(1, keepdim=True)

    @torch.no_grad()
    def predict(self, image, top_k: int = 3) -> list[dict]:
        img = load_image(image)
        probs = torch.stack([
            self._model_probs(m, tf(img).unsqueeze(0).to(self.device))
            for m, tf in zip(self.models, self.tfs)
        ]).mean(0)[0].cpu()
        conf, idx = probs.topk(min(top_k, len(probs)))
        return [{"medicine": self.labels.classes[i],
                 "generic": self.labels.generic_of.get(self.labels.classes[i], ""),
                 "confidence": round(float(c), 4)} for c, i in zip(conf, idx.tolist())]

    def analyze(self, image, top_k: int = 3) -> dict:
        preds = self.predict(image, top_k)
        best = preds[0]
        info = self.kb.lookup(best["generic"])
        return {
            "prediction": best,
            "alternatives": preds[1:],
            "low_confidence": best["confidence"] < self.low_conf,
            "generic_agreement": sum(p["generic"] == best["generic"] for p in preds) / len(preds),
            "info": info,
        }

    @staticmethod
    def to_bangla_report(res: dict) -> str:
        p, info = res["prediction"], res["info"] or {}
        lines = [
            f"### 💊 শনাক্তকৃত ওষুধ: **{p['medicine']}**",
            f"- নির্ভরযোগ্যতা (confidence): **{p['confidence'] * 100:.1f}%**",
            f"- জেনেরিক নাম: **{p['generic'] or 'অজানা'}**"
            + (f" ({info['bn_name']})" if info.get("bn_name") else ""),
        ]
        if res["low_confidence"]:
            lines.append("- ⚠️ মডেল এই শব্দটি নিয়ে নিশ্চিত নয় — নিচের বিকল্পগুলোও দেখুন এবং ফার্মাসিস্টের সাথে মিলিয়ে নিন।")
        if info:
            for key, title in (("class_bn", "ওষুধের শ্রেণি"), ("uses_bn", "ব্যবহার"),
                               ("side_effects_bn", "সাধারণ পার্শ্বপ্রতিক্রিয়া"),
                               ("caution_bn", "সতর্কতা")):
                if info.get(key):
                    lines.append(f"\n**{title}:** {info[key]}")
            for key, title in (("uses_en", "Uses (EN)"), ("side_effects_en", "Side effects (EN)")):
                if info.get(key) and not info.get(key.replace("_en", "_bn")):
                    lines.append(f"\n**{title}:** {info[key]}")
        else:
            lines.append("\nএই জেনেরিকের বাংলা বিবরণ এখনো ডেটাবেসে যোগ করা হয়নি।")
        if res["alternatives"]:
            alts = ", ".join(f"{a['medicine']} ({a['confidence'] * 100:.0f}%)" for a in res["alternatives"])
            lines.append(f"\n**অন্যান্য সম্ভাব্য নাম:** {alts}")
        lines.append(f"\n---\n{DISCLAIMER_BN}")
        return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, nargs="+", help="one run folder, or several to ensemble")
    p.add_argument("--image", required=True, nargs="+")
    p.add_argument("--no_tta", action="store_true", help="faster, slightly less accurate")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--json", action="store_true", help="Print raw JSON instead of the Bangla report")
    args = p.parse_args(argv)
    an = PrescriptionAnalyzer(args.run_dir, args.ckpt, tta=not args.no_tta)
    for img in args.image:
        res = an.analyze(img)
        print(f"\n===== {img} =====")
        print(json.dumps(res, ensure_ascii=False, indent=2) if args.json else an.to_bangla_report(res))


if __name__ == "__main__":
    main()
