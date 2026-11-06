"""Dataset discovery, loading and augmentation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLIT_ALIASES = {
    "train": ("training", "train"),
    "val": ("validation", "valid", "val"),
    "test": ("testing", "test"),
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def find_split_dirs(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    found: dict[str, Path] = {}
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        name = d.name.lower()
        for split, aliases in SPLIT_ALIASES.items():
            if split not in found and name in aliases and list(d.rglob("*.csv")):
                found[split] = d
    missing = set(SPLIT_ALIASES) - set(found)
    if missing:
        raise FileNotFoundError(
            f"Could not find split folder(s) {sorted(missing)} under {root}. "
            "Expected folders named Training / Validation / Testing that contain a CSV."
        )
    return found


def _pick_col(cols: list[str], *keywords: str) -> str | None:
    for kw in keywords:
        for c in cols:
            if kw in c.lower():
                return c
    return None


def load_split(split_dir: str | Path) -> pd.DataFrame:
    split_dir = Path(split_dir)
    csvs = sorted(split_dir.rglob("*.csv"))
    df = pd.read_csv(csvs[0])
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)
    img_col = _pick_col(cols, "image", "file", "img", "path")
    med_col = _pick_col(cols, "medicine", "label", "word", "class")
    gen_col = _pick_col(cols, "generic")
    if img_col is None or med_col is None:
        raise ValueError(f"Cannot identify image/label columns in {csvs[0]}: {cols}")

    by_name, by_stem = {}, {}
    for p in split_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            by_name[p.name] = p
            by_stem[p.stem] = p

    rows, missing = [], 0
    for _, r in df.iterrows():
        key = str(r[img_col]).strip()
        p = by_name.get(Path(key).name) or by_stem.get(Path(key).stem)
        if p is None:
            missing += 1
            continue
        rows.append({"path": str(p),
                     "medicine": str(r[med_col]).strip(),
                     "generic": str(r[gen_col]).strip() if gen_col else ""})
    if missing:
        print(f"[warn] {split_dir.name}: {missing} CSV rows had no matching image file")
    return pd.DataFrame(rows)


FIXES = Path(__file__).resolve().parents[1] / "data" / "label_fixes.csv"


def load_all(root: str | Path, use_fixes: bool = True) -> dict[str, pd.DataFrame]:
    dirs = find_split_dirs(root)
    frames = {split: load_split(d) for split, d in dirs.items()}
    if use_fixes and FIXES.exists():
        all_df = pd.concat(frames.values())
        generic_of = all_df.drop_duplicates("medicine").set_index("medicine")["generic"].to_dict()
        fx = pd.read_csv(FIXES, dtype=str)
        bad = set(fx.correct_label) - set(generic_of) - {"DROP"}
        if bad:
            raise ValueError(f"label_fixes.csv has unknown medicine names: {bad}")
        changed = dropped = 0
        for split, df in frames.items():
            names = df["path"].map(lambda p: Path(p).name)
            for _, f in fx[fx["split"] == split].iterrows():
                m = names == f["image"]
                if f["correct_label"] == "DROP":
                    dropped += int(m.sum()); df.drop(df.index[m], inplace=True)
                    names = names[~m]
                else:
                    changed += int(m.sum())
                    df.loc[m, "medicine"] = f["correct_label"]
                    df.loc[m, "generic"] = generic_of[f["correct_label"]]
            frames[split] = df.reset_index(drop=True)
        print(f"[label_fixes] relabelled {changed}, dropped {dropped} images")
    return frames


# --------------------------------------------------------------------------- #
# Label map
# --------------------------------------------------------------------------- #
@dataclass
class LabelMap:
    classes: list[str]
    generic_of: dict[str, str]

    @classmethod
    def from_frames(cls, frames: dict[str, pd.DataFrame]) -> "LabelMap":
        all_df = pd.concat(frames.values(), ignore_index=True)
        classes = sorted(all_df["medicine"].unique())
        generic_of = all_df.drop_duplicates("medicine").set_index("medicine")["generic"].to_dict()
        return cls(classes, generic_of)

    @property
    def to_idx(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.classes)}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"classes": self.classes, "generic_of": self.generic_of},
                                         ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LabelMap":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["classes"], d["generic_of"])


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
class PadToSquare:
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = max(w, h)
        return ImageOps.pad(img, (s, s), color=(255, 255, 255)) if w != h else img


class ResizePad:
    """Fit the word inside (H, W) keeping its aspect ratio, pad the rest with white."""

    def __init__(self, h: int, w: int):
        self.h, self.w = h, w

    def __call__(self, img: Image.Image) -> Image.Image:
        return ImageOps.pad(img, (self.w, self.h), color=(255, 255, 255))


class InkThickness:
    """Randomly make pen strokes thicker (MinFilter) or thinner (MaxFilter)."""

    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        import random
        from PIL import ImageFilter
        if random.random() > self.p:
            return img
        return img.filter(ImageFilter.MinFilter(3) if random.random() < 0.5 else ImageFilter.MaxFilter(3))


def build_transforms(img_size: int = 224, train: bool = True, strong: bool = False,
                     input_hw: tuple[int, int] | None = None) -> T.Compose:
    if input_hw:
        base = [ResizePad(*input_hw)]
    else:
        base = [PadToSquare(), T.Resize((img_size, img_size))]
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return T.Compose(base + norm)
    # no flips — flipped text is a different word
    aug = [
        T.RandomApply([T.RandomAffine(degrees=5, translate=(0.05, 0.05),
                                      scale=(0.9, 1.1), shear=5, fill=255)], p=0.8),
        T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.2),
    ]
    if strong:
        aug = [InkThickness(p=0.3),
               T.RandomApply([T.ElasticTransform(alpha=30.0, sigma=5.0, fill=255)], p=0.4),
               T.RandomPerspective(distortion_scale=0.2, p=0.3, fill=255)] + aug
    tail = [T.RandomErasing(p=0.25 if strong else 0.2, scale=(0.02, 0.08), value=1.0)]
    return T.Compose(base + aug + norm + tail)


def transforms_from_cfg(cfg: dict, train: bool = False) -> T.Compose:
    hw = cfg.get("input_hw")
    return build_transforms(cfg.get("img_size", 224), train, cfg.get("aug") == "strong",
                            tuple(hw) if hw else None)


def load_image(path_or_img) -> Image.Image:
    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    return img.convert("L").convert("RGB")


class PrescriptionWords(Dataset):
    def __init__(self, df: pd.DataFrame, label_map: LabelMap, transform):
        self.paths = df["path"].tolist()
        to_idx = label_map.to_idx
        self.targets = [to_idx[m] for m in df["medicine"]]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(load_image(self.paths[i])), torch.tensor(self.targets[i])