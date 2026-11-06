"""Dataset discovery, loading and augmentation.

The Kaggle dataset (mamun1113/doctors-handwritten-prescription-bd-dataset) looks like:

    Doctor’s Handwritten Prescription BD dataset/
        Training/    <images sub-folder> + <labels>.csv  (60%)
        Validation/  <images sub-folder> + <labels>.csv  (20%)
        Testing/     <images sub-folder> + <labels>.csv  (20%)

Each CSV maps an image file name to MEDICINE_NAME and GENERIC_NAME.
Folder / file names are discovered automatically, so small naming
differences between dataset versions do not break the code.
"""
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
    """Return {'train': Path, 'val': Path, 'test': Path} found anywhere under root."""
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
    """Read one split -> DataFrame[path, medicine, generic]."""
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

    # index every image file under the split folder by name and by stem
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
        rows.append(
            {
                "path": str(p),
                "medicine": str(r[med_col]).strip(),
                "generic": str(r[gen_col]).strip() if gen_col else "",
            }
        )
    if missing:
        print(f"[warn] {split_dir.name}: {missing} CSV rows had no matching image file")
    return pd.DataFrame(rows)


def load_all(root: str | Path) -> dict[str, pd.DataFrame]:
    dirs = find_split_dirs(root)
    return {split: load_split(d) for split, d in dirs.items()}


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
        generic_of = (
            all_df.drop_duplicates("medicine").set_index("medicine")["generic"].to_dict()
        )
        return cls(classes, generic_of)

    @property
    def to_idx(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.classes)}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"classes": self.classes, "generic_of": self.generic_of},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "LabelMap":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["classes"], d["generic_of"])


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
class PadToSquare:
    """Pad with white so the word keeps its aspect ratio before resizing."""

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = max(w, h)
        return ImageOps.pad(img, (s, s), color=(255, 255, 255), centering=(0.5, 0.5)) \
            if w != h else img


def build_transforms(img_size: int = 224, train: bool = True) -> T.Compose:
    base = [PadToSquare(), T.Resize((img_size, img_size))]
    if train:
        # NOTE: no horizontal/vertical flips — flipped text is a different "word".
        aug = [
            T.RandomApply([T.RandomAffine(degrees=5, translate=(0.05, 0.05),
                                          scale=(0.9, 1.1), shear=5, fill=255)], p=0.8),
            T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.2),
        ]
        tail = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                T.RandomErasing(p=0.2, scale=(0.02, 0.08), value=1.0)]
        return T.Compose(base + aug + tail)
    return T.Compose(base + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def load_image(path_or_img) -> Image.Image:
    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    return img.convert("L").convert("RGB")  # grayscale handwriting, 3 channels for CNNs


class PrescriptionWords(Dataset):
    def __init__(self, df: pd.DataFrame, label_map: LabelMap, transform):
        self.paths = df["path"].tolist()
        to_idx = label_map.to_idx
        self.targets = [to_idx[m] for m in df["medicine"]]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        img = load_image(self.paths[i])
        return self.transform(img), torch.tensor(self.targets[i])
