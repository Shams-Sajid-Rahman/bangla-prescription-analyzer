"""Medicine knowledge base: generic name -> Bangla description.

Check coverage:
    python -m src.knowledge_base check --label_map runs/exp1/label_map.json
Build kb_extra.json from the "Medicine Dataset of Bangladesh" (generic.csv):
    python -m src.knowledge_base build --generic_csv generic.csv --label_map runs/exp1/label_map.json [--translate]
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BN_KB = ROOT / "data" / "bangla_generic_info.json"
EXTRA_KB = ROOT / "data" / "kb_extra.json"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


class KnowledgeBase:
    def __init__(self, bn_path: Path = BN_KB, extra_path: Path = EXTRA_KB):
        self.entries: dict[str, dict] = {}
        self.alias: dict[str, str] = {}
        for path, source in ((extra_path, "extra"), (bn_path, "curated")):  # curated overrides
            if Path(path).exists():
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                for generic, info in data.items():
                    if generic.startswith("_"):
                        continue
                    self.entries[generic] = info | {"generic": generic, "source": source}
                    for a in [generic, *info.get("aliases", [])]:
                        self.alias[norm(a)] = generic

    def lookup(self, generic: str) -> dict | None:
        if not generic:
            return None
        key = norm(generic)
        if key in self.alias:
            return self.entries[self.alias[key]]
        # combination products e.g. "Vitamin B complex + Zinc": try each part
        for part in re.split(r"[+/,&]| and ", generic):
            k = norm(part)
            if k and k in self.alias:
                return self.entries[self.alias[k]]
        base = norm(re.sub(r"\(.*?\)", "", generic))  # "Ketoconazole (Cream)" -> "ketoconazole"
        if base in self.alias:
            return self.entries[self.alias[base]]
        close = difflib.get_close_matches(key, list(self.alias), n=1, cutoff=0.85)
        return self.entries[self.alias[close[0]]] if close else None


# --------------------------------------------------------------------------- #
# CLI helpers
# --------------------------------------------------------------------------- #
def _generics_from_label_map(path: str) -> list[str]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return sorted({g for g in d["generic_of"].values() if g})


def cmd_check(args):
    kb = KnowledgeBase()
    gens = _generics_from_label_map(args.label_map)
    missing = [g for g in gens if kb.lookup(g) is None]
    print(f"{len(gens)} generics in dataset, {len(gens) - len(missing)} covered by the KB.")
    if missing:
        print("Add Bangla entries for these in data/bangla_generic_info.json:")
        for g in missing:
            print("  -", g)


def cmd_build(args):
    import pandas as pd

    df = pd.read_csv(args.generic_csv)
    cols = {c.lower().strip(): c for c in df.columns}

    def col(*kws):
        for kw in kws:
            for lc, c in cols.items():
                if kw in lc:
                    return c
        return None

    name_c = col("generic name", "generic_name", "name")
    fields = {
        "uses_en": col("indication description", "indication"),
        "side_effects_en": col("side effects description", "side_effect"),
        "caution_en": col("precautions description", "precaution"),
        "class_en": col("drug class", "therapeutic class"),
    }
    print("Using columns:", {"name": name_c, **fields})
    wanted = {norm(g) for g in _generics_from_label_map(args.label_map)} if args.label_map else None

    translator = None
    if args.translate:
        from transformers import pipeline
        translator = pipeline("translation", model="csebuetnlp/banglat5_nmt_en_bn", max_length=512)

    out = {}
    for _, r in df.iterrows():
        g = str(r[name_c]).strip()
        if wanted is not None and norm(g) not in wanted:
            continue
        info = {}
        for k, c in fields.items():
            if c is not None and isinstance(r[c], str):
                text = re.sub(r"\s+", " ", r[c]).strip()[: args.max_chars]
                info[k] = text
                if translator and k != "class_en":
                    info[k.replace("_en", "_bn")] = translator(text)[0]["translation_text"]
        if info:
            out[g] = info
    EXTRA_KB.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(out)} entries to {EXTRA_KB}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("--label_map", required=True); c.set_defaults(fn=cmd_check)
    b = sub.add_parser("build")
    b.add_argument("--generic_csv", required=True)
    b.add_argument("--label_map", default=None, help="Keep only generics present in the dataset")
    b.add_argument("--translate", action="store_true", help="EN->BN with csebuetnlp/banglat5_nmt_en_bn")
    b.add_argument("--max_chars", type=int, default=600)
    b.set_defaults(fn=cmd_build)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()