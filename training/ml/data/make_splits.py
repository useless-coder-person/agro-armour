"""Pest splits: official IP102 test stays test; a stratified 10% val is carved from official trainval.

Stratification key = the image's rarest selected class (by trainval image count). Fixed seed.
Writes $AGROARMOUR_WORK/datasets/pest/{splits/*.txt, pest.yaml} and the committed id lists ml/splits/pest_*.txt.
"""
import collections
import json
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATASETS, REPO  # noqa: E402

SEED, VAL_FRAC = 0, 0.10


def main():
    out = DATASETS / "pest"
    meta = json.load(open(out / "meta.json"))
    names = json.load(open(out / "hygiene.json"))["names"]
    tv = sorted(s for s, m in meta.items() if m["split"] == "trainval")
    freq = collections.Counter(c for s in tv for c in meta[s]["classes"])
    strata = collections.defaultdict(list)
    for s in tv:
        strata[min(meta[s]["classes"], key=lambda c: freq[c])].append(s)
    rng = random.Random(SEED)
    val = []
    for c in sorted(strata):
        g = strata[c]
        rng.shuffle(g)
        val += g[:max(1, round(len(g) * VAL_FRAC))]
    val = set(val)
    splits = {"train": [s for s in tv if s not in val], "val": sorted(val),
              "test": sorted(s for s, m in meta.items() if m["split"] == "test")}
    assert not (set(splits["train"]) & set(splits["test"])) and not (val & set(splits["test"]))

    (out / "splits").mkdir(exist_ok=True)
    (REPO / "ml/splits").mkdir(exist_ok=True)
    for k, v in splits.items():
        (out / "splits" / f"{k}.txt").write_text("".join(f"{out}/images/{s}.jpg\n" for s in v))
        (REPO / "ml/splits" / f"pest_{k}.txt").write_text("".join(f"{s}\n" for s in v))
        cls = collections.Counter(c for s in v for c in meta[s]["classes"])
        print(f"{k}: {len(v)} images, min class images {min(cls.values())}, classes present {len(cls)}/{len(names)}")
    yaml.safe_dump({"path": str(out), "train": "splits/train.txt", "val": "splits/val.txt", "test": "splits/test.txt",
                    "names": dict(enumerate(names))}, open(out / "pest.yaml", "w"), sort_keys=False)
    print("wrote", out / "pest.yaml")


if __name__ == "__main__":
    main()
