"""Build the leaf-disease classification dataset (Ultralytics folder layout, symlinks only).

Splits (seed 0):
  plantdoc      : provided test kept as test; 10% of provided train -> val (stratified)
  paddy_doctor  : stratified 80/10/10
  plantvillage  : capped at `cap` per class; train only, except labels with no field source (80/10/10)
Leakage guard: perceptual hash (pHash, Hamming <= 4) across splits of field sources and lab-only labels; train/val copies of
anything near-identical to a test image are dropped.
Train sampler: each class capped at 1,500 images; classes under 300 oversampled by repeated links.

Output: $AGROARMOUR_WORK/datasets/disease/{train,val,test}/<label>/<src>__<file>, manifest.csv, stats.json.
"""
import collections
import csv
import json
import random
import sys
from pathlib import Path

import imagehash
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATASETS, RAW, REPO  # noqa: E402

SEED, CAP_TRAIN, MIN_TRAIN, HAMMING = 0, 1500, 300, 4
EXT = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def stratified(items, fracs, rng):
    """items: list of (path, label). Returns list of split names aligned with fracs keys."""
    by = collections.defaultdict(list)
    for it in items:
        by[it[1]].append(it)
    out = []
    for lab in sorted(by):
        g = sorted(by[lab])
        rng.shuffle(g)
        n = len(g)
        cuts, acc = [], 0
        for name, f in fracs[:-1]:
            k = round(n * f)
            out += [(p, l, name) for p, l in g[acc:acc + k]]
            acc += k
        out += [(p, l, fracs[-1][0]) for p, l in g[acc:]]
    return out


def main():
    cfg = yaml.safe_load(open(REPO / "ml/configs/classes_disease.yaml"))
    rng = random.Random(SEED)
    rows = []  # (path, label, source, split)
    for src, sc in cfg["sources"].items():
        root = RAW / sc["root"]
        if src == "plantdoc":
            for split in ("train", "test"):
                items = [(p, sc["map"][d.name]) for d in sorted((root / split).iterdir()) if d.name in sc["map"]
                         for p in sorted(d.iterdir()) if p.suffix in EXT]
                if split == "test":
                    rows += [(p, l, src, "test") for p, l in items]
                else:
                    rows += [(p, l, src, s) for p, l, s in stratified(items, [("val", 0.1), ("train", 0.9)], rng)]
        else:
            items = [(p, sc["map"][d.name]) for d in sorted(root.iterdir()) if d.name in sc["map"]
                     for p in sorted(d.iterdir()) if p.suffix in EXT]
            if "cap" in sc:
                by = collections.defaultdict(list)
                for it in items:
                    by[it[1]].append(it)
                field_labels = {r[1] for r in rows if r[3] in ("val", "test")}  # field sources come first in cfg
                for lab in sorted(by):
                    g = rng.sample(sorted(by[lab]), min(sc["cap"], len(by[lab])))
                    if lab in field_labels:
                        rows += [(p, l, src, "train") for p, l in g]
                    else:  # lab-only label: every split needs the class folder; field metrics exclude these rows
                        rows += [(p, l, src, s) for p, l, s in
                                 stratified(g, [("test", 0.1), ("val", 0.1), ("train", 0.8)], rng)]
            else:
                rows += [(p, l, src, s) for p, l, s in
                         stratified(items, [("test", 0.1), ("val", 0.1), ("train", 0.8)], rng)]

    # unreadable files
    good, bad = [], []
    for r in rows:
        try:
            with Image.open(r[0]) as im:
                im.verify()
            good.append(r)
        except Exception:
            bad.append(str(r[0]))
    rows = good

    # near-duplicate leakage guard on field sources
    field = {s for s, sc in cfg["sources"].items() if sc["field"]}
    lab_eval = {r[1] for r in rows if r[2] not in field and r[3] != "train"}  # lab-only labels with val/test rows
    hashes = {}
    for r in rows:
        if r[2] in field or r[1] in lab_eval:  # PlantVillage has several shots per leaf
            with Image.open(r[0]) as im:
                hashes[r[0]] = imagehash.phash(im.convert("RGB"))
    test_h = [(hashes[r[0]], r[0]) for r in rows if r[3] == "test" and r[0] in hashes]
    leak = set()
    for r in rows:
        if r[3] != "test" and r[0] in hashes and any(hashes[r[0]] - h <= HAMMING for h, _ in test_h):
            leak.add(r[0])
    rows = [r for r in rows if r[0] not in leak]

    out = DATASETS / "disease"
    if out.exists():
        for p in sorted(out.rglob("*"), reverse=True):
            if p.is_symlink() or p.is_file():
                p.unlink()
            elif p.is_dir():
                p.rmdir()
    labels = sorted({r[1] for r in rows})
    stats = {"labels": labels, "unreadable": bad, "dropped_near_dup_of_test": len(leak), "counts": {}}
    with open(out.mkdir(parents=True, exist_ok=True) or out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "source", "split", "copies"])
        for split in ("train", "val", "test"):
            by = collections.defaultdict(list)
            for r in rows:
                if r[3] == split:
                    by[r[1]].append(r)
            for lab in labels:
                g = sorted(by.get(lab, []))
                if split == "train":
                    if len(g) > CAP_TRAIN:
                        g = rng.sample(g, CAP_TRAIN)
                    reps = [1] * len(g)
                    if g and len(g) < MIN_TRAIN:  # oversample by repetition up to MIN_TRAIN
                        for i in range(MIN_TRAIN - len(g)):
                            reps[i % len(g)] += 1
                else:
                    reps = [1] * len(g)
                d = out / split / lab
                d.mkdir(parents=True, exist_ok=True)
                for (p, l, s, _), k in zip(g, reps):
                    for c in range(k):
                        (d / f"{s}__{Path(p).stem}{'' if c == 0 else f'_r{c}'}{Path(p).suffix.lower()}").symlink_to(p)
                    w.writerow([p, l, s, split, k])
                src_counts = collections.Counter(r[2] for r in g)
                stats["counts"].setdefault(lab, {})[split] = {"unique": len(g), "with_repeats": sum(reps), **src_counts}
    json.dump(stats, open(out / "stats.json", "w"), indent=1)
    print(f"{len(labels)} labels; unreadable {len(bad)}; near-dup-of-test dropped {len(leak)}")
    print(f"{'label':34s} {'train(uniq/rep)':>16s} {'val':>5s} {'test':>5s}")
    for lab in labels:
        c = stats["counts"][lab]
        tr = c.get("train", {})
        print(f"{lab:34s} {tr.get('unique', 0):7d}/{tr.get('with_repeats', 0):<7d} "
              f"{c.get('val', {}).get('unique', 0):5d} {c.get('test', {}).get('unique', 0):5d}")


if __name__ == "__main__":
    main()
