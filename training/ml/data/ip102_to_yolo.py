"""Build the pest detection dataset from the IP102 YOLO repack, keeping only the selected classes.

Source labels are already YOLO-normalised (Kaggle leonidkulyk/ip102-yolov5), so the VOC-specific
hygiene items map to: real image size read from the JPEG (used for the tiny-box
check), boxes clipped to the image, zero/sub-2px boxes dropped, corrupt JPEGs dropped, exact
duplicate images hashed and reported (and removed from trainval when they also occur in test).

Output ($AGROARMOUR_WORK/datasets/pest): images/<id>.jpg (symlinks), labels/<id>.txt (remapped ids),
meta.json (per-image classes + official split), hygiene.json. Run make_splits.py next.
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATASETS, IP102_SPLITS, IP102_YOLO, REPO  # noqa: E402

MIN_PX = 2


def clip_box(cx, cy, w, h):
    """Clip a normalised xywh box to [0,1]; returns None if nothing is left."""
    x1, y1, x2, y2 = max(cx - w / 2, 0), max(cy - h / 2, 0), min(cx + w / 2, 1), min(cy + h / 2, 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def selected_classes():
    cfg = yaml.safe_load(open(REPO / "ml/configs/classes_pest.yaml"))
    sel = {**cfg["baseline"], **cfg["added"]}
    ids = sorted(sel)
    return ids, [sel[i]["name"] for i in ids]


def main():
    assert clip_box(0.5, 0.5, 2, 2) == (0.5, 0.5, 1, 1) and clip_box(1.2, 0.5, 0.1, 0.1) is None  # self-check
    ids, names = selected_classes()
    remap = {c: i for i, c in enumerate(ids)}
    official = {}
    for s in ("trainval", "test"):
        for l in open(IP102_SPLITS / f"{s}.txt"):
            if l.strip():
                official[l.strip()] = s

    out = DATASETS / "pest"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    log = collections.Counter()
    corrupt, hashes, meta = [], collections.defaultdict(list), {}
    for rs in ("train", "val"):
        for img in sorted((IP102_YOLO / "images" / rs).glob("*.jpg")):
            stem = img.stem
            try:
                with Image.open(img) as im:
                    im.verify()
                with Image.open(img) as im:
                    W, H = im.size
                    im.load()  # verify() misses truncated data
            except Exception as e:
                corrupt.append(f"{stem}: {e}")
                continue
            hashes[hashlib.md5(img.read_bytes()).hexdigest()].append(stem)
            rows, unselected = [], 0
            for line in open(IP102_YOLO / "labels" / rs / f"{stem}.txt"):
                p = line.split()
                if not p:
                    continue
                c = int(p[0])
                if c not in remap:
                    unselected += 1
                    continue
                b = clip_box(*map(float, p[1:5]))
                if b is None or b[2] * W < MIN_PX or b[3] * H < MIN_PX:
                    log["boxes_dropped_degenerate"] += 1
                    continue
                if max(abs(u - v) for u, v in zip(b, map(float, p[1:5]))) > 1e-4:
                    log["boxes_clipped"] += 1
                rows.append((remap[c], *b))
            log["boxes_unselected"] += unselected
            if not rows:
                log["images_no_selected_box"] += 1
                continue
            if unselected:  # an unlabelled insect would be learned as background
                log["images_mixed_dropped"] += 1
                continue
            meta[stem] = {"split": official.get(stem, "unlisted"), "classes": sorted({r[0] for r in rows}),
                          "n_boxes": len(rows), "src": str(img), "wh": [W, H]}
            (out / "labels" / f"{stem}.txt").write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in rows))
            link = out / "images" / f"{stem}.jpg"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(img)

    # exact duplicates: report all groups; drop trainval copies of anything that is in test (leak)
    dups = [g for g in hashes.values() if len(g) > 1]
    leak = 0
    for g in dups:
        if any(official.get(s) == "test" for s in g):
            for s in g:
                if official.get(s) == "trainval" and s in meta:
                    meta.pop(s)
                    leak += 1
    stats = {"selected_ids": ids, "names": names, "kept_images": len(meta), "corrupt": corrupt,
             "duplicate_groups": len(dups), "trainval_dropped_as_test_duplicates": leak,
             "dup_examples": dups[:20], **log,
             "per_split": dict(collections.Counter(m["split"] for m in meta.values()))}
    json.dump(meta, open(out / "meta.json", "w"))
    json.dump(stats, open(out / "hygiene.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("dup_examples", "names")}, indent=1))


if __name__ == "__main__":
    main()
