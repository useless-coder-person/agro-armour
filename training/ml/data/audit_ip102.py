"""Audit IP102 detection data: full 102-class histogram per split + check the 34-class baseline table.

Splits reported:
  repack_train / repack_val : the Kaggle leonidkulyk/ip102-yolov5 split (the counted split)
  trainval / test           : the official IP102 VOC ImageSets/Main lists (what we train/evaluate on)

Exit code 1 if the baseline table disagrees with the data.
Writes $AGROARMOUR_WORK/datasets/pest/audit.{json,md}.
"""
import collections
import json
import sys

import yaml

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common import DATASETS, IP102_SPLITS, IP102_YOLO, REPO  # noqa: E402


def main() -> int:
    names = yaml.safe_load(open(IP102_YOLO / "ip102.yaml"))["names"]
    cfg = yaml.safe_load(open(REPO / "ml/configs/classes_pest.yaml"))
    official = {s: {l.strip() for l in open(IP102_SPLITS / f"{s}.txt") if l.strip()} for s in ("trainval", "test")}

    boxes = collections.defaultdict(collections.Counter)
    imgs = collections.defaultdict(collections.Counter)
    for rs in ("train", "val"):
        for lf in (IP102_YOLO / "labels" / rs).iterdir():
            cls = [int(l.split()[0]) for l in open(lf) if l.strip()]
            stem = lf.stem
            osplit = "trainval" if stem in official["trainval"] else "test" if stem in official["test"] else "unlisted"
            for split in (f"repack_{rs}", osplit, "all"):
                boxes[split].update(cls)
                imgs[split].update(set(cls))

    splits = ["repack_train", "repack_val", "trainval", "test", "all"]
    rows = [{"id": c, "name": names[c], **{f"{s}_boxes": boxes[s][c] for s in splits},
             **{f"{s}_imgs": imgs[s][c] for s in splits}} for c in range(len(names))]

    out = DATASETS / "pest"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out / "audit.json", "w"), indent=1)
    md = ["| id | name | " + " | ".join(f"{s} boxes/imgs" for s in splits) + " | in 34 |",
          "|---|---|" + "---|" * len(splits) + "---|"]
    for r in rows:
        md.append(f"| {r['id']} | {r['name']} | " + " | ".join(f"{r[s+'_boxes']}/{r[s+'_imgs']}" for s in splits)
                  + f" | {'Y' if r['id'] in cfg['baseline'] else ''} |")
    (out / "audit.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # 1) baseline table check
    bad = [(c, v["boxes"], boxes["repack_train"][c]) for c, v in cfg["baseline"].items()
           if v["boxes"] != boxes["repack_train"][c]]
    print(f"\nBaseline 34-class check vs repack_train boxes: {'OK' if not bad else 'MISMATCH ' + str(bad)}")
    unlisted = sum(boxes["unlisted"].values())
    print(f"Images in repack but not in official lists: boxes={unlisted}")

    # 2) classes >=100 boxes outside the 34 (reported, never auto-added)
    for basis in ("repack_train", "trainval", "all"):
        extra = [(c, names[c], boxes[basis][c]) for c in range(len(names))
                 if boxes[basis][c] >= 100 and c not in cfg["baseline"]]
        print(f"Classes >=100 boxes not in the 34 ({basis}): {extra}")

    # 3) India-priority expansion candidates (40-99 boxes, same basis as the baseline table)
    by_name = {n.lower(): i for i, n in enumerate(names)}
    cand = list(cfg["india_priority"]) + [by_name[n.lower()] for n in cfg["india_priority_names"] if n.lower() in by_name]
    missing = [n for n in cfg["india_priority_names"] if n.lower() not in by_name]
    print("India-priority candidates (id, name, repack_train boxes, trainval boxes, eligible 40-99):")
    for c in cand:
        b = boxes["repack_train"][c]
        print(f"  {c:3d} {names[c]:28s} {b:5d} {boxes['trainval'][c]:5d} {'YES' if 40 <= b <= 99 else 'no'}")
    if missing:
        print(f"  names not found in IP102 list (check spelling by hand): {missing}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
