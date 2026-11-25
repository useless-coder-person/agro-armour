"""Label QC for the lesion dataset: find photos whose SAM main-leaf mask failed, and stop them teaching "leaf = background".

  python ml/data/lesion_qc.py <lesion_ds>        (also run inside the Kaggle training kernel, see kaggle_lesion_train/make.sh)

Lesion masks are human-drawn; only the leaf mask (label 1) comes from SAM. The 2026-11-24 contact-sheet check showed two
failure modes, both measurable on the labels alone:
- scattered: the lesions span several leaves (whole-plant shots), so the chosen leaf covers < 50 % of the lesions' convex hull;
- collapsed: SAM returned little more than the lesion itself (leaf < 10 % of the photo and tissue < lesion area).
Flagged photos keep their lesion and leaf pixels but their background (0) becomes ignore (255), and manifest.csv gets
leaf_ok=0 so eval_lesion.py leaves them out of the severity (r) comparison. Idempotent.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HULL_MIN, COLLAPSE_LEAF, COLLAPSE_RIM = 0.5, 0.10, 1.0


def leaf_ok(lab):
    les, leaf = lab == 2, (lab == 1) | (lab == 2)
    if not les.any():
        return True
    hull = np.zeros(lab.shape, np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero(les.astype(np.uint8))), 1)
    hullcov = (leaf & (hull > 0)).sum() / max(1, int(hull.sum()))
    collapsed = leaf.mean() < COLLAPSE_LEAF and (lab == 1).sum() < les.sum()
    return hullcov >= HULL_MIN and not collapsed


def qc(root):
    root = Path(root)
    rows = list(csv.DictReader(open(root / "manifest.csv")))
    n = Counter()
    for r in rows:
        ok = True
        if r["source"] in ("riceseg", "plantseg"):
            p = root / r["label"]
            lab = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            ok = leaf_ok(lab)
            if not ok and (lab == 0).any():
                lab[lab == 0] = 255
                cv2.imwrite(str(p), lab)
            n[f"{r['source']}_{r['split']}_{'ok' if ok else 'flagged'}"] += 1
        r["leaf_ok"] = int(ok)
    with open(root / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(dict(sorted(n.items()))), flush=True)
    return n


if __name__ == "__main__":
    # self-check: a leaf with one spot passes; a lesion-only mask and two far-apart spots on a sliver fail
    a = np.zeros((100, 100), np.uint8); a[10:90, 10:90] = 1; a[40:50, 40:50] = 2
    b = np.zeros((100, 100), np.uint8); b[40:50, 40:50] = 2
    c = np.zeros((100, 100), np.uint8); c[0:5, 0:100] = 1; c[0:5, 0:5] = 2; c[95:100, 95:100] = 2
    assert leaf_ok(a) and not leaf_ok(b) and not leaf_ok(c)
    qc(sys.argv[1])
