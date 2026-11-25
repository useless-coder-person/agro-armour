"""Lesion severity evaluation on the held-out test split of the lesion dataset.

  python ml/eval/eval_lesion.py --data ~/cropdoc/datasets/lesion_ds [--pte ~/cropdoc/models/lesion/model.pte]

Scores the exported .pte the way the app runs it (letterbox 512, pad 114, argmax over the unpadded area) and, on the
same photos, the Phase 1 colour rules (a port of the deleted LesionEstimator.kt) for comparison. Per source: leaf and
lesion IoU, mean absolute error of r = lesion / (leaf + lesion), and LOW/MEDIUM/HIGH agreement at 10 % / 30 %.
The reference leaf masks are SAM-derived (ml/data/kaggle_lesion_build); lesion masks are human-drawn.
Writes runs/lesion/eval/lesion_eval.json and docs/report/figures/lesion_eval.md.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from train_lesion_seg import letterbox  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BINS = ["LOW", "MEDIUM", "HIGH"]


def level(r):
    return None if r is None else 0 if r < 0.10 else 1 if r <= 0.30 else 2


def colour_ratio(rgb):
    """Port of the Phase 1 LesionEstimator.kt (hue rules), for comparison only."""
    h, w = rgb.shape[:2]
    cw, ch = int(w * 0.8), int(h * 0.8)
    crop = rgb[(h - ch) // 2:(h - ch) // 2 + ch, (w - cw) // 2:(w - cw) // 2 + cw]
    s = 256 / max(cw, ch)
    b = cv2.resize(crop, (max(1, int(cw * s)), max(1, int(ch * s))), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    hsv = cv2.cvtColor(b, cv2.COLOR_RGB2HSV)
    hue, sat, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    leaf = (hue >= 65) & (hue <= 170) & (sat > 0.18) & (v > 0.15)
    cand = ~leaf & ((((hue < 62) | (hue > 330)) & (sat > 0.2) & (v >= 0.15) & (v <= 0.95)) | ((sat < 0.15) & (v >= 0.25) & (v <= 0.75)))
    near = cv2.dilate(leaf.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    nl, ns = int(leaf.sum()), int((cand & near).sum())
    return 0.0 if nl + ns == 0 else ns / (nl + ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pte", default=str(Path.home() / "cropdoc/models/lesion/model.pte"))
    ap.add_argument("--imgsz", type=int, default=512)
    a = ap.parse_args()
    from executorch.runtime import Runtime
    method = Runtime.get().load_program(a.pte).load_method("forward")
    root = Path(a.data)
    rows = [r for r in csv.DictReader(open(root / "manifest.csv")) if r["split"] == "test"]
    acc = {}
    for r in rows:
        rgb = cv2.cvtColor(cv2.imread(str(root / r["image"])), cv2.COLOR_BGR2RGB)
        lab = cv2.imread(str(root / r["label"]), cv2.IMREAD_UNCHANGED)
        x, lb = letterbox(rgb, lab, a.imgsz)
        pred = method.execute([torch.from_numpy(x).permute(2, 0, 1)[None].float().contiguous() / 255])[0][0].argmax(0).numpy()  # contiguous: the runtime ignores strides
        valid = lb != 255  # unpadded area
        s = acc.setdefault(r["source"], {"inter": np.zeros(3), "union": np.zeros(3), "pairs": [], "colour": [], "n": 0})
        s["n"] += 1
        full = valid & (lb <= 2)
        for k in (1, 2):
            s["inter"][k] += ((pred == k) & (lb == k) & full).sum(); s["union"][k] += (((pred == k) | (lb == k)) & full).sum()
        if r["source"] in ("riceseg", "plantseg") and r.get("leaf_ok") != "0":  # reference r exists (lesion_qc.py)
            gl, gs = (lb == 1).sum(), (lb == 2).sum(); pl, ps = ((pred == 1) & valid).sum(), ((pred == 2) & valid).sum()
            p_r = None if pl + ps < 0.03 * valid.sum() else ps / (pl + ps)  # same rule as LesionMath.MIN_LEAF
            s["pairs"].append((gs / (gl + gs), p_r))
            s["colour"].append(colour_ratio(rgb))
        if r["source"] == "not_a_leaf":  # all background: any severity estimate here is a false one
            s["est"] = s.get("est", 0) + int(((pred == 1) | (pred == 2))[valid].sum() >= 0.03 * valid.sum())
    out, md = {}, ["| test source | photos (with reference r) | leaf IoU | lesion IoU | r MAE (model / colour rules) | severity agreement (model / colour rules) | model: no estimate |",
                   "|---|---|---|---|---|---|---|"]
    conf_md = []
    for src, s in sorted(acc.items()):
        iou = s["inter"] / np.maximum(s["union"], 1)
        o = {"photos": s["n"], "leaf_iou": round(float(iou[1]), 3), "lesion_iou": round(float(iou[2]), 3)}
        if s["pairs"]:
            ref = [g for g, _ in s["pairs"]]
            got = [(g, p) for g, p in s["pairs"] if p is not None]
            conf = np.zeros((3, 3), int)
            for g, p in got:
                conf[level(g), level(p)] += 1
            cconf = np.zeros((3, 3), int)
            for g, c in zip(ref, s["colour"]):
                cconf[level(g), level(c)] += 1
            o.update({"r_mae": round(float(np.mean([abs(g - p) for g, p in got])), 3),
                      "r_mae_colour": round(float(np.mean([abs(g - c) for g, c in zip(ref, s["colour"])])), 3),
                      "level_agree": round(float(np.trace(conf) / max(1, conf.sum())), 3),
                      "level_agree_colour": round(float(np.trace(cconf) / max(1, cconf.sum())), 3),
                      "no_estimate": len(ref) - len(got), "confusion": conf.tolist(), "confusion_colour": cconf.tolist(),
                      "reference_levels": [int(sum(level(g) == k for g in ref)) for k in range(3)]})
            md.append(f"| {src} | {s['n']} ({len(ref)}) | {o['leaf_iou']} | {o['lesion_iou']} | {o['r_mae']} / {o['r_mae_colour']} | "
                      f"{o['level_agree']} / {o['level_agree_colour']} | {o['no_estimate']} |")
            conf_md += ["", f"{src}: rows = reference level, columns = model level (LOW, MEDIUM, HIGH): {conf.tolist()}; "
                            f"colour rules: {cconf.tolist()}"]
        else:
            if "est" in s:
                o["false_estimates"] = s["est"]
                md.append(f"| {src} | {s['n']} | - | - | - | - | gave an estimate on {s['est']} |")
            else:
                md.append(f"| {src} | {s['n']} | {o['leaf_iou']} | {o['lesion_iou']} | - | - | - |")
        out[src] = o
    md += conf_md + ["", "Reference leaf masks come from SAM 2 prompted at the human-drawn lesions; lesion masks are human-drawn "
                     "(RiceSeg-5932, PlantSeg); photos whose SAM leaf mask failed the label QC (ml/data/lesion_qc.py) have no reference r. PlantVillage rows have leaf outlines only, not-a-leaf rows are all background. "
                     "Levels: LOW < 10 %, MEDIUM 10-30 %, HIGH > 30 % of the main leaf."]
    d = REPO / "runs/lesion/eval"; d.mkdir(parents=True, exist_ok=True)
    (d / "lesion_eval.json").write_text(json.dumps(out, indent=1))
    (REPO / "docs/report/figures/lesion_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
