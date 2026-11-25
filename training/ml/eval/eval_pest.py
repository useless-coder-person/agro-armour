"""Pest detector evaluation on the exported .pte, scored the way the app decodes it.

Covers what the plan splits into eval_pest / eval_groups / pick_thresholds (one pass over the cached detections):
  1. Ultralytics val() of the .pte on the official test split (iou 0.5, max_det 50 = the app's NMS):
     mAP50, mAP50-95, P, R, per-class AP, confusion matrix.
  2. App decode (make_golden.decode_raw, conf 0.05) cached for val, test and 500 pest-free leaf images.
  3. Per-group threshold on val: lowest top-detection score with image-level precision >= 0.85.
     Bands: HIGH >= t, MEDIUM >= 0.6 t. -> models/<name>/thresholds.json (bundled in the app).
  4. Test: group-level image accuracy (single-GT-group images, top detection's group), group confusion,
     false-positive rate on clean leaves at HIGH and at MEDIUM.

  python ml/eval/eval_pest.py --name pest
"""
import argparse
import collections
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATASETS, MODELS, REPO, RUNS, WORK, append_experiment  # noqa: E402
from data.ip102_to_yolo import selected_classes  # noqa: E402
import export.make_golden as mg  # noqa: E402

PREC_TARGET, MED_FRAC, CACHE_CONF = 0.85, 0.6, 0.05


def group_of_class():
    ids, _ = selected_classes()
    g = yaml.safe_load(open(REPO / "ml/configs/groups.yaml"))
    inv = {ip: name for name, members in g.items() for ip in members}
    missing = [i for i in ids if i not in inv]
    assert not missing, f"classes without a group: {missing}"
    return [inv[i] for i in ids], sorted(g)


def gt_classes(img_path):
    lbl = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
    return {int(l.split()[0]) for l in open(lbl) if l.strip()} if lbl.exists() else set()


def run_cache(method, paths, cache):
    if cache.exists():
        return json.load(open(cache))
    mg.CONF = CACHE_CONF
    out = {}
    for k, p in enumerate(paths):
        rgb = cv2.imread(str(p))[:, :, ::-1].copy()
        h, w = rgb.shape[:2]
        lb_img, lb = mg.letterbox(rgb, 640)
        x = torch.from_numpy(lb_img.transpose(2, 0, 1).copy())[None].float() / 255
        dets = mg.decode_raw(method.execute([x])[0].numpy(), lb, w, h)
        out[str(p)] = [{"cls": d["cls"], "score": round(d["score"], 4),
                        "area": (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]) / (w * h)} for d in dets]
        if k % 500 == 0:
            print(f"  {cache.name}: {k}/{len(paths)}", flush=True)
    json.dump(out, open(cache, "w"))
    return out


def top_group(dets, c2g):
    return (c2g[dets[0]["cls"]], dets[0]["score"]) if dets else (None, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pest")
    ap.add_argument("--skip-val", action="store_true", help="skip the Ultralytics mAP pass")
    a = ap.parse_args()
    from executorch.runtime import Runtime
    from ultralytics import YOLO

    d = MODELS / a.name
    ev = RUNS / "pest" / "P1" / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    c2g, groups = group_of_class()
    _, names = selected_classes()
    res = {"model_sha256": json.load(open(d / "io_spec.json"))["sha256"]}

    # 1. mAP on the official test split, .pte through Ultralytics' runtime path
    if not a.skip_val:
        alias = WORK / "scratch" / f"{a.name}_executorch_model"
        alias.parent.mkdir(exist_ok=True)
        if not alias.exists():
            alias.symlink_to(d)
        m = YOLO(str(alias), task="detect").val(data=str(DATASETS / "pest/pest.yaml"), split="test", imgsz=640,
                                                 batch=1, device="cpu", iou=0.5, max_det=50, project=str(ev),
                                                 name="test_val", exist_ok=True, plots=True, verbose=False)
        res["test_map"] = {"mAP50": m.box.map50, "mAP50_95": m.box.map, "P": m.box.mp, "R": m.box.mr}
        res["per_class_ap50"] = {names[c]: float(m.box.ap50[i]) for i, c in enumerate(m.box.ap_class_index)}
        print("test", res["test_map"])

    # 2. cached app-decode detections
    method = Runtime.get().load_program(str(d / "model.pte")).load_method("forward")
    split = {s: [l.strip() for l in open(DATASETS / f"pest/splits/{s}.txt") if l.strip()] for s in ("val", "test")}
    rows = list(csv.DictReader(open(DATASETS / "disease/manifest.csv")))
    pd_kinds = set(yaml.safe_load(open(REPO / "ml/configs/classes_disease.yaml"))["kinds"]["pest_damage"])
    clean = sorted(r["path"] for r in rows if r["split"] in ("val", "test") and r["label"] not in pd_kinds)
    random.Random(0).shuffle(clean)
    clean = clean[:500]  # pest-free leaves from disease val/test only (negatives for P2 come from train)
    det = {s: run_cache(method, split[s], ev / f"dets_{s}.json") for s in ("val", "test")}
    det["clean"] = run_cache(method, clean, ev / "dets_clean.json")

    # 3. thresholds on val (image level: top detection's group vs GT groups)
    thr = {}
    for g in groups:
        pts = []
        for p, ds in det["val"].items():
            tg, s = top_group(ds, c2g)
            if tg == g:
                pts.append((s, g in {c2g[c] for c in gt_classes(p)}))
        pts.sort(key=lambda t: -t[0])
        best, tp = None, 0
        for i, (s, ok) in enumerate(pts, 1):
            tp += ok
            if tp / i >= PREC_TARGET and i >= 3:
                best = s  # keep lowering while precision holds
        thr[g] = {"threshold": round(best, 3) if best is not None else None, "val_support": len(pts)}
    found = [v["threshold"] for v in thr.values() if v["threshold"] is not None]
    default = round(float(np.median(found)), 3) if found else 0.5
    for v in thr.values():
        if v["threshold"] is None:
            v["threshold"], v["fallback"] = default, True
    json.dump({"precision_target": PREC_TARGET, "medium_frac": MED_FRAC, "default": default,
               "class_to_group": c2g, "groups": thr}, open(d / "thresholds.json", "w"), indent=1)

    def band(g, s):
        t = thr[g]["threshold"]
        return "HIGH" if s >= t else "MEDIUM" if s >= MED_FRAC * t else "LOW"

    # 4. test: group image accuracy + confusion, fine-class top-1 on single-class images
    n = ok = n_fine = ok_fine = 0
    conf = collections.Counter()
    for p, ds in det["test"].items():
        gc = gt_classes(p)
        gg = {c2g[c] for c in gc}
        if len(gg) != 1:
            continue
        (g_true,) = gg
        tg, _ = top_group(ds, c2g)
        n += 1
        ok += tg == g_true
        conf[(g_true, tg)] += 1
        if len(gc) == 1:
            n_fine += 1
            ok_fine += bool(ds) and ds[0]["cls"] in gc
    per_true = collections.Counter()
    for (t, _), v in conf.items():
        per_true[t] += v
    confused = sorted(((t, p, v / per_true[t]) for (t, p), v in conf.items() if t != p and p and v / per_true[t] > 0.25),
                      key=lambda x: -x[2])
    fp_high = sum(any(band(c2g[x["cls"]], x["score"]) == "HIGH" for x in ds) for ds in det["clean"].values())
    fp_med = sum(any(band(c2g[x["cls"]], x["score"]) != "LOW" for x in ds) for ds in det["clean"].values())
    nc = len(det["clean"])
    res.update({"group_image_acc": ok / n, "group_images": n, "fine_top1_single_class": ok_fine / n_fine,
                "fine_images": n_fine, "clean_fp_high": fp_high / nc, "clean_fp_medium": fp_med / nc, "clean_n": nc,
                "group_confusions_over_25pct": [{"true": t, "pred": p, "frac": round(f, 3)} for t, p, f in confused],
                "group_recall": {g: round(conf[(g, g)] / per_true[g], 3) for g in groups if per_true[g]},
                "thresholds": {g: v["threshold"] for g, v in thr.items()}})
    json.dump(res, open(ev / "eval_pest.json", "w"), indent=1)

    md = ["| metric | value |", "|---|---|"]
    if "test_map" in res:
        md += [f"| test mAP50 (42 fine classes, .pte, app NMS) | {res['test_map']['mAP50']:.3f} |",
               f"| test mAP50-95 | {res['test_map']['mAP50_95']:.3f} |",
               f"| test precision / recall (Ultralytics best-F1 point) | {res['test_map']['P']:.3f} / {res['test_map']['R']:.3f} |"]
    md += [f"| group-level image accuracy (test, {n} single-group images) | {ok / n:.3f} |",
           f"| fine-class top-1 (test, {n_fine} single-class images) | {ok_fine / n_fine:.3f} |",
           f"| clean-leaf images with a HIGH pest detection ({nc} images) | {fp_high / nc:.3f} |",
           f"| clean-leaf images with a MEDIUM-or-higher detection | {fp_med / nc:.3f} |", "",
           "| group | val threshold | val support | test recall (top-1 group) |", "|---|---|---|---|"]
    md += [f"| {g} | {thr[g]['threshold']}{' (fallback)' if thr[g].get('fallback') else ''} | {thr[g]['val_support']} | "
           f"{res['group_recall'].get(g, '-')} |" for g in groups]
    if "per_class_ap50" in res:
        md += ["", "| fine class | test AP50 |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in res["per_class_ap50"].items()]
    md += ["", "Group pairs confused > 25 % (true → predicted): " +
           (", ".join(f"{x['true']} → {x['pred']} ({x['frac']:.0%})" for x in res["group_confusions_over_25pct"]) or "none")]
    (ev / "eval_pest.md").write_text("\n".join(md) + "\n")
    fig = REPO / "docs/report/figures"
    fig.mkdir(parents=True, exist_ok=True)
    (fig / "pest_eval.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    append_experiment({"task": "pest-eval", "name": "P1 .pte", "config": "eval_pest.py", "data": "official test split",
                       "metrics": (f"mAP50 {res['test_map']['mAP50']:.3f} / " if "test_map" in res else "") +
                       f"group acc {ok / n:.3f} / clean FP(HIGH) {fp_high / nc:.3f}", "wall": "-", "machine": "PRANAV"})


if __name__ == "__main__":
    main()
