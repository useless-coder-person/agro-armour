"""Disease classifier evaluation on the test split, field and lab sources separately.

Top-1, macro-F1 and per-class recall, with and without the crop prior (zero other crops' classes,
keep other__not_a_leaf, renormalise). Probabilities come from the .pt checkpoint through Ultralytics' predict
(same eval transform as val); verify_pte.py separately gates .pte vs .pt top-1 within 1 point.

  python ml/eval/eval_disease.py --weights runs/disease/D1/weights/best.pt
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATASETS, REPO, WORK, append_experiment  # noqa: E402

FIELD = {"plantdoc", "paddy_doctor"}


def crop_prior(p, names, crop):
    keep = np.array([n.startswith(crop + "__") or n.startswith("other__") for n in names], float)
    q = p * keep
    return q / q.sum() if q.sum() > 0 else p


def scores(y, yhat, labels):
    rec, f1 = {}, []
    for c in labels:
        tp = sum(1 for a, b in zip(y, yhat) if a == c and b == c)
        n_true, n_pred = y.count(c), yhat.count(c)
        if n_true == 0:
            continue
        r, pr = tp / n_true, (tp / n_pred if n_pred else 0.0)
        rec[c] = (round(r, 3), n_true)
        f1.append(2 * pr * r / (pr + r) if pr + r else 0.0)
    acc = sum(a == b for a, b in zip(y, yhat)) / len(y)
    return {"n": len(y), "top1": acc, "macro_f1": float(np.mean(f1)), "recall": rec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/disease/D1/weights/best.pt")
    a = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO(str(WORK / a.weights))
    names = [model.names[i] for i in range(len(model.names))]
    rows = [r for r in csv.DictReader(open(DATASETS / "disease/manifest.csv")) if r["split"] == "test"]
    probs = {}
    paths = [r["path"] for r in rows]
    for i in range(0, len(paths), 64):
        for p, r in zip(paths[i:i + 64], model.predict(paths[i:i + 64], imgsz=224, verbose=False, device=0)):
            probs[p] = r.probs.data.cpu().numpy()
    res = {}
    for group, sel in (("field", lambda r: r["source"] in FIELD), ("lab", lambda r: r["source"] == "plantvillage"),
                       ("not_a_leaf", lambda r: r["label"] == "other__not_a_leaf")):
        rs = [r for r in rows if sel(r)]
        if not rs:
            continue
        y = [r["label"] for r in rs]
        raw = [names[int(probs[r["path"]].argmax())] for r in rs]
        pri = [names[int(crop_prior(probs[r["path"]], names, r["label"].split("__")[0]).argmax())] for r in rs]
        res[group] = {"no_prior": scores(y, raw, names), "crop_prior": scores(y, pri, names),
                      "confusions": collections.Counter(f"{t} -> {p}" for t, p in zip(y, pri) if t != p).most_common(15)}
    out = (WORK / a.weights).parents[1] / "eval"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "eval_disease.json", "w"), indent=1)

    md = ["| source | images | top-1 (no prior) | macro-F1 (no prior) | top-1 (crop prior) | macro-F1 (crop prior) |",
          "|---|---|---|---|---|---|"]
    for g, v in res.items():
        md.append(f"| {g} | {v['crop_prior']['n']} | {v['no_prior']['top1']:.3f} | {v['no_prior']['macro_f1']:.3f} | "
                  f"{v['crop_prior']['top1']:.3f} | {v['crop_prior']['macro_f1']:.3f} |")
    if "other__not_a_leaf" in names:
        nl = names.index("other__not_a_leaf")
        field_rows = [r for r in rows if r["source"] in FIELD]
        wrong = sum(int(probs[r["path"]].argmax()) == nl for r in field_rows)
        md += ["", f"other__not_a_leaf recall (no prior): {res['not_a_leaf']['no_prior']['recall'].get('other__not_a_leaf', ('-', 0))[0]} "
               f"(n={res['not_a_leaf']['no_prior']['n']}, gate 0.90). Field leaf photos called not-a-leaf: {wrong}/{len(field_rows)}."]
    md += ["", "| class | field recall, crop prior (n) |", "|---|---|"]
    md += [f"| {c} | {r} ({n}) |" for c, (r, n) in sorted(res["field"]["crop_prior"]["recall"].items())]
    md += ["", "Most common field confusions (with crop prior): " +
           "; ".join(f"{k} ×{n}" for k, n in res["field"]["confusions"])]
    (out / "eval_disease.md").write_text("\n".join(md) + "\n")
    (REPO / "docs/report/figures/disease_eval.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    f = res["field"]["crop_prior"]
    append_experiment({"task": "disease-eval", "name": Path(a.weights).parts[-3], "config": "eval_disease.py", "data": "test split",
                       "metrics": f"field macro-F1 (crop prior) {f['macro_f1']:.3f} / top-1 {f['top1']:.3f}",
                       "wall": "-", "machine": "PRANAV"})


if __name__ == "__main__":
    main()
