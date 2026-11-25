"""Verify an exported .pte against its PyTorch checkpoint and record its I/O contract.

1. Loads models/<name>/model.pte with the ExecuTorch Python runtime, records input/output shapes and the
   decoder the app must use into models/<name>/io_spec.json (read by the Android app).
2. Tensor parity: same letterboxed input through the .pt and the .pte, max |diff| reported.
3. Metric parity (--data): Ultralytics val() on both; asserts |mAP50 (det) or top1 (cls) diff| <= 1 point.

  python ml/export/verify_pte.py --name coco_yolo26n --weights yolo26n.pt --data coco128.yaml
"""
import argparse
import hashlib
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import MODELS, WORK  # noqa: E402


def load_pte(path):
    from executorch.runtime import Runtime
    return Runtime.get().load_program(str(path)).load_method("forward")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", help="dataset yaml for metric parity (skip if omitted)")
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox

    os.chdir(WORK)
    d = MODELS / a.name
    meta = yaml.safe_load(open(d / "metadata.yaml"))
    task, imgsz, nc = meta["task"], meta["imgsz"], len(meta["names"])
    method = load_pte(d / "model.pte")
    x = torch.zeros(1, 3, *imgsz)
    outs = method.execute([x])
    oshape = [list(o.shape) for o in outs]
    if task == "detect":
        s = oshape[0]
        decoder = "e2e" if s[-1] == 6 else "raw" if s[1] == 4 + nc else "unknown"
    else:
        decoder = "cls"
    assert decoder != "unknown", f"unexpected output shape {oshape}"
    spec = {"task": task, "input": [1, 3, *imgsz], "input_dtype": "float32", "input_layout": "NCHW RGB [0,1]",
            "letterbox_pad": 114 if task == "detect" else None, "outputs": oshape, "decoder": decoder, "nc": nc,
            "executorch": version("executorch").split("+")[0],
            "sha256": hashlib.sha256((d / "model.pte").read_bytes()).hexdigest()}

    # tensor parity on real images
    pt = YOLO(a.weights).model.float().eval()
    imgs = [Path(__import__("ultralytics").__file__).parent / "assets" / f for f in ("bus.jpg", "zidane.jpg")]
    import cv2
    diffs = []
    for p in imgs:
        im = cv2.imread(str(p))[:, :, ::-1]
        if task == "detect":
            im = LetterBox(imgsz, auto=False)(image=im)
        else:
            im = cv2.resize(im, tuple(imgsz[::-1]))  # shape check only; cls eval transform is checked by val()
        t = torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1)))[None].float() / 255
        with torch.no_grad():
            ref = pt(t)
        ref = ref[0] if isinstance(ref, (list, tuple)) else ref
        got = method.execute([t])[0]
        diffs.append(float((ref - got).abs().max()))
    spec["tensor_parity_max_abs"] = max(diffs)
    print("output shapes", oshape, "decoder", decoder, "tensor max|diff|", diffs)

    if a.data:
        key = "metrics/mAP50(B)" if task == "detect" else "metrics/accuracy_top1"
        kw = dict(data=a.data, imgsz=imgsz[0], batch=1, device="cpu", split=a.split, plots=False, verbose=False)
        m_pt = YOLO(a.weights).val(**kw).results_dict[key]
        alias = WORK / "scratch" / f"{a.name}_executorch_model"  # Ultralytics detects ExecuTorch by this dir suffix
        alias.parent.mkdir(exist_ok=True)
        if not alias.exists():
            alias.symlink_to(d)
        m_et = YOLO(str(alias), task=task).val(**kw).results_dict[key]
        spec["metric_parity"] = {"metric": key, "split": a.split, "pt": m_pt, "pte": m_et, "data": a.data}
        print(f"{key}: pt {m_pt:.4f}  pte {m_et:.4f}")
        assert abs(m_pt - m_et) <= 0.01, "pte metric differs from checkpoint by more than 1 point"
    json.dump(spec, open(d / "io_spec.json", "w"), indent=1)
    print("wrote", d / "io_spec.json")


if __name__ == "__main__":
    main()
