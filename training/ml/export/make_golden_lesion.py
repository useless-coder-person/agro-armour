"""Golden files for the on-device lesion parity test (DeviceGoldenParityTest.lesionParityOnDevice).

  python ml/export/make_golden_lesion.py --data ~/cropdoc/datasets/lesion_ds [--n 6]

Writes android/app/src/androidTest/assets/golden/lesion/ (git-ignored): NN_src.png (lossless source), NN_cls.u8 (argmax
class per letterboxed pixel from the host .pte, 255 on padding), manifest.json (w, h, r), model.pte.
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import MODELS, REPO  # noqa: E402
from export.make_golden import letterbox  # noqa: E402

MIN_LEAF = 0.03  # vision/LesionSegmenter.kt LesionMath.MIN_LEAF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    from executorch.runtime import Runtime
    pte = MODELS / "lesion/model.pte"
    size = json.loads((MODELS / "lesion/io_spec.json").read_text())["input"][2]
    method = Runtime.get().load_program(str(pte)).load_method("forward")
    rows = [r for r in csv.DictReader(open(Path(a.data) / "manifest.csv"))
            if r["split"] == "test" and r["source"] in ("riceseg", "plantseg") and r.get("leaf_ok") != "0"][:a.n]
    out = REPO / "android/app/src/androidTest/assets/golden/lesion"
    shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
    items = []
    for k, r in enumerate(rows):
        rgb = cv2.cvtColor(cv2.imread(str(Path(a.data) / r["image"])), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        cv2.imwrite(str(out / f"{k:02d}_src.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        x, lb = letterbox(rgb, size)
        cls = method.execute([torch.from_numpy(x).permute(2, 0, 1)[None].float().contiguous() / 255])[0][0].argmax(0).numpy().astype(np.uint8)  # contiguous: the runtime ignores strides
        inside = np.zeros_like(cls, bool)
        inside[lb["pad_top"]:lb["pad_top"] + lb["new_h"], lb["pad_left"]:lb["pad_left"] + lb["new_w"]] = True
        cls[~inside] = 255
        leaf, les = int((cls == 1).sum()), int((cls == 2).sum())
        ratio = None if leaf + les < MIN_LEAF * inside.sum() else les / (leaf + les)
        (out / f"{k:02d}_cls.u8").write_bytes(cls.tobytes())
        items.append({"id": f"{k:02d}", "w": w, "h": h, "ratio": ratio, "image": r["image"]})
    (out / "manifest.json").write_text(json.dumps({"size": size, "items": items}, indent=1))
    shutil.copy(pte, out / "model.pte")
    print(json.dumps(items))


if __name__ == "__main__":
    main()
