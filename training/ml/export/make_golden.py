"""Golden parity files for the Android instrumented test.

For N fixed images, writes into android/app/src/androidTest/assets/golden/<name>/ (git-ignored: may contain
dataset imagery):
  NN_src.png        decoded source image, lossless (removes JPEG-decoder differences from the parity test)
  NN_src.rgb        the same pixels as raw RGB uint8 HWC (w, h in manifest) for JVM unit tests
  NN_input.u8       letterboxed RGB uint8, HWC, 640x640x3. Model input == this / 255, transposed to NCHW
                    (stored as uint8: exactly equivalent to the float32 tensor, 4x smaller)
  NN_output.f32     raw .pte output, float32 little-endian, shape in manifest.json
  NN_dets.json      reference decode (this file's decode_raw), boxes in source-image pixels
  manifest.json     shapes, letterbox params, decoder settings

  python ml/export/make_golden.py --name coco_yolo26n --images <list.txt|dir> [--n 20]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import MODELS, REPO  # noqa: E402

CONF, IOU, MAX_DET = 0.25, 0.5, 50  # must match vision/Decoders.kt


def letterbox_params(h, w, size=640):
    """Mirror of ultralytics LetterBox(auto=False, center=True).get_params."""
    r = min(size / h, size / w)
    nw, nh = max(round(w * r), 1), max(round(h * r), 1)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    return r, nw, nh, round(dw - 0.1), round(dh - 0.1)


def letterbox(rgb, size=640):
    h, w = rgb.shape[:2]
    r, nw, nh, left, top = letterbox_params(h, w, size)
    im = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR) if (w, h) != (nw, nh) else rgb
    out = np.full((size, size, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = im
    return out, {"ratio": r, "pad_left": left, "pad_top": top, "new_w": nw, "new_h": nh}


def iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-9)


def decode_raw(out, lb, w, h):
    """(1, 4+nc, A) -> list of dets in source pixels. Class-wise greedy NMS, sorted by score desc."""
    p = out[0].T  # (A, 4+nc)
    cls = p[:, 4:].argmax(1)
    conf = p[np.arange(len(p)), 4 + cls]
    keep = conf >= CONF
    p, cls, conf = p[keep], cls[keep], conf[keep]
    xyxy = np.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2, p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2], 1)
    order = np.argsort(-conf, kind="stable")
    kept = []
    for i in order:
        same = [k for k in kept if cls[k] == cls[i]]
        if same and iou(xyxy[i], xyxy[same]).max() > IOU:
            continue
        kept.append(i)
        if len(kept) == MAX_DET:
            break
    dets = []
    for i in kept:
        b = (xyxy[i] - [lb["pad_left"], lb["pad_top"], lb["pad_left"], lb["pad_top"]]) / lb["ratio"]
        b = np.clip(b, 0, [w, h, w, h])
        dets.append({"cls": int(cls[i]), "score": float(conf[i]), "box": [float(v) for v in b]})
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--images", required=True, help="dir or txt list of image paths")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--extra", nargs="*", default=[], help="images always included first")
    a = ap.parse_args()
    from executorch.runtime import Runtime
    from ultralytics.data.augment import LetterBox

    src = Path(a.images)
    paths = sorted(src.glob("*.jpg")) if src.is_dir() else [Path(l.strip()) for l in open(src) if l.strip()]
    paths = [Path(p) for p in a.extra] + paths
    paths = paths[:a.n]
    d = MODELS / a.name
    spec = json.load(open(d / "io_spec.json"))
    assert spec["decoder"] == "raw", "only the raw decoder has a reference implementation here"
    method = Runtime.get().load_program(str(d / "model.pte")).load_method("forward")
    out_dir = REPO / "android/app/src/androidTest/assets/golden" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    import torch
    for k, p in enumerate(paths):
        rgb = cv2.imread(str(p))[:, :, ::-1].copy()
        h, w = rgb.shape[:2]
        lb_img, lb = letterbox(rgb, spec["input"][2])
        assert np.array_equal(lb_img, LetterBox(spec["input"][2:], auto=False)(image=rgb)), "letterbox mirror drifted"
        x = torch.from_numpy(lb_img.transpose(2, 0, 1).copy())[None].float() / 255
        out = method.execute([x])[0].numpy().astype("<f4")
        dets = decode_raw(out, lb, w, h)
        stem = f"{k:02d}"
        cv2.imwrite(str(out_dir / f"{stem}_src.png"), rgb[:, :, ::-1])
        rgb.tofile(out_dir / f"{stem}_src.rgb")  # same pixels, raw HWC for JVM tests (no image decoder there)
        lb_img.tofile(out_dir / f"{stem}_input.u8")
        out.tofile(out_dir / f"{stem}_output.f32")
        json.dump(dets, open(out_dir / f"{stem}_dets.json", "w"), indent=1)
        items.append({"id": stem, "source": p.name, "w": w, "h": h, "letterbox": lb, "n_dets": len(dets)})
    json.dump({"model": a.name, "model_sha256": spec["sha256"], "input": spec["input"], "output": spec["outputs"][0],
               "conf": CONF, "iou": IOU, "max_det": MAX_DET, "items": items},
              open(out_dir / "manifest.json", "w"), indent=1)
    print(f"wrote {len(items)} golden items to {out_dir}; dets per image: {[i['n_dets'] for i in items]}")


if __name__ == "__main__":
    main()
