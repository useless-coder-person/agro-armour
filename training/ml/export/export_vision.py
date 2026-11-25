"""Export an Ultralytics checkpoint to ExecuTorch (XNNPACK, FP32) into $AGROARMOUR_WORK/models/<name>/.

  python ml/export/export_vision.py yolo26n.pt --name coco_yolo26n --imgsz 640
  python ml/export/export_vision.py runs/pest/P1/weights/best.pt --name pest --imgsz 640

Writes model.pte, metadata.yaml, labels.json. Run verify_pte.py next (writes io_spec.json).
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import MODELS, WORK  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--name", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    from ultralytics import YOLO

    w = Path(a.weights)
    if not w.is_absolute() and not w.exists():
        w = WORK / w
    os.chdir(WORK)  # hub weights (e.g. yolo26n.pt) download here
    src = Path(YOLO(str(w)).export(format="executorch", imgsz=a.imgsz, batch=1))
    out = MODELS / a.name
    out.mkdir(parents=True, exist_ok=True)
    pte = next(src.glob("*.pte"))
    shutil.copy(pte, out / "model.pte")
    shutil.copy(src / "metadata.yaml", out / "metadata.yaml")
    meta = yaml.safe_load(open(src / "metadata.yaml"))
    json.dump([meta["names"][i] for i in sorted(meta["names"])], open(out / "labels.json", "w"), indent=1)
    print("exported", out, f"{(out / 'model.pte').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
