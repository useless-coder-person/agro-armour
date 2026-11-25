"""Train the pest detector (also used by train_disease.py) from a YAML config. Output: $AGROARMOUR_WORK/runs/<task>/<name>/.

  python ml/train/train_pest.py ml/configs/train_pest_n640.yaml --name P1 [--device 0] [--epochs 2]
  python ml/train/train_pest.py ml/configs/train_pest_n640.yaml --name P1 --resume
"""
import argparse
import csv
import os
import platform
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RUNS, WORK, append_experiment  # noqa: E402

def main(task="pest", key="metrics/mAP50(B)", key2="metrics/mAP50-95(B)", data_tag="pest42 official-split v1"):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epochs", type=int, help="override (dry runs)")
    ap.add_argument("--fraction", type=float, help="override dataset fraction (dry runs)")
    a = ap.parse_args()

    from ultralytics import YOLO
    a.config = os.path.abspath(a.config)
    os.chdir(WORK)  # pretrained weights download here, not into the repo
    cfg = yaml.safe_load(open(a.config))
    model, data = cfg.pop("model"), str(WORK / cfg.pop("data"))
    data_tag = cfg.pop("data_tag", data_tag)
    if a.epochs:
        cfg["epochs"] = a.epochs
    if a.fraction:
        cfg["fraction"] = a.fraction
    run = RUNS / task / a.name
    t0 = time.time()
    if a.resume:
        YOLO(str(run / "weights" / "last.pt")).train(resume=True, device=a.device)
    else:
        YOLO(model).train(data=data, project=str(RUNS / task), name=a.name, exist_ok=False, device=a.device, **cfg)
    rows = list(csv.DictReader(open(run / "results.csv")))
    best = max(rows, key=lambda r: float(r[key]))
    append_experiment({
        "task": task, "name": a.name + (" (resumed)" if a.resume else ""), "config": Path(a.config).name,
        "data": data_tag,
        "metrics": f"best val {key.split('/')[1]} {float(best[key]):.3f} / {key2.split('/')[1]} {float(best[key2]):.3f} @ep{best['epoch']} ({len(rows)} ep)",
        "wall": f"{(time.time() - t0) / 60:.1f} min (this invocation)", "machine": platform.node()})


if __name__ == "__main__":
    main()
