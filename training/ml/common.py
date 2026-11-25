"""Paths and small helpers shared by ml/ scripts. Run everything inside WSL2/Linux.

Heavy data never lives in the repo: raw downloads sit under AGROARMOUR_RAW, and all
generated datasets/runs/models sit under AGROARMOUR_WORK (WSL ext4, not /mnt/c).
"""
import datetime
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("AGROARMOUR_RAW", "~/arm-data/raw")).expanduser()
WORK = Path(os.environ.get("AGROARMOUR_WORK", "~/cropdoc")).expanduser()
DATASETS = WORK / "datasets"
RUNS = WORK / "runs"
MODELS = WORK / "models"

IP102_YOLO = RAW / "ip102_det" / "IP102_YOLOv5"          # Kaggle leonidkulyk/ip102-yolov5
IP102_SPLITS = RAW / "ip102_official_splits"             # official VOC ImageSets/Main lists


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "nogit"


def append_experiment(row: dict) -> None:
    """Append one row to docs/EXPERIMENTS.md (experiment log)."""
    cols = ["date", "sha", "task", "name", "config", "data", "metrics", "wall", "machine"]
    row = {"date": datetime.date.today().isoformat(), "sha": git_sha(), **row}
    line = "| " + " | ".join(str(row.get(c, "")) for c in cols) + " |\n"
    (REPO / "docs").mkdir(exist_ok=True)
    with open(REPO / "docs" / "EXPERIMENTS.md", "a", encoding="utf-8") as f:
        f.write(line)
