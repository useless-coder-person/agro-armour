"""Train the lesion segmentation net (ml/train/lesion_net.py) on the dataset built by ml/data/kaggle_lesion_build.

  python ml/train/train_lesion_seg.py ml/configs/train_lesion_512.yaml --data <lesion_ds> --out runs/lesion/L1 [--resume]

Labels (PNG): 0 background (incl. other leaves), 1 main-leaf tissue, 2 lesion, 3 plant whose leaf/lesion split is unknown
(PlantVillage diseased leaves: only the leaf outline is known), 255 ignore. Loss: cross-entropy on 0-2, -log(p_leaf +
p_lesion) on 3, plus soft Dice on lesion. Images are letterboxed exactly as the app does (pad 114, INTER_LINEAR).
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lesion_net import LesionNet  # noqa: E402


def letterbox(img, lab, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = max(1, round(w * r)), max(1, round(h * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    lab = cv2.resize(lab, (nw, nh), interpolation=cv2.INTER_NEAREST)
    top, left = round((size - nh) / 2 - 0.1), round((size - nw) / 2 - 0.1)
    oi = np.full((size, size, 3), 114, np.uint8); ol = np.full((size, size), 255, np.uint8)
    oi[top:top + nh, left:left + nw] = img; ol[top:top + nh, left:left + nw] = lab
    return oi, ol


class Ds(torch.utils.data.Dataset):
    def __init__(self, root, split, size, train):
        self.root, self.size, self.train = Path(root), size, train
        self.rows = [r for r in csv.DictReader(open(self.root / "manifest.csv")) if r["split"] == split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.cvtColor(cv2.imread(str(self.root / r["image"])), cv2.COLOR_BGR2RGB)
        lab = cv2.imread(str(self.root / r["label"]), cv2.IMREAD_UNCHANGED)
        if self.train:
            img, lab = augment(img, lab)
        img, lab = letterbox(img, lab, self.size)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255, torch.from_numpy(lab).long()


def augment(img, lab):
    h, w = lab.shape
    k = random.randrange(4)
    img, lab = np.rot90(img, k), np.rot90(lab, k)
    if random.random() < 0.5:
        img, lab = img[:, ::-1], lab[:, ::-1]
    h, w = lab.shape
    s = math.sqrt(random.uniform(0.4, 1.0)); ar = math.exp(random.uniform(-0.3, 0.3))
    cw, ch = min(w, int(w * s * ar)), min(h, int(h * s / ar))
    x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
    img, lab = np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw]), np.ascontiguousarray(lab[y0:y0 + ch, x0:x0 + cw])
    # Photometric jitter: the model must not lean on absolute colour or white balance.
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + random.uniform(-6, 6)) % 180
    hsv[..., 1] *= random.uniform(0.6, 1.4); hsv[..., 2] *= random.uniform(0.6, 1.4)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    img = np.clip((img.astype(np.float32) - 128) * random.uniform(0.7, 1.3) + 128, 0, 255).astype(np.uint8)
    if random.random() < 0.2:
        img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.5, 1.8))
    if random.random() < 0.3:
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(35, 90)])
        img = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    return img, lab


def loss_fn(logits, y):
    p = logits.softmax(1)
    full = y <= 2
    ce = F.cross_entropy(logits, torch.where(full, y, torch.full_like(y, 255)), ignore_index=255, reduction="sum")
    plant = y == 3
    pl = -torch.log((p[:, 1] + p[:, 2])[plant].clamp_min(1e-6)).sum()
    n = (full.sum() + plant.sum()).clamp_min(1)
    t = (y == 2) & full
    p2 = p[:, 2][full]
    dice = 1 - (2 * (p2 * t[full]).sum() + 1) / (p2.sum() + t.sum() + 1)
    return (ce + pl) / n + 0.5 * dice


@torch.no_grad()
def evaluate(model, dl, dev):
    inter, union = torch.zeros(3), torch.zeros(3)
    model.eval()
    for x, y in dl:
        pred = model(x.to(dev)).argmax(1).cpu()
        full = y <= 2
        for k in (1, 2):
            a, b = (pred == k) & full, (y == k)
            inter[k] += (a & b).sum(); union[k] += (a | b).sum()
    iou = (inter / union.clamp_min(1)).tolist()
    return {"leaf_iou": iou[1], "lesion_iou": iou[2], "miou": (iou[1] + iou[2]) / 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    mk = lambda s, tr: torch.utils.data.DataLoader(Ds(a.data, s, cfg["imgsz"], tr), batch_size=cfg["batch"], shuffle=tr,  # noqa: E731
                                                   num_workers=cfg["workers"], drop_last=tr, persistent_workers=True)
    tr, va = mk("train", True), mk("val", False)
    model = LesionNet(pretrained_backbone=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    start, best, hist = 0, -1.0, []
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best, hist = ck["epoch"] + 1, ck["best"], ck["hist"]
    t_start = time.time()
    for ep in range(start, cfg["epochs"]):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        m = {"epoch": ep, "loss": tot / len(tr), **evaluate(model, va, dev), "epoch_s": round(time.time() - t0)}
        hist.append(m); print(json.dumps(m), flush=True)
        if m["miou"] > best:
            best = m["miou"]; torch.save({"model": model.state_dict(), "epoch": ep, "metrics": m}, out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                    "best": best, "hist": hist}, out / "last.pt")
    b = max(hist, key=lambda h: h["miou"])
    (out / "summary.json").write_text(json.dumps({"best": b, "epochs_run": len(hist), "wall_min": round((time.time() - t_start) / 60, 1),
                                                   "config": cfg, "hist": hist}, indent=1))


if __name__ == "__main__":
    main()
