"""Kaggle kernel: lesion-segmentation dataset for Agro ARMour.

Labels (PNG): 0 background (incl. other leaves), 1 main-leaf tissue, 2 lesion, 3 plant with unknown leaf/lesion split,
255 ignore. Sources:
- RiceSeg-5932 lesion masks (CC BY 4.0, doi:10.17632/92jc6w6mcy.1) on the Sethy et al. rice photos (Kaggle copy).
- PlantSeg v7 (CC BY-NC 4.0, Zenodo 17719108): our crops' leaf diseases only; its own train/val/test split.
- Main-leaf masks for both: SAM 2.1 (Apache-2.0) prompted with points inside the lesion mask (lesions lie on the
  leaf); leaf = SAM mask OR lesion. contact.jpg shows 50 random overlays for the spot check.
- PlantVillage (lab, CC0): colour photo + leaf outline from its `segmented` copy; healthy -> 1, diseased -> 3. 100/class.
- not-a-leaf COCO images (pranavg77/cropdoc-not-a-leaf): all 0.
RiceSeg photos are augmented copies of ~1.9k originals: copies (pHash <= 6 incl. flips/rotations) share one group, and
groups are split 80/10/10 with seed 0. Output: lesion_ds.zip (images/, labels/, manifest.csv), contact.jpg, stats.json.
"""
import csv
import io
import json
import random
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imagehash", "sam2==1.1.0"], check=True)
import imagehash  # noqa: E402
import torch  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402

IN, TMP, OUT = Path("/kaggle/input"), Path("/kaggle/temp"), Path("/kaggle/working")
DS = TMP / "lesion_ds"
(DS / "images").mkdir(parents=True, exist_ok=True)
(DS / "labels").mkdir(exist_ok=True)
rng = random.Random(0)
MAXSIDE = 768
rows, stats = [], Counter()


def fetch(url, dst):
    if not dst.exists():
        subprocess.run(["curl", "-sL", "-o", str(dst), url], check=True)
    return dst


def save(name, img, lab, split, source, crop, disease, licence):
    img = img.convert("RGB")
    lab = Image.fromarray(lab)
    s = MAXSIDE / max(img.size)
    if s < 1:
        size = (round(img.width * s), round(img.height * s))
        img, lab = img.resize(size, Image.BILINEAR), lab.resize(size, Image.NEAREST)
    img.save(DS / "images" / f"{name}.jpg", quality=92)
    lab.save(DS / "labels" / f"{name}.png")
    rows.append({"image": f"images/{name}.jpg", "label": f"labels/{name}.png", "split": split, "source": source,
                 "crop": crop, "disease": disease, "licence": licence})


sam = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-base-plus", device="cuda")


def leaf_label(img, lesion):
    """Main leaf from SAM prompted at lesion pixels: the largest candidate that still covers the lesions."""
    ys, xs = np.nonzero(lesion)
    if len(ys) == 0:
        return None
    idx = rng.sample(range(len(ys)), min(8, len(ys)))
    pts = np.stack([xs[idx], ys[idx]], 1).astype(np.float32)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam.set_image(np.asarray(img.convert("RGB")))
        masks, _, _ = sam.predict(point_coords=pts, point_labels=np.ones(len(pts)), multimask_output=True)
    masks = masks.astype(bool)
    cover = [(m & lesion).sum() / lesion.sum() for m in masks]
    frac = [m.mean() for m in masks]
    ok = [i for i in range(len(masks)) if cover[i] >= 0.8 and frac[i] <= 0.95]
    k = max(ok, key=lambda i: frac[i]) if ok else int(np.argmax(cover))
    lab = np.zeros(lesion.shape, np.uint8)
    lab[masks[k] | lesion] = 1
    lab[lesion] = 2
    return lab


def phash(im):
    return int(str(imagehash.phash(im)), 16)


# ---------- RiceSeg ----------
z = zipfile.ZipFile(fetch("https://data.mendeley.com/public-files/datasets/92jc6w6mcy/files/"
                          "fd65f577-8ca7-4787-98ee-2c574e53debc/file_downloaded", TMP / "riceseg.zip"))
rice_root = next(IN.rglob("BLAST1_001.jpg")).parent.parent
items = [(n.split("/")[-2], n.split("/")[-1], n) for n in z.namelist() if n.endswith(".jpg")]
H = np.array([phash(Image.open(rice_root / c / f).convert("L")) for c, f, _ in items], dtype=np.uint64)
popcount = np.vectorize(lambda v: bin(int(v)).count("1"))
parent = list(range(len(items)))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


for i, (c, f, _) in enumerate(items):
    im = Image.open(rice_root / c / f).convert("L")
    variants = [int(H[i])] + [phash(im.transpose(t)) for t in (Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,
                                                                Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270)]
    for v in variants:
        for j in np.nonzero(popcount(np.bitwise_xor(H, np.uint64(v))) <= 6)[0]:
            parent[find(int(j))] = find(i)
groups = sorted({find(i) for i in range(len(items))})
rng.shuffle(groups)
gsplit = {g: ("test" if k < len(groups) * 0.1 else "val" if k < len(groups) * 0.2 else "train") for k, g in enumerate(groups)}
stats["riceseg_groups"] = len(groups)
for i, (c, f, n) in enumerate(items):
    img = Image.open(rice_root / c / f).convert("RGB")
    lesion = np.asarray(Image.open(io.BytesIO(z.read(n))).convert("L").resize(img.size, Image.NEAREST)) > 127
    lab = leaf_label(img, lesion)
    if lab is None:
        stats["riceseg_empty"] += 1
        continue
    sp = gsplit[find(i)]
    save(f"riceseg_{Path(f).stem}", img, lab, sp, "riceseg", "rice", c.lower(), "CC BY 4.0")
    stats[f"riceseg_{sp}"] += 1
print(dict(stats), flush=True)

# ---------- PlantSeg ----------
KEEP = {"corn gray leaf spot": "maize", "corn northern leaf blight": "maize", "corn rust": "maize",
        "potato early blight": "potato", "potato late blight": "potato", "rice blast": "rice",
        "tomato bacterial leaf spot": "tomato", "tomato early blight": "tomato", "tomato late blight": "tomato",
        "tomato leaf mold": "tomato", "tomato septoria leaf spot": "tomato"}
SPLIT = {"Training": "train", "Validation": "val", "Test": "test"}
pz = zipfile.ZipFile(fetch("https://zenodo.org/records/17719108/files/plantseg.zip?download=1", TMP / "plantseg.zip"))
for r in csv.DictReader(io.TextIOWrapper(pz.open("plantseg/Metadata.csv"), encoding="utf-8-sig")):
    if r["Disease"] not in KEEP:
        continue
    sp = SPLIT[r["Split"]]
    try:
        img = Image.open(pz.open(f"plantseg/images/{sp}/{r['Name']}")).convert("RGB")
        lesion = np.asarray(Image.open(pz.open(f"plantseg/annotations/{sp}/{r['Label file']}"))) > 0
    except KeyError:
        stats["plantseg_missing"] += 1
        continue
    if lesion.shape != (img.height, img.width):
        lesion = np.asarray(Image.fromarray(lesion).resize(img.size, Image.NEAREST))
    lab = leaf_label(img, lesion)
    if lab is None:
        stats["plantseg_empty"] += 1
        continue
    save(f"plantseg_{Path(r['Name']).stem}", img, lab, sp, "plantseg", KEEP[r["Disease"]], r["Disease"], "CC BY-NC 4.0")
    stats[f"plantseg_{sp}"] += 1
print(dict(stats), flush=True)

# ---------- PlantVillage (lab): colour photo + leaf outline from the segmented copy ----------
pv = next(IN.rglob("segmented")).parent
for cdir in sorted((pv / "segmented").iterdir()):
    crop = {"Tomato": "tomato", "Potato": "potato", "Corn_(maize)": "maize"}.get(cdir.name.split("___")[0])
    if not crop:
        continue
    healthy = cdir.name.endswith("healthy")
    colour = {p.stem.lower(): p for p in (pv / "color" / cdir.name).iterdir()}
    files = sorted(cdir.iterdir())
    rng.shuffle(files)
    k = 0
    for f in files:
        c = colour.get(f.stem.replace("_final_masked", "").lower())
        if c is None or k >= 100:
            continue
        fg = np.asarray(Image.open(f).convert("L")) > 12
        lab = np.where(fg, 1 if healthy else 3, 0).astype(np.uint8)
        sp = "test" if k < 10 else "val" if k < 20 else "train"
        save(f"pv_{cdir.name}_{k}", Image.open(c), lab, sp, "plantvillage", crop, cdir.name, "CC0 1.0")
        k += 1
        stats["plantvillage"] += 1

# ---------- not a leaf: all background ----------
nz = zipfile.ZipFile(next(IN.rglob("not_a_leaf.zip")))
for k, n in enumerate(sorted(x for x in nz.namelist() if x.endswith(".jpg"))):
    img = Image.open(io.BytesIO(nz.read(n)))
    sp = "test" if k % 10 == 0 else "val" if k % 10 == 1 else "train"
    save(f"nal_{Path(n).stem}", img, np.zeros((img.height, img.width), np.uint8), sp, "not_a_leaf", "", "",
         "see not-a-leaf manifest")
    stats["not_a_leaf"] += 1

with open(DS / "manifest.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# ---------- contact sheet: 50 random SAM-labelled images (leaf tinted green, lesion red) ----------
pick = rng.sample([r for r in rows if r["source"] in ("riceseg", "plantseg")], 50)
sheet = Image.new("RGB", (2000, 1000), "white")
for i, r in enumerate(pick):
    a = np.asarray(Image.open(DS / r["image"]).convert("RGB")).astype(np.float32)
    lb = np.asarray(Image.open(DS / r["label"]))
    a[lb == 1] = a[lb == 1] * 0.6 + np.array([0, 255, 0]) * 0.4
    a[lb == 2] = a[lb == 2] * 0.4 + np.array([255, 0, 0]) * 0.6
    t = Image.fromarray(a.astype(np.uint8))
    t.thumbnail((196, 196))
    sheet.paste(t, ((i % 10) * 200 + 2, (i // 10) * 200 + 2))
sheet.save(OUT / "contact.jpg", quality=85)
json.dump({**stats, "rows": len(rows), "contact": [r["image"] for r in pick]}, open(OUT / "stats.json", "w"), indent=1)
subprocess.run(["bash", "-c", f"cd {TMP} && zip -qr {OUT}/lesion_ds.zip lesion_ds"], check=True)
print(json.dumps(stats))
