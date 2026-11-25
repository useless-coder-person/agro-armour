"""Kaggle kernel: build the other__not_a_leaf class from COCO 2017 val + train.

Keeps images whose per-image licence is CC BY 2.0, CC BY-SA 2.0, "No known copyright restrictions" or
"United States Government Work"; drops images with plant/vegetable/fruit categories or with more than 12 % green
pixels, so foliage is never labelled not-a-leaf. Output: not_a_leaf.zip (900 JPEGs, long side 320) + manifest.csv
(file, coco id, licence, flickr_url) for attribution.
"""
import csv, json, random, zipfile
from pathlib import Path
import numpy as np
from PIL import Image

root = next(Path("/kaggle/input").rglob("coco2017"))
OK_LIC = {"Attribution License", "Attribution-ShareAlike License", "No known copyright restrictions",
          "United States Government Work"}
BAD_CATS = {"potted plant", "broccoli", "carrot", "banana", "apple", "orange"}
rng = random.Random(0)
picked, out = [], Path("/kaggle/working")
for split in ("val2017", "train2017"):
    ann = json.load(open(root / f"annotations/instances_{split}.json"))
    lic = {l["id"]: l["name"] for l in ann["licenses"]}
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    bad = {}
    for a in ann["annotations"]:
        if cats[a["category_id"]] in BAD_CATS:
            bad[a["image_id"]] = True
    imgs = [i for i in ann["images"] if lic[i["license"]] in OK_LIC and i["id"] not in bad]
    rng.shuffle(imgs)
    for i in imgs:
        im = Image.open(root / split / i["file_name"]).convert("RGB")
        im.thumbnail((320, 320))
        hsv = np.asarray(im.convert("HSV")).astype(int)
        green = ((hsv[..., 0] > 25) & (hsv[..., 0] < 115) & (hsv[..., 1] > 60) & (hsv[..., 2] > 40)).mean()
        if green > 0.12:
            continue
        name = f"coco_{i['id']}.jpg"
        im.save(out / name, quality=90)
        picked.append((name, i["id"], lic[i["license"]], i.get("flickr_url", "")))
        if len(picked) == 900:
            break
    if len(picked) == 900:
        break
with zipfile.ZipFile(out / "not_a_leaf.zip", "w") as z:
    for name, *_ in picked:
        z.write(out / name, name)
        (out / name).unlink()
with open(out / "manifest.csv", "w", newline="") as f:
    csv.writer(f).writerows([("file", "coco_id", "licence", "flickr_url"), *picked])
print(len(picked), "images")
