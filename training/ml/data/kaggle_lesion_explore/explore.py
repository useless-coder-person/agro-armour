"""Kaggle kernel: download PlantSeg v7 (Zenodo 17719108, CC BY-NC 4.0) and print its layout, metadata and the
hosts/diseases relevant to Agro ARMour (rice, tomato, potato, maize). Output: plantseg_meta.csv + layout.txt only."""
import collections, csv, hashlib, io, shutil, subprocess, zipfile
from pathlib import Path
import numpy as np
from PIL import Image

W = Path("/kaggle/working"); Z = Path("/kaggle/temp/plantseg.zip"); Z.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(["curl", "-sL", "-o", str(Z), "https://zenodo.org/records/17719108/files/plantseg.zip?download=1"], check=True)
print("md5", hashlib.md5(Z.read_bytes()).hexdigest(), "(Zenodo: 9358a66dff88cdd15c4fe009763c40a3)")
zf = zipfile.ZipFile(Z)
names = zf.namelist()
tops = collections.Counter("/".join(n.split("/")[:3]) for n in names if not n.endswith("/"))
lines = [f"{len(names)} entries"] + [f"{c:6d} {t}" for t, c in sorted(tops.items())][:60]
metas = [n for n in names if n.lower().endswith(".csv")]
lines.append(f"csv files: {metas}")
for m in metas:
    shutil.copyfileobj(zf.open(m), open(W / Path(m).name, "wb"))
    rows = list(csv.DictReader(io.TextIOWrapper(zf.open(m), encoding="utf-8", errors="replace")))
    lines.append(f"{m}: {len(rows)} rows, columns {list(rows[0].keys()) if rows else []}")
    lines += [str(r) for r in rows[:5]]
    for col in (rows[0].keys() if rows else []):
        vals = collections.Counter(r[col] for r in rows)
        if 1 < len(vals) < 400:
            hits = {v: c for v, c in vals.items() if any(k in v.lower() for k in ("rice", "tomato", "potato", "corn", "maize"))}
            lines.append(f"column {col}: {len(vals)} values; relevant: {sorted(hits.items())}")
pngs = [n for n in names if n.endswith(".png")][:20]
for n in pngs[:8]:
    a = np.array(Image.open(zf.open(n)))
    lines.append(f"{n}: shape {a.shape} values {np.unique(a)[:10]}")
(W / "layout.txt").write_text("\n".join(lines)); print("\n".join(lines))
