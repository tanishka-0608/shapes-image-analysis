"""Regenerates the synthetic shapes dataset (deterministic, seed=42).

Usage: python generate_dataset.py [output_dir]
"""
import os, math, random, csv, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

random.seed(42)
np.random.seed(42)

import sys
ROOT = sys.argv[1] if len(sys.argv) > 1 else "./shapes_dataset"
CLASSES = ["circle", "square", "triangle", "star", "pentagon", "cross"]
PER_CLASS = 900          # 6 * 900 = 5,400 images
SIZE = 64
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}

def rand_color(lo=0, hi=255):
    return tuple(random.randint(lo, hi) for _ in range(3))

def contrast_pair():
    while True:
        bg, fg = rand_color(), rand_color()
        if sum(abs(a - b) for a, b in zip(bg, fg)) > 200:
            return bg, fg

def regular_polygon(cx, cy, r, n, rot):
    return [(cx + r * math.cos(rot + 2 * math.pi * i / n),
             cy + r * math.sin(rot + 2 * math.pi * i / n)) for i in range(n)]

def star_points(cx, cy, r, rot):
    pts = []
    for i in range(10):
        rad = r if i % 2 == 0 else r * 0.45
        ang = rot + math.pi * i / 5
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    return pts

def draw_shape(name, d, cx, cy, r, rot, fg):
    if name == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg)
    elif name == "square":
        d.polygon(regular_polygon(cx, cy, r * 1.25, 4, rot + math.pi / 4), fill=fg)
    elif name == "triangle":
        d.polygon(regular_polygon(cx, cy, r * 1.2, 3, rot), fill=fg)
    elif name == "pentagon":
        d.polygon(regular_polygon(cx, cy, r * 1.1, 5, rot), fill=fg)
    elif name == "star":
        d.polygon(star_points(cx, cy, r * 1.25, rot), fill=fg)
    elif name == "cross":
        w = r * 0.38
        for a in (rot, rot + math.pi / 2):
            dx, dy = math.cos(a), math.sin(a)
            nx, ny = -dy, dx
            L = r * 1.1
            d.polygon([(cx + dx * L + nx * w, cy + dy * L + ny * w),
                       (cx + dx * L - nx * w, cy + dy * L - ny * w),
                       (cx - dx * L - nx * w, cy - dy * L - ny * w),
                       (cx - dx * L + nx * w, cy - dy * L + ny * w)], fill=fg)

def make_image(name):
    bg, fg = contrast_pair()
    S = SIZE * 4  # supersample for anti-aliasing
    img = Image.new("RGB", (S, S), bg)
    d = ImageDraw.Draw(img)
    r = random.uniform(0.18, 0.34) * S
    margin = r * 1.3
    cx = random.uniform(margin, S - margin)
    cy = random.uniform(margin, S - margin)
    rot = random.uniform(0, 2 * math.pi)
    draw_shape(name, d, cx, cy, r, rot, fg)
    img = img.resize((SIZE, SIZE), Image.LANCZOS)
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.9)))
    arr = np.asarray(img).astype(np.float32)
    arr += np.random.normal(0, random.uniform(2, 12), arr.shape)  # noise
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

if os.path.exists(ROOT):
    shutil.rmtree(ROOT)

rows = {s: [] for s in SPLITS}
for cls in CLASSES:
    idx = list(range(PER_CLASS))
    random.shuffle(idx)
    n_train = int(PER_CLASS * SPLITS["train"])
    n_val = int(PER_CLASS * SPLITS["val"])
    assign = {"train": idx[:n_train], "val": idx[n_train:n_train + n_val],
              "test": idx[n_train + n_val:]}
    for split, ids in assign.items():
        os.makedirs(f"{ROOT}/{split}/{cls}", exist_ok=True)
        for i in ids:
            fn = f"{cls}_{i:04d}.png"
            make_image(cls).save(f"{ROOT}/{split}/{cls}/{fn}", optimize=True)
            rows[split].append((f"{split}/{cls}/{fn}", cls, CLASSES.index(cls)))

for split, r in rows.items():
    random.shuffle(r)
    with open(f"{ROOT}/{split}_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "label", "label_id"])
        w.writerows(r)

with open(f"{ROOT}/README.txt", "w") as f:
    f.write(f"""Synthetic Shapes Dataset
========================
Images     : {PER_CLASS * len(CLASSES)} PNG, {SIZE}x{SIZE} RGB
Classes    : {', '.join(f'{i}={c}' for i, c in enumerate(CLASSES))}
Splits     : train 70% / val 15% / test 15% (balanced per class)
Variation  : random colours, position, size, rotation, blur, Gaussian noise
Layout     : <split>/<class>/<file>.png  +  <split>_labels.csv (filepath,label,label_id)
""")

print("Dataset written to", ROOT, {k: len(v) for k, v in rows.items()})
