#!/usr/bin/env python3
"""Build the AVM 6-class YOLO dataset from public / own sources (class re-mapping).

The detector classes are fixed in ai/classes.yaml (person, bollard, fender, quay_edge,
small_vessel, buoy). Every source dataset uses its own class names, so each source is
converted with an explicit mapping  <source name> -> <AVM class>; unmapped classes are dropped.

Sources
  coco : COCO 2017 annotation zip + images downloaded on demand (person -> person, boat -> small_vessel)
  yolo : any dataset already in YOLO format (e.g. a Roboflow "YOLOv8" export). Bollard / fender /
         buoy / quay-edge data come in through this route.

Examples
  # COCO subset (train 1200 images per class, val 150)
  python ai/prepare_dataset.py --out datasets/avm6 --coco-zip annotations_trainval2017.zip \
        --coco-train-per-class 1200 --coco-val-per-class 150

  # add a Roboflow export, mapping its class names to ours
  python ai/prepare_dataset.py --out datasets/avm6 --append \
        --yolo ~/Downloads/bollard-3 --yolo-map "bollard:bollard,Bollard:bollard"

Licences: check each dataset's licence before use / delivery (docs/ai_dataset_plan.md).
"""
import argparse
import json
import random
import shutil
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CLASSES = yaml.safe_load(open(ROOT / "classes.yaml"))["names"]        # {0: person, ...}
NAME_TO_ID = {v: k for k, v in CLASSES.items()}

COCO_MAP = {"person": "person", "boat": "small_vessel"}
COCO_URL = "http://images.cocodataset.org/{split}/{name}"
MIN_BOX_PX = 12                                                        # ignore tiny boxes


def yolo_line(cls_id, x, y, w, h, img_w, img_h):
    cx, cy = (x + w / 2) / img_w, (y + h / 2) / img_h
    return f"{cls_id} {cx:.6f} {cy:.6f} {w / img_w:.6f} {h / img_h:.6f}"


def download(url, dst):
    if dst.exists() and dst.stat().st_size > 0:
        return True
    try:
        with urllib.request.urlopen(url, timeout=30) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f)
        return True
    except Exception as e:                                             # noqa: BLE001
        print(f"  download failed {url}: {e}", file=sys.stderr)
        dst.unlink(missing_ok=True)
        return False


def add_coco(zip_path, out, split, per_class, seed):
    """split: 'train' or 'val'. Picks up to `per_class` images per mapped class."""
    inner = f"annotations/instances_{split}2017.json"
    print(f"[coco] reading {inner} ...")
    with zipfile.ZipFile(zip_path) as z:
        data = json.load(z.open(inner))
    cat_name = {c["id"]: c["name"] for c in data["categories"]}
    images = {i["id"]: i for i in data["images"]}
    per_image = {}                                                     # image id -> [(avm id, x, y, w, h)]
    for a in data["annotations"]:
        name = COCO_MAP.get(cat_name[a["category_id"]])
        if name is None or a.get("iscrowd", 0):
            continue
        x, y, w, h = a["bbox"]
        if w < MIN_BOX_PX or h < MIN_BOX_PX:
            continue
        per_image.setdefault(a["image_id"], []).append((NAME_TO_ID[name], x, y, w, h))

    rng = random.Random(seed)
    chosen = set()
    for name in set(COCO_MAP.values()):
        cid = NAME_TO_ID[name]
        ids = sorted(i for i, anns in per_image.items() if any(a[0] == cid for a in anns))
        rng.shuffle(ids)
        chosen.update(ids[:per_class])
        print(f"[coco/{split}] {name}: {len(ids)} candidate images, taking {min(per_class, len(ids))}")

    img_dir = out / "images" / split
    lab_dir = out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    def job(iid):
        info = images[iid]
        stem = f"coco_{iid:012d}"
        if not download(COCO_URL.format(split=f"{split}2017", name=info["file_name"]), img_dir / f"{stem}.jpg"):
            return False
        lines = [yolo_line(c, x, y, w, h, info["width"], info["height"]) for c, x, y, w, h in per_image[iid]]
        (lab_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        return True

    with ThreadPoolExecutor(16) as ex:
        ok = sum(ex.map(job, sorted(chosen)))
    print(f"[coco/{split}] {ok}/{len(chosen)} images ready")


def add_yolo(src, out, name_map, val_fraction, seed):
    """Convert a YOLO-format dataset (with a data.yaml listing its class names)."""
    src = Path(src)
    cfg = yaml.safe_load(open(next(src.glob("data.y*ml"))))
    names = cfg["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    id_map = {i: NAME_TO_ID[name_map[n]] for i, n in enumerate(names) if n in name_map}
    print(f"[yolo] {src.name}: source classes {names}\n       mapped: "
          + ", ".join(f"{names[i]}->{CLASSES[j]}" for i, j in id_map.items()))
    if not id_map:
        sys.exit("no class of this dataset is mapped: check --yolo-map")

    rng = random.Random(seed)
    n_img = 0
    for split in ("train", "valid", "val", "test"):
        idir = src / split / "images"
        if not idir.exists():
            idir = src / "images" / split
        ldir = idir.parent / "labels" if idir.parent.name == split else src / "labels" / split
        if not idir.exists():
            continue
        for img in sorted(idir.glob("*.*")):
            lab = ldir / f"{img.stem}.txt"
            if not lab.exists():
                continue
            lines = []
            for ln in lab.read_text().splitlines():
                p = ln.split()
                if len(p) >= 5 and int(p[0]) in id_map:
                    lines.append(" ".join([str(id_map[int(p[0])])] + p[1:5]))
            if not lines:
                continue
            target = "val" if (split in ("valid", "val", "test") or rng.random() < val_fraction) else "train"
            (out / "images" / target).mkdir(parents=True, exist_ok=True)
            (out / "labels" / target).mkdir(parents=True, exist_ok=True)
            stem = f"{src.name}_{img.stem}"
            shutil.copy(img, out / "images" / target / f"{stem}{img.suffix}")
            (out / "labels" / target / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            n_img += 1
    print(f"[yolo] {n_img} images added")


def write_yaml(out):
    doc = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
           "names": {int(k): v for k, v in CLASSES.items()}}
    yaml.safe_dump(doc, open(out / "data.yaml", "w"), sort_keys=False)
    # class statistics
    for split in ("train", "val"):
        counts = {v: 0 for v in CLASSES.values()}
        n_img = len(list((out / "labels" / split).glob("*.txt"))) if (out / "labels" / split).exists() else 0
        for f in (out / "labels" / split).glob("*.txt") if (out / "labels" / split).exists() else []:
            for ln in f.read_text().splitlines():
                counts[CLASSES[int(ln.split()[0])]] += 1
        print(f"[{split}] {n_img} images, boxes per class: {counts}")
        empty = [k for k, v in counts.items() if v == 0]
        if empty and split == "train":
            print(f"        WARNING: no training data yet for: {', '.join(empty)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true", help="keep existing dataset content")
    ap.add_argument("--coco-zip")
    ap.add_argument("--coco-train-per-class", type=int, default=1200)
    ap.add_argument("--coco-val-per-class", type=int, default=150)
    ap.add_argument("--yolo", action="append", default=[], help="YOLO-format dataset folder (repeatable)")
    ap.add_argument("--yolo-map", action="append", default=[], help='"source name:avm name,..." per --yolo')
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() and not a.append:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    if a.coco_zip:
        add_coco(a.coco_zip, out, "val", a.coco_val_per_class, a.seed)
        add_coco(a.coco_zip, out, "train", a.coco_train_per_class, a.seed)
    if len(a.yolo) != len(a.yolo_map):
        sys.exit("--yolo and --yolo-map must be given in pairs")
    for src, mp in zip(a.yolo, a.yolo_map):
        m = dict(kv.split(":", 1) for kv in mp.split(","))
        bad = [v for v in m.values() if v not in NAME_TO_ID]
        if bad:
            sys.exit(f"unknown AVM class in map: {bad}; valid: {list(NAME_TO_ID)}")
        add_yolo(src, out, m, a.val_fraction, a.seed)
    write_yaml(out)


if __name__ == "__main__":
    main()
