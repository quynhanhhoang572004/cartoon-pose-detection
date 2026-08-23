"""CVAT (skeleton-only) -> Ultralytics YOLO-pose, reading images straight from a zip.

Tailored for the 4-character cartoon set (bugs_bunny / pink_panther / sylvester /
mickey_mouse), annotated in CVAT with skeletons only (NO bounding boxes, NO class
labels). This converter:

  * reads image bytes directly from the CVAT image zip,
  * DROPS every image that has no skeleton (the ones whose tag you deleted),
  * derives the class from the FILENAME PREFIX (so per-character categories are
    kept for the few-shot / leave-one-character-out setup),
  * DERIVES the bbox from the visible keypoints (+padding), because PoseAnything's
    loader discards any annotation without a valid bbox,
  * splits train/val PER CHARACTER so each character appears in both.

Output layout (Ultralytics YOLO-pose):
    <output_dir>/images/{train,val}/*.jpg
    <output_dir>/labels/{train,val}/*.txt      # cls xc yc w h  (x y v)*21  (normalized)
    <output_dir>/data.yaml

Then feed it to PoseAnything:
    python PoseAnything/tools/yolo2coco_pose.py --data-yaml <output_dir>/data.yaml --split train --out ...
"""
import argparse
import random
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

EXPECTED_ORDER = [
    "head", "nose", "ear_left", "ear_right", "eye_left", "eye_right", "neck",
    "shoulder_left", "shoulder_right", "hip_left", "hip_right",
    "elbow_left", "elbow_right", "hand_left", "hand_right",
    "knee_left", "knee_right", "foot_left", "foot_right", "tail_base", "tail_tip",
]

# character -> class id. IDs continue after tom=0/jerry=1 so this set stays
# mergeable with the Tom & Jerry dataset. Edit here if your scheme differs.
CHARACTER_TO_CLASS = {
    "tom": 0, "jerry": 1,
    "bugs_bunny": 2, "pink_panther": 3, "sylvester": 4, "mickey_mouse": 5,
}
# names list ordered by id (index == class id)
NAMES = [c for c, _ in sorted(CHARACTER_TO_CLASS.items(), key=lambda kv: kv[1])]

SKELETON_LABEL = "character_with_ears"


def character_of(filename):
    """Map 'bugs_bunny_v2_00101.jpg' -> 'bugs_bunny' via longest prefix match."""
    stem = Path(filename).name
    best = None
    for char in CHARACTER_TO_CLASS:
        if stem.startswith(char + "_") and (best is None or len(char) > len(best)):
            best = char
    return best


def parse_skeleton(skeleton_elem, W, H):
    """Return (kpts_flat[63], n_visible) reordered to EXPECTED_ORDER.

    Visibility follows CVAT semantics:
        outside=1 -> 0 (not labeled)   occluded=1 -> 1 (occluded)   else -> 2 (visible)
    """
    pts = {}
    for p in skeleton_elem.findall(".//points"):
        label = p.get("label")
        coords = p.get("points")
        if not coords:
            continue
        x, y = map(float, coords.split(","))
        if p.get("outside") == "1":
            pts[label] = (0.0, 0.0, 0)
        elif p.get("occluded") == "1":
            pts[label] = (min(1.0, max(0.0, x / W)), min(1.0, max(0.0, y / H)), 1)
        else:
            pts[label] = (min(1.0, max(0.0, x / W)), min(1.0, max(0.0, y / H)), 2)

    flat, n_vis = [], 0
    for name in EXPECTED_ORDER:
        if name in pts:
            flat.extend(pts[name])
            if pts[name][2] > 0:
                n_vis += 1
        else:
            flat.extend([0.0, 0.0, 0])
    return flat, n_vis


def bbox_from_kpts(flat, pad=0.2, min_size=0.03):
    """Normalized cxcywh bbox from VISIBLE keypoints only (+padding)."""
    xs, ys = [], []
    for k in range(len(EXPECTED_ORDER)):
        x, y, v = flat[3 * k:3 * k + 3]
        if v > 0:
            xs.append(x)
            ys.append(y)
    if len(xs) < 1:
        return None
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    w, h = x1 - x0, y1 - y0
    x0 -= w * pad; x1 += w * pad
    y0 -= h * pad; y1 += h * pad
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(1.0, x1), min(1.0, y1)
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = max(x1 - x0, min_size), max(y1 - y0, min_size)
    return xc, yc, bw, bh


def convert(xml_path, images_zip, output_dir, train_ratio=0.8, pad=0.2, seed=0):
    random.seed(seed)
    root = ET.parse(xml_path).getroot()
    out = Path(output_dir)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    zf = zipfile.ZipFile(images_zip)
    zip_names = set(zf.namelist())

    stats = defaultdict(int)
    per_char_images = defaultdict(list)   # char -> list of (fname, [label_lines])

    for image in root.findall(".//image"):
        fname = image.get("name")
        W, H = int(image.get("width")), int(image.get("height"))

        skeletons = image.findall(f'.//skeleton[@label="{SKELETON_LABEL}"]')
        if not skeletons:
            stats["dropped_no_skeleton"] += 1          # tag you deleted -> skip image
            continue

        char = character_of(fname)
        if char is None:
            print(f"  ! unknown character prefix, skipped: {fname}")
            stats["dropped_unknown_char"] += 1
            continue
        cls = CHARACTER_TO_CLASS[char]

        label_lines = []
        for sk in skeletons:
            flat, n_vis = parse_skeleton(sk, W, H)
            if n_vis < 2:
                stats["dropped_skeleton_too_few_kpts"] += 1
                continue
            bb = bbox_from_kpts(flat, pad=pad)
            if bb is None:
                stats["dropped_skeleton_no_bbox"] += 1
                continue
            xc, yc, bw, bh = bb
            kp = " ".join(
                f"{v:.6f}" if i % 3 != 2 else str(int(v))
                for i, v in enumerate(flat))
            label_lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {kp}")

        if not label_lines:
            stats["dropped_no_valid_skeleton"] += 1
            continue
        if fname not in zip_names:
            print(f"  ! image not in zip, skipped: {fname}")
            stats["dropped_missing_image"] += 1
            continue

        per_char_images[char].append((fname, label_lines))
        stats["total_skeletons"] += len(label_lines)

    # split per character so each appears in both train and val
    for char, items in per_char_images.items():
        random.shuffle(items)
        k = int(len(items) * train_ratio)
        for split, chunk in (("train", items[:k]), ("val", items[k:])):
            for fname, lines in chunk:
                (out / "images" / split / Path(fname).name).write_bytes(zf.read(fname))
                (out / "labels" / split / f"{Path(fname).stem}.txt").write_text(
                    "\n".join(lines), encoding="utf-8")
                stats[f"images_{split}"] += 1

    write_yaml(out)
    print_report(stats, per_char_images, train_ratio)


def write_yaml(out):
    names_yaml = "\n".join(f"  - {n}" for n in NAMES)
    kpt_yaml = "\n".join(f"  - {n}" for n in EXPECTED_ORDER)
    yaml = f"""path: {out.absolute()}
train: images/train
val: images/val

nc: {len(NAMES)}
names:
{names_yaml}

kpt_shape: [21, 3]
kpt_label:
{kpt_yaml}

skeleton:
  - [0, 1]
  - [0, 2]
  - [0, 3]
  - [1, 4]
  - [1, 5]
  - [1, 6]
  - [6, 7]
  - [6, 8]
  - [7, 9]
  - [8, 10]
  - [9, 10]
  - [7, 11]
  - [8, 12]
  - [11, 13]
  - [12, 14]
  - [9, 15]
  - [10, 16]
  - [15, 17]
  - [16, 18]
  - [19, 20]
  - [9, 19]

flip_idx: [0, 1, 3, 2, 5, 4, 6, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 19, 20]
"""
    (out / "data.yaml").write_text(yaml, encoding="utf-8")


def print_report(stats, per_char_images, train_ratio):
    print("\n=== conversion report ===")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print("\n  images kept per character (train+val):")
    for char in sorted(per_char_images):
        n = len(per_char_images[char])
        tr = int(n * train_ratio)
        flag = "  <-- WARNING: < num_shots+1" if n < 6 else ""
        print(f"    {char:<14} total={n:<4} train={tr:<4} val={n - tr}{flag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="data/annotations (1).xml")
    ap.add_argument("--zip", default="data/label_set_cvat.zip")
    ap.add_argument("--out", default="data/cartoon_yolo")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--pad", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    convert(a.xml, a.zip, a.out, a.train_ratio, a.pad, a.seed)
    print("\nDone.")
