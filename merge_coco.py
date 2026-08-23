"""Merge several PoseAnything/COCO keypoint JSONs into one.

Use it to combine the 4-character cartoon set with the Tom & Jerry set (and any
future character) into a single dataset for training / leave-one-character-out.

  * categories are unioned BY NAME onto a canonical order, so a character keeps
    ONE stable id no matter which file it came from (tom=1, jerry=2, ...);
  * image_id and annotation_id are re-indexed to avoid collisions;
  * duplicate file_names across inputs are detected and reported.

Usage:
  python merge_coco.py --out data/merged/coco_train.json \
      data/cartoon_yolo/coco_train.json /path/to/tom_jerry/coco_train.json
"""
import argparse
import json
import random
from collections import Counter, defaultdict

# canonical character -> id (must match your data.yaml order + cat_offset=1).
CANONICAL = {
    "tom": 1, "jerry": 2,
    "bugs_bunny": 3, "pink_panther": 4, "sylvester": 5, "mickey_mouse": 6,
}


def merge(inputs, out_path, max_per_cat=None, seed=0):
    id2name_canon = {v: k for k, v in CANONICAL.items()}
    categories, seen_files = [], {}
    images, annotations = [], []
    next_img_id, next_ann_id = 1, 1
    per_cat = Counter()

    # build the canonical category list once (only those that appear)
    used_names = set()

    for path in inputs:
        d = json.load(open(path, encoding="utf-8"))
        local_id2name = {c["id"]: c["name"] for c in d["categories"]}
        # keep skeleton/keypoints metadata from the first file that defines a name
        for c in d["categories"]:
            if c["name"] not in used_names and c["name"] in CANONICAL:
                used_names.add(c["name"])

        img_remap = {}
        for im in d["images"]:
            fn = im["file_name"]
            if fn in seen_files:
                print(f"  ! duplicate file_name across inputs: {fn} (from {path})")
            seen_files[fn] = path
            new = dict(im)
            new["id"] = next_img_id
            img_remap[im["id"]] = next_img_id
            next_img_id += 1
            images.append(new)

        for an in d["annotations"]:
            name = local_id2name[an["category_id"]]
            if name not in CANONICAL:
                print(f"  ! unknown category '{name}' skipped")
                continue
            new = dict(an)
            new["id"] = next_ann_id
            new["image_id"] = img_remap[an["image_id"]]
            new["category_id"] = CANONICAL[name]
            next_ann_id += 1
            annotations.append(new)
            per_cat[name] += 1

    # optional balancing: cap each category to at most max_per_cat annotations
    if max_per_cat:
        random.seed(seed)
        by_cat = defaultdict(list)
        for i, an in enumerate(annotations):
            by_cat[an["category_id"]].append(i)
        keep = set()
        for cat, idxs in by_cat.items():
            random.shuffle(idxs)
            keep.update(idxs[:max_per_cat])
        annotations = [an for i, an in enumerate(annotations) if i in keep]
        kept_imgs = {an["image_id"] for an in annotations}
        images = [im for im in images if im["id"] in kept_imgs]
        per_cat = Counter(id2name_canon[an["category_id"]] for an in annotations)
        print(f"  [balanced] capped each character to <= {max_per_cat} annotations")

    # rebuild categories in canonical id order, reusing metadata from inputs
    meta_by_name = {}
    for path in inputs:
        d = json.load(open(path, encoding="utf-8"))
        for c in d["categories"]:
            meta_by_name.setdefault(c["name"], c)
    # only emit categories that actually have annotations (avoid empty-cat crash)
    surviving = sorted(set(per_cat), key=lambda n: CANONICAL[n])
    for name in surviving:
        src = meta_by_name[name]
        categories.append({
            "id": CANONICAL[name], "name": name,
            "supercategory": src.get("supercategory", "cartoon"),
            "keypoints": src.get("keypoints", []),
            "skeleton": src.get("skeleton", []),
        })

    import os
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump({"images": images, "annotations": annotations,
               "categories": categories}, open(out_path, "w", encoding="utf-8"))

    print(f"[done] {out_path}")
    print(f"  images={len(images)}  annotations={len(annotations)}  "
          f"categories={len(categories)}")
    print("  category ids:", {c["id"]: c["name"] for c in categories})
    print("  instances per character:", dict(per_cat))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-cat", type=int, default=None,
                    help="cap each character to at most N annotations (balancing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("inputs", nargs="+", help="COCO jsons to merge")
    a = ap.parse_args()
    merge(a.inputs, a.out, a.max_per_cat, a.seed)
