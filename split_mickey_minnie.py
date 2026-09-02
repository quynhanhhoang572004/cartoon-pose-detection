"""Split the merged 'mickey_mouse' category into mickey_mouse + minnie_mouse,
DIRECTLY on the already-merged COCO json (no need to rerun the CVAT pipeline).

Because the CVAT skeletons share one label and every file is named
'mickey_mouse_*', nothing in the data distinguishes Mickey from Minnie. So the
split needs a MANUAL list of the images that are actually Minnie.

Workflow
--------
1) dump the mickey filenames so you can mark which are Minnie:
     python split_mickey_minnie.py list \
         --ann data/merged/coco_train.json > mickey_train.txt
     python split_mickey_minnie.py list \
         --ann data/merged/coco_val.json   > mickey_val.txt

   Open each .txt, DELETE the lines that are Mickey, keep only the Minnie
   filenames (one per line). Those remaining lines = the Minnie list.

2) apply the split (run for train and val separately):
     python split_mickey_minnie.py split \
         --ann data/merged/coco_train.json \
         --minnie mickey_train.txt \
         --out data/merged/coco_train.json          # overwrite in place
     python split_mickey_minnie.py split \
         --ann data/merged/coco_val.json \
         --minnie mickey_val.txt \
         --out data/merged/coco_val.json

Then set valid_class_ids=[1,2,3,4,5,6,7] in the train/eval configs.
"""
import argparse
import json
from pathlib import Path

MICKEY_ID = 6          # canonical id of mickey_mouse in your merged json
MINNIE_ID = 7          # new id for minnie_mouse
MINNIE_NAME = "minnie_mouse"


def cmd_list(a):
    d = json.load(open(a.ann, encoding="utf-8"))
    id2name = {im["id"]: im["file_name"] for im in d["images"]}
    # every image that has at least one mickey annotation
    files = sorted({id2name[an["image_id"]]
                    for an in d["annotations"] if an["category_id"] == MICKEY_ID})
    for f in files:
        print(f)


def cmd_split(a):
    d = json.load(open(a.ann, encoding="utf-8"))
    minnie_files = {ln.strip() for ln in Path(a.minnie).read_text(
        encoding="utf-8").splitlines() if ln.strip()}

    name2id = {im["file_name"]: im["id"] for im in d["images"]}
    minnie_img_ids = {name2id[f] for f in minnie_files if f in name2id}
    missing = [f for f in minnie_files if f not in name2id]
    if missing:
        print(f"  ! {len(missing)} listed files not in this json (ok if wrong split):")
        for f in missing[:10]:
            print("      ", f)

    # flip category_id 6 -> 7 for annotations on Minnie images
    flipped = 0
    for an in d["annotations"]:
        if an["category_id"] == MICKEY_ID and an["image_id"] in minnie_img_ids:
            an["category_id"] = MINNIE_ID
            flipped += 1

    # add the minnie category, copying mickey's keypoints/skeleton metadata
    cats = {c["id"]: c for c in d["categories"]}
    if MINNIE_ID not in cats and MICKEY_ID in cats:
        mk = cats[MICKEY_ID]
        d["categories"].append({
            "id": MINNIE_ID, "name": MINNIE_NAME,
            "supercategory": mk.get("supercategory", "cartoon"),
            "keypoints": mk.get("keypoints", []),
            "skeleton": mk.get("skeleton", []),
        })
        d["categories"].sort(key=lambda c: c["id"])

    # drop mickey category if nothing is left under it (avoids empty-cat crash)
    left_mickey = sum(1 for an in d["annotations"] if an["category_id"] == MICKEY_ID)
    if left_mickey == 0:
        d["categories"] = [c for c in d["categories"] if c["id"] != MICKEY_ID]

    json.dump(d, open(a.out, "w", encoding="utf-8"))
    from collections import Counter
    per = Counter(an["category_id"] for an in d["annotations"])
    print(f"[done] {a.out}")
    print(f"  flipped {flipped} annotations -> minnie_mouse (id {MINNIE_ID})")
    print(f"  mickey (id {MICKEY_ID}) annotations left: {left_mickey}")
    print(f"  annotations per category id: {dict(sorted(per.items()))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    lp = sub.add_parser("list", help="dump mickey filenames to stdout")
    lp.add_argument("--ann", required=True)
    lp.set_defaults(func=cmd_list)

    sp = sub.add_parser("split", help="flip listed Minnie images to a new category")
    sp.add_argument("--ann", required=True)
    sp.add_argument("--minnie", required=True, help="txt file: one Minnie filename per line")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_split)

    a = ap.parse_args()
    a.func(a)
