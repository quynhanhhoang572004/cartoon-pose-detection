"""Convert an Ultralytics YOLO-pose dataset into MP-100 / COCO keypoint JSON
that PoseAnything's TransformerPoseDataset can read.

YOLO label line (per instance, all coords normalized 0-1 except visibility):
    cls  xc yc w h  px1 py1 v1  px2 py2 v2  ...  pxK pyK vK

COCO output (what PoseAnything expects):
    - keypoints: flat [x, y, v, ...] in ABSOLUTE pixels
    - bbox:      [x, y, w, h]        in ABSOLUTE pixels
    - categories[i].skeleton: 0-indexed pairs (PoseAnything uses 0-indexed, so
      the skeleton from data.yaml maps through unchanged)

Usage:
    python tools/yolo2coco_pose.py \
        --data-yaml /path/to/data.yaml \
        --split train \
        --out data/mp100/annotations/cartoon_train.json

Notes on the few-shot (novel-character) split:
    The few-shot benchmark splits by CHARACTER, not by image. Two common ways:
      1. Keep separate data.yaml folders per character group and run this
         converter once per group (train / val / test characters disjoint).
      2. Convert everything, then select categories per tap via `valid_class_ids`
         in the PoseAnything config.
    Either way, the train and test CATEGORY sets must not overlap.
"""
import argparse
import json
import os

import yaml
from PIL import Image

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def parse_args():
    p = argparse.ArgumentParser(description='YOLO-pose -> COCO (PoseAnything)')
    p.add_argument('--data-yaml', required=True, help='Ultralytics data.yaml')
    p.add_argument('--split', default='train',
                   help="which split key in the yaml to convert (e.g. train/val)")
    p.add_argument('--out', required=True, help='output COCO json path')
    p.add_argument('--cat-offset', type=int, default=1,
                   help='first category_id (COCO ids usually start at 1)')
    p.add_argument('--img-id-offset', type=int, default=1)
    return p.parse_args()


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def images_dir(root, split_val):
    # split_val may be e.g. "images/train" or an absolute path
    d = split_val if os.path.isabs(split_val) else os.path.join(root, split_val)
    return os.path.normpath(d)


def label_path_for(img_path):
    # Ultralytics convention: .../images/... -> .../labels/... , ext -> .txt
    base = os.path.splitext(img_path)[0] + '.txt'
    parts = base.replace('\\', '/').split('/')
    if 'images' in parts:
        parts[len(parts) - 1 - parts[::-1].index('images')] = 'labels'
    return os.path.normpath('/'.join(parts))


def build_categories(names, skeleton, kpt_labels, cat_offset):
    cats = []
    for i, name in enumerate(names):
        cats.append({
            'id': i + cat_offset,
            'name': name,
            'supercategory': 'cartoon',
            'keypoints': kpt_labels,
            # PoseAnything reads skeleton directly; keep it 0-indexed
            'skeleton': [list(pair) for pair in skeleton],
        })
    return cats


def main():
    args = parse_args()
    cfg = load_yaml(args.data_yaml)

    root = cfg.get('path', os.path.dirname(os.path.abspath(args.data_yaml)))
    names = cfg['names']
    if isinstance(names, dict):  # ultralytics sometimes uses {0: 'tom', ...}
        names = [names[k] for k in sorted(names)]
    num_kpt = cfg['kpt_shape'][0]
    kpt_labels = cfg.get('kpt_label', [f'kp_{i}' for i in range(num_kpt)])
    skeleton = cfg.get('skeleton', [])

    img_root = images_dir(root, cfg[args.split])
    if not os.path.isdir(img_root):
        raise FileNotFoundError(f'image dir not found: {img_root}')

    categories = build_categories(names, skeleton, kpt_labels, args.cat_offset)

    images, annotations = [], []
    img_id = args.img_id_offset
    ann_id = 1
    n_skipped = 0

    files = sorted(f for f in os.listdir(img_root)
                   if f.lower().endswith(IMG_EXTS))
    for fname in files:
        img_path = os.path.join(img_root, fname)
        lbl_path = label_path_for(img_path)
        with Image.open(img_path) as im:
            W, H = im.size

        images.append({'id': img_id, 'file_name': fname,
                       'width': W, 'height': H})

        if os.path.exists(lbl_path):
            with open(lbl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    vals = line.split()
                    if not vals:
                        continue
                    cls = int(float(vals[0]))
                    xc, yc, bw, bh = map(float, vals[1:5])
                    kpt_vals = list(map(float, vals[5:]))
                    if len(kpt_vals) != num_kpt * 3:
                        n_skipped += 1
                        continue

                    # bbox: normalized cxcywh -> absolute xywh
                    x = (xc - bw / 2) * W
                    y = (yc - bh / 2) * H
                    w = bw * W
                    h = bh * H

                    kpts = []
                    n_vis = 0
                    for k in range(num_kpt):
                        px, py, v = kpt_vals[3 * k:3 * k + 3]
                        vi = int(round(v))
                        if vi > 0:
                            kpts += [px * W, py * H, vi]
                            n_vis += 1
                        else:
                            kpts += [0, 0, 0]

                    annotations.append({
                        'id': ann_id,
                        'image_id': img_id,
                        'category_id': cls + args.cat_offset,
                        'keypoints': kpts,
                        'num_keypoints': n_vis,
                        'bbox': [x, y, w, h],
                        'area': w * h,
                        'iscrowd': 0,
                    })
                    ann_id += 1
        img_id += 1

    coco = {'images': images, 'annotations': annotations,
            'categories': categories}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(coco, f)

    print(f'[done] {args.out}')
    print(f'  images={len(images)}  annotations={len(annotations)}  '
          f'categories={len(categories)}  skipped_lines={n_skipped}')
    per_cat = {c['name']: 0 for c in categories}
    id2name = {c['id']: c['name'] for c in categories}
    for a in annotations:
        per_cat[id2name[a['category_id']]] += 1
    print('  instances per character:', per_cat)


if __name__ == '__main__':
    main()
