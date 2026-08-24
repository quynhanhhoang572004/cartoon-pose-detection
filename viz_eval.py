# -*- coding: utf-8 -*-
"""Visualize the ACTUAL eval predictions (from result_keypoints.json) on the
original images — these are the exact predictions that produced the PCK numbers,
in original-image coordinates. No model / mmcv needed.

  1) run test.py first so it writes result_keypoints.json, e.g.:
       python test.py configs/cartoon/train_merged.py work_dirs/train_new/latest.pth
     (it saves result_keypoints.json in the current dir or the res_folder)
  2) then:
       python viz_eval.py --result result_keypoints.json \
         --ann  /.../data/merged/coco_val.json \
         --img-dir /.../data/merged/images/val \
         --out viz_eval_new --per-cat 4 --score 0.0
"""
import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

SKELETON = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (1, 6), (6, 7), (6, 8),
            (7, 9), (8, 10), (9, 10), (7, 11), (8, 12), (11, 13), (12, 14),
            (9, 15), (10, 16), (15, 17), (16, 18), (19, 20), (9, 19)]


def char_of(fname):
    m = re.match(r'([a-z_]+?)(_v2)?_?\d', Path(fname).name)
    for c in ['bugs_bunny', 'pink_panther', 'mickey_mouse', 'sylvester', 'tom', 'jerry']:
        if Path(fname).name.startswith(c):
            return c
    return m.group(1) if m else 'other'


def draw(img, kpts, vis, radius, thick):
    """Draw only the GT-visible keypoints (the ones PCK actually scores)."""
    pts = np.array(kpts, dtype=float)          # [21, 3] image-space
    for aa, bb in SKELETON:
        if vis[aa] and vis[bb]:
            cv2.line(img, (int(pts[aa][0]), int(pts[aa][1])),
                     (int(pts[bb][0]), int(pts[bb][1])), (0, 200, 0), thick, cv2.LINE_AA)
    for k in range(len(pts)):
        if vis[k]:
            cv2.circle(img, (int(pts[k][0]), int(pts[k][1])), radius, (0, 0, 255), -1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', required=True, help='result_keypoints.json from test.py')
    ap.add_argument('--ann', required=True, help='COCO json (for image_id -> file_name)')
    ap.add_argument('--img-dir', required=True)
    ap.add_argument('--out', default='viz_eval')
    ap.add_argument('--per-cat', type=int, default=4)
    ap.add_argument('--score', type=float, default=0.0, help='min keypoint score to draw')
    ap.add_argument('--scale', type=float, default=0.6,
                    help='line/dot thickness scale (smaller = thinner)')
    a = ap.parse_args()

    res = json.load(open(a.result))
    coco = json.load(open(a.ann))
    id2name = {im['id']: im['file_name'] for im in coco['images']}
    # GT visibility per image -> draw only keypoints eval actually scores
    img2vis = {}
    for an in coco['annotations']:
        v = np.array(an['keypoints']).reshape(-1, 3)[:, 2] > 0
        iid = an['image_id']
        if iid not in img2vis or v.sum() > img2vis[iid].sum():
            img2vis[iid] = v

    # dedup by image_id (an image may be queried in several episodes), group by character
    seen, by_char = set(), defaultdict(list)
    for r in res:
        iid = r['image_id']
        if iid in seen:
            continue
        seen.add(iid)
        fname = id2name.get(iid)
        if fname is None:
            continue
        by_char[char_of(fname)].append((fname, r['keypoints'], iid))

    for char, items in by_char.items():
        out_dir = Path(a.out) / char
        out_dir.mkdir(parents=True, exist_ok=True)
        for fname, kpts, iid in items[:a.per_cat]:
            img = cv2.imread(str(Path(a.img_dir) / fname))
            if img is None:
                print('  ! missing', fname)
                continue
            H, W = img.shape[:2]
            # thin lines, scaled to image size (so consistent across resolutions)
            r = max(1, round(min(H, W) * 0.006 * a.scale))
            t = max(1, round(min(H, W) * 0.003 * a.scale))
            vis = img2vis.get(iid, np.ones(len(kpts), bool))
            drawn = draw(img, kpts, vis, r, t)
            cv2.imwrite(str(out_dir / (Path(fname).stem + '_eval.png')), drawn)
            print(f'[{char}] {fname} done')
    print('DONE ->', a.out)


if __name__ == '__main__':
    main()
