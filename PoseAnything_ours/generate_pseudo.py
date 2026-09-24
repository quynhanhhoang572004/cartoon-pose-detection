r"""DRAFT for review — generate pseudo-labels for UNLABELED cartoon frames using
the SUPPORT-ENSEMBLE agreement filter (our CAPE-specific noise filter).

Idea
----
For each unlabeled query frame, run the model with N DIFFERENT support subsets
drawn from the ~20 labeled images of that character. A predicted keypoint is
KEPT only if the N subsets AGREE on its location (spread < --eps). Keypoints the
subsets disagree on are dropped (marked invisible). Frames with fewer than
--min-visible agreed keypoints are skipped. Only CAPE can do this, because only
CAPE takes a variable "support" as input.

Output: a COCO json of pseudo-labels (in ORIGINAL query-image pixel coords) that
merge_coco.py can combine with the real labels for self-training.

  python generate_pseudo.py \
    --config configs/cartoon/train_merged.py \
    --checkpoint work_dirs/train_supcon_100ep/best_PCK_epoch_70.pth \
    --support-coco /.../data/merged/coco_val.json \
    --support-img-dir /.../data/merged/images/val \
    --category 4 \                               # pink_panther id in the COCO
    --unlabeled-dir /.../data/pink_panther_batch2 \
    --out /.../data/pseudo_pink.json \
    --n-subsets 3 --shots 5 --eps 0.05 --min-visible 8
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmpose.models import build_posenet
from torchvision import transforms

from models import *  # noqa: F401,F403  (registers custom classes)
from models.datasets.pipelines.top_down_transform import TopDownGenerateTargetFewShot
# reuse the exact crop / pad helpers the model was trained/evaluated with
from demo_batch import SKELETON, Resize_Pad, kpts_to_pad, crop_char


# ----- coordinate helper: undo the whole-image pad+resize (256 -> original px)
def pad_to_orig(pts, H, W):
    p = np.asarray(pts, dtype=float).copy()
    if W >= H:                      # landscape: padded top/bottom, scaled by 256/W
        p *= W / 256.0
        p[:, 1] -= (W - H) / 2.0
    else:                           # portrait: padded left/right, scaled by 256/H
        p *= H / 256.0
        p[:, 0] -= (H - W) / 2.0
    return p


# ----- build each labeled support image's model tensors ONCE (reused per subset)
def build_support_pool(anns, img_dir, gen, data_cfg, preprocess, device):
    pool = []
    for an in anns:
        img = cv2.imread(str(Path(img_dir) / an['file_name']))
        if img is None:
            continue
        kp = np.array(an['keypoints']).reshape(-1, 3)
        crop, ox, oy = crop_char(img, kp)                 # crop to the character bbox
        ch, cw = crop.shape[:2]
        vis = kp[:, 2] > 0
        kp_c = torch.tensor(kp[:, :2] - np.array([ox, oy])).float()
        kp_pad = kpts_to_pad(kp_c, ch, cw)
        kp3d = torch.cat([kp_pad, torch.zeros(kp_pad.shape[0], 1)], -1)
        w = torch.tensor(vis).float()[:, None]
        w3 = torch.cat([w, w, torch.zeros_like(w)], -1)
        t, tw = gen._msra_generate_target(data_cfg, kp3d.numpy(), w3.numpy(), sigma=1)
        pool.append({
            'img': preprocess(crop).flip(0)[None].to(device),
            'tgt': torch.tensor(t).float()[None].to(device),
            'w': torch.tensor(tw).float()[None].to(device),
            'kp3d': kp3d,
            'cen': kp3d[:, :2].mean(0),
            'scl': kp3d[:, :2].max(0)[0] - kp3d[:, :2].min(0)[0],
        })
    return pool


# ----- one forward pass: (a support subset, one query) -> 21 keypoints in 256 space
def predict(model, subset, q_t, skeleton, device):
    K = len(subset)
    data = {
        'img_s': [e['img'] for e in subset], 'img_q': q_t,
        'target_s': [e['tgt'] for e in subset],
        'target_weight_s': [e['w'] for e in subset],
        'target_q': None, 'target_weight_q': None, 'return_loss': False,
        'img_metas': [{'sample_skeleton': [skeleton] * K, 'query_skeleton': skeleton,
                       'sample_joints_3d': [e['kp3d'] for e in subset],
                       'query_joints_3d': subset[0]['kp3d'],
                       'sample_center': [e['cen'] for e in subset], 'query_center': subset[0]['cen'],
                       'sample_scale': [e['scl'] for e in subset], 'query_scale': subset[0]['scl'],
                       'sample_rotation': [0] * K, 'query_rotation': 0,
                       'sample_bbox_score': [1] * K, 'query_bbox_score': 1,
                       'query_image_file': '', 'sample_image_file': [''] * K}]}
    with torch.no_grad():
        out = model(**data)
    return np.array(torch.as_tensor(out['points']).squeeze().cpu()).reshape(-1, 2)[:21]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--support-coco', required=True, help='COCO json holding the labeled support set')
    ap.add_argument('--support-img-dir', required=True)
    ap.add_argument('--category', type=int, required=True, help='character category id in the COCO')
    ap.add_argument('--unlabeled-dir', required=True, help='folder of unlabeled query frames')
    ap.add_argument('--out', required=True, help='output pseudo-label COCO json')
    ap.add_argument('--n-subsets', type=int, default=3, help='N independent support subsets to cross-check')
    ap.add_argument('--shots', type=int, default=5, help='support images per subset')
    ap.add_argument('--eps', type=float, default=0.05,
                    help='agreement radius as a fraction of 256 (0.05 = ~13px)')
    ap.add_argument('--min-visible', type=int, default=8, help='min agreed keypoints to keep a frame')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    rng = random.Random(a.seed)
    eps_px = a.eps * 256.0

    cfg = Config.fromfile(a.config)
    imgsz = cfg.model.encoder_config.img_size
    model = build_posenet(cfg.model)
    load_checkpoint(model, a.checkpoint, map_location='cpu')
    model.eval().to(a.device)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        Resize_Pad(imgsz, imgsz)])
    gen = TopDownGenerateTargetFewShot()
    data_cfg = dict(cfg.data_cfg)
    data_cfg['image_size'] = np.array([imgsz, imgsz])
    data_cfg['joint_weights'] = None
    data_cfg['use_different_joint_weights'] = False

    # labeled support pool for this character
    coco = json.load(open(a.support_coco))
    id2name = {im['id']: im['file_name'] for im in coco['images']}
    anns = [dict(an, file_name=id2name[an['image_id']])
            for an in coco['annotations'] if an['category_id'] == a.category]
    if len(anns) < a.shots:
        raise SystemExit(f'need >= {a.shots} labeled support, have {len(anns)}')
    pool = build_support_pool(anns, a.support_img_dir, gen, data_cfg, preprocess, a.device)
    skeleton = next(c['skeleton'] for c in coco['categories'] if c['id'] == a.category)

    frames = sorted(p for ext in ('*.jpg', '*.png') for p in Path(a.unlabeled_dir).glob(ext))
    out_images, out_anns = [], []
    iid = aid = 1
    kept = skipped = 0
    for fp in frames:
        q_img = cv2.imread(str(fp))
        if q_img is None:
            continue
        H, W = q_img.shape[:2]
        q_t = preprocess(q_img).flip(0)[None].to(a.device)

        # --- SUPPORT-ENSEMBLE: predict with N different support subsets ---
        preds = []
        for _ in range(a.n_subsets):
            subset = rng.sample(pool, a.shots)
            preds.append(predict(model, subset, q_t, skeleton, a.device))
        preds = np.stack(preds, 0)                    # [N, 21, 2] in 256 space

        # --- AGREEMENT FILTER: keep a keypoint only if the subsets agree ---
        mean = preds.mean(0)                           # [21, 2]
        spread = np.linalg.norm(preds - mean[None], axis=2).max(0)   # [21] worst deviation
        visible = spread < eps_px                      # True = subsets agree -> trust
        if visible.sum() < a.min_visible:
            skipped += 1
            continue

        # map the agreed points back to ORIGINAL query pixels; drop the rest
        pts_orig = pad_to_orig(mean, H, W)
        kpts = []
        for k in range(21):
            if visible[k]:
                kpts += [float(pts_orig[k, 0]), float(pts_orig[k, 1]), 2]
            else:
                kpts += [0.0, 0.0, 0]                  # dropped -> invisible

        out_images.append({'id': iid, 'file_name': fp.name, 'width': W, 'height': H})
        out_anns.append({'id': aid, 'image_id': iid, 'category_id': a.category,
                         'keypoints': kpts, 'num_keypoints': int(visible.sum()),
                         'is_pseudo': 1})
        iid += 1; aid += 1; kept += 1

    out = {'images': out_images, 'annotations': out_anns,
           'categories': [c for c in coco['categories'] if c['id'] == a.category]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, 'w'))
    print(f'[done] kept {kept} pseudo-labeled frames, skipped {skipped} (low agreement)')
    print(f'       -> {a.out}')


if __name__ == '__main__':
    main()
