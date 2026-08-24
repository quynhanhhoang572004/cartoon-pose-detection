"""Batch prediction visualizer (no clicking) for qualitative results.

Runs the model on a few (support, query) pairs sampled from a COCO val file,
using the GT keypoints of the support image as the support annotation, and saves
the predicted keypoints drawn on the query image (one subfolder per character).

Run on the SERVER (needs mmcv/mmpose/GPU). Works for whichever repo you launch it
from (old PoseAnything or PoseAnything_ours) — the arch comes from that repo's
registry, so `python setup.py develop` in the repo first.

  python demo_batch.py \
    --config configs/cartoon/train_merged.py \
    --checkpoint work_dirs/train_new/latest.pth \
    --ann /home/subnh3/projects/QuynhAnh/cartoon-pose-detection/data/merged/coco_val.json \
    --img-dir /home/subnh3/projects/QuynhAnh/cartoon-pose-detection/data/merged/images/val \
    --outdir viz_pred_new --per-cat 3
"""
import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmpose.models import build_posenet
from torchvision import transforms

from models import *  # noqa: registers custom classes
from models.datasets.pipelines.top_down_transform import TopDownGenerateTargetFewShot
from tools.visualization import plot_results

# 21-kpt skeleton (0-indexed), identical to data.yaml
SKELETON = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (1, 6), (6, 7), (6, 8),
            (7, 9), (8, 10), (9, 10), (7, 11), (8, 12), (11, 13), (12, 14),
            (9, 15), (10, 16), (15, 17), (16, 18), (19, 20), (9, 19)]


class Resize_Pad:
    def __init__(self, w=256, h=256):
        self.w, self.h = w, h

    def __call__(self, image):
        _, w_1, h_1 = image.shape
        if round(w_1 / h_1, 2) != 1:
            if w_1 / h_1 > 1:
                hp = int(w_1 - h_1) // 2
                image = F.pad(image, (hp, 0, hp, 0), 0, "constant")
            else:
                wp = int(h_1 - w_1) // 2
                image = F.pad(image, (0, wp, 0, wp), 0, "constant")
        return F.resize(image, [self.h, self.w])


def kpts_to_pad(keypoints, H, W):
    """Map keypoints (image px, [N,2]) into the Resize_Pad 256 space."""
    kp = keypoints.clone()
    if W / H > 1:
        hp = int(W - H) // 2
        kp[:, 1] = keypoints[:, 1] + hp
        kp *= (256. / W)
    else:
        wp = int(H - W) // 2
        kp[:, 0] = keypoints[:, 0] + wp
        kp *= (256. / H)
    return kp


def resize_pad_raw(img_bgr):
    """Raw uint8 RGB 256x256 for display (same pad/resize as the model input)."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    if W / H > 1:
        hp = (W - H) // 2
        img = cv2.copyMakeBorder(img, hp, hp, 0, 0, cv2.BORDER_CONSTANT, value=0)
    elif H / W > 1:
        wp = (H - W) // 2
        img = cv2.copyMakeBorder(img, 0, 0, wp, wp, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(img, (256, 256))


def crop_char(img_bgr, kp, pad=0.25):
    """Crop the image to the character's bbox (from visible keypoints) + padding —
    matches the top-down crop the model was TRAINED on. Returns crop + (ox, oy)."""
    vis = kp[:, 2] > 0
    if vis.sum() < 2:
        return img_bgr, 0, 0
    xs, ys = kp[vis, 0], kp[vis, 1]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0, y1 - y0
    x0 -= bw * pad; x1 += bw * pad
    y0 -= bh * pad; y1 += bh * pad
    H, W = img_bgr.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(W, int(x1)), min(H, int(y1))
    if x1 - x0 < 5 or y1 - y0 < 5:
        return img_bgr, 0, 0
    return img_bgr[y0:y1, x0:x1], x0, y0


def norm_pose(ann):
    """Bbox-normalized keypoints [21,2] in [0,1] + visibility, for pose comparison."""
    kp = np.array(ann['keypoints']).reshape(-1, 3).astype(float)
    vis = kp[:, 2] > 0
    if vis.sum() < 2:
        return None, vis
    xy = kp[:, :2]
    x0, y0 = xy[vis, 0].min(), xy[vis, 1].min()
    s = max(xy[vis, 0].max() - x0, xy[vis, 1].max() - y0) + 1e-6
    return (xy - np.array([x0, y0])) / s, vis


def pose_dist(qa, sa):
    """Mean L2 between two normalized poses over commonly-visible keypoints."""
    q, qv = norm_pose(qa)
    s, sv = norm_pose(sa)
    if q is None or s is None:
        return 1e9
    m = qv & sv
    if m.sum() < 3:
        return 1e9
    return float(np.linalg.norm(q[m] - s[m], axis=1).mean())


def draw_pose(img256_rgb, pts, vis, radius, thick):
    """Draw skeleton (green) + keypoints (red) with controllable thickness."""
    im = img256_rgb.copy()
    pts = np.asarray(pts, dtype=float)
    if pts.max() <= 1.5:      # normalized coords -> scale to 256
        pts = pts * 256.0
    for aa, bb in SKELETON:
        if vis[aa] and vis[bb]:
            cv2.line(im, (int(pts[aa][0]), int(pts[aa][1])),
                     (int(pts[bb][0]), int(pts[bb][1])), (0, 200, 0), thick, cv2.LINE_AA)
    for k in range(len(pts)):
        if vis[k]:
            cv2.circle(im, (int(pts[k][0]), int(pts[k][1])), radius, (255, 40, 40), -1, cv2.LINE_AA)
    return im


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--ann', required=True, help='COCO val json')
    p.add_argument('--img-dir', required=True, help='folder with the val images')
    p.add_argument('--outdir', default='viz_pred')
    p.add_argument('--per-cat', type=int, default=3, help='query images per character')
    p.add_argument('--num-shots', type=int, default=5, help='K support images per episode')
    p.add_argument('--radius', type=int, default=2, help='keypoint dot radius')
    p.add_argument('--thick', type=int, default=1, help='skeleton line thickness')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def main():
    a = parse_args()
    random.seed(a.seed)
    cfg = Config.fromfile(a.config)
    imgsz = cfg.model.encoder_config.img_size

    model = build_posenet(cfg.model)
    load_checkpoint(model, a.checkpoint, map_location='cpu')
    model.eval().to(a.device)

    coco = json.load(open(a.ann))
    id2img = {im['id']: im for im in coco['images']}
    id2cat = {c['id']: c['name'] for c in coco['categories']}
    by_cat = {}
    for an in coco['annotations']:
        by_cat.setdefault(an['category_id'], []).append(an)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        Resize_Pad(imgsz, imgsz)])
    genHeatMap = TopDownGenerateTargetFewShot()
    data_cfg = dict(cfg.data_cfg)
    data_cfg['image_size'] = np.array([imgsz, imgsz])
    data_cfg['joint_weights'] = None
    data_cfg['use_different_joint_weights'] = False

    def build_one(ann):
        img = cv2.imread(str(Path(a.img_dir) / id2img[ann['image_id']]['file_name']))
        kp = np.array(ann['keypoints']).reshape(-1, 3)  # [21,3] px
        crop, ox, oy = crop_char(img, kp)               # crop to character bbox
        ch, cw = crop.shape[:2]
        vis = kp[:, 2] > 0
        kp_c = torch.tensor(kp[:, :2] - np.array([ox, oy])).float()  # kpts in crop coords
        kp_pad = kpts_to_pad(kp_c, ch, cw)
        kp3d = torch.cat([kp_pad, torch.zeros(kp_pad.shape[0], 1)], -1)
        w = torch.tensor(vis).float()[:, None]
        w3 = torch.cat([w, w, torch.zeros_like(w)], -1)
        t, tw = genHeatMap._msra_generate_target(data_cfg, kp3d.numpy(), w3.numpy(), sigma=1)
        img_t = preprocess(crop).flip(0)[None].to(a.device)
        return (img_t, torch.tensor(t).float()[None].to(a.device),
                torch.tensor(tw).float()[None].to(a.device), kp3d, crop)

    def n_visible(ann):
        return int((np.array(ann['keypoints']).reshape(-1, 3)[:, 2] > 0).sum())

    K = a.num_shots
    rng = random.Random(a.seed)
    for cat, anns in by_cat.items():
        if len(anns) < K + 1:
            continue
        # RANDOM support, but drawn from reasonably full-body images (>=15 visible
        # kpts) so a close-up support doesn't break full-body queries.
        full = [x for x in anns if n_visible(x) >= 15]
        pool = full if len(full) >= K + 1 else sorted(anns, key=n_visible, reverse=True)
        pool = list(pool)
        rng.shuffle(pool)
        sup_anns = pool[:K]
        sups = [build_one(x) for x in sup_anns]
        s_imgs = [s[0] for s in sups]
        s_tgts = [s[1] for s in sups]
        s_ws = [s[2] for s in sups]
        s_kps = [s[3] for s in sups]
        s_crops = [s[4] for s in sups]
        cen = [k[:, :2].mean(0) for k in s_kps]
        scl = [k[:, :2].max(0)[0] - k[:, :2].min(0)[0] for k in s_kps]
        out_dir = str(Path(a.outdir) / id2cat[cat])
        os.makedirs(out_dir, exist_ok=True)

        # montage of the K cropped support images with GT keypoints
        sup_drawn = []
        for si, sa in enumerate(sup_anns):
            raw = resize_pad_raw(s_crops[si])
            vis = np.array(sa['keypoints']).reshape(-1, 3)[:, 2] > 0
            sup_drawn.append(draw_pose(raw, s_kps[si][:, :2].numpy(), vis, a.radius, a.thick))
        montage = cv2.hconcat(sup_drawn)
        cv2.imwrite(f'{out_dir}/_support_x{K}.png', cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))

        # queries: pick those whose POSE is CLOSEST to the support set (full-body),
        # so support covers the query pose -> cleaner matching.
        q_cands = [x for x in pool[K:] if n_visible(x) >= 12] or list(pool[K:])
        q_pool = sorted(q_cands, key=lambda qq: min(pose_dist(qq, sa) for sa in sup_anns))
        for q in q_pool[:a.per_cat]:
            qi = id2img[q['image_id']]
            q_full = cv2.imread(str(Path(a.img_dir) / qi['file_name']))
            q_kp = np.array(q['keypoints']).reshape(-1, 3)
            q_crop, _, _ = crop_char(q_full, q_kp)      # crop query to its character
            q_t = preprocess(q_crop).flip(0)[None].to(a.device)
            data = {
                'img_s': s_imgs, 'img_q': q_t,
                'target_s': s_tgts, 'target_weight_s': s_ws,
                'target_q': None, 'target_weight_q': None, 'return_loss': False,
                'img_metas': [{'sample_skeleton': [SKELETON] * K, 'query_skeleton': SKELETON,
                               'sample_joints_3d': s_kps, 'query_joints_3d': s_kps[0],
                               'sample_center': cen, 'query_center': cen[0],
                               'sample_scale': scl, 'query_scale': scl[0],
                               'sample_rotation': [0] * K, 'query_rotation': 0,
                               'sample_bbox_score': [1] * K, 'query_bbox_score': 1,
                               'query_image_file': '', 'sample_image_file': [''] * K}]}
            with torch.no_grad():
                out = model(**data)
            pts = np.array(torch.as_tensor(out['points']).squeeze().cpu()).reshape(-1, 2)[:21]
            raw_q = resize_pad_raw(q_crop)
            drawn = draw_pose(raw_q, pts, np.ones(21, bool), a.radius, a.thick)
            cv2.imwrite(f'{out_dir}/{Path(qi["file_name"]).stem}_pred.png',
                        cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))
            print(f'[{id2cat[cat]}] query={qi["file_name"]} (K={K}) done')

    print('DONE ->', a.outdir)


if __name__ == '__main__':
    main()
