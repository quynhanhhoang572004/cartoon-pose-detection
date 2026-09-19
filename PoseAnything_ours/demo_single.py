"""Single (support, query) inference in the demo_batch STYLE (thin cv2 lines +
bbox crop like training) — but with your OWN images, like demo_headless.

Support keypoints come from a JSON (same format as demo_headless --kp):
    {"keypoints": [[x,y], ...21...], "skeleton": [[i,j], ...]}   # pixel coords
Points stored as (0,0) are treated as invisible (not used, not drawn).

The support is cropped to its character bbox (from its keypoints), matching how
the model was trained/evaluated. The query has no annotation, so by default the
whole query is padded; pass --query-bbox x0 y0 x1 y1 to crop it the same way
(recommended if the character does not fill the frame).

  python demo_single.py \
    --support .../images/val/bugs_bunny_v2_00101.jpg \
    --query   .../test_data/bugs_bunny_02603.jpg \
    --kp support_keypoints.json \
    --config configs/cartoon/train_merged.py \
    --checkpoint work_dirs/train_supcon_100ep/best_PCK_epoch_70.pth \
    --outdir ../data/test_result --radius 2 --thick 1
"""
import argparse
import json
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
# reuse the exact drawing / crop / pad helpers demo_batch uses
from demo_batch import (SKELETON, Resize_Pad, kpts_to_pad, resize_pad_raw,
                        crop_char, draw_pose)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--support', required=True)
    p.add_argument('--query', required=True)
    p.add_argument('--kp', required=True, help='JSON: {"keypoints":[[x,y],...], "skeleton":[...]}')
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--outdir', default='output')
    p.add_argument('--query-bbox', nargs=4, type=float, default=None,
                   metavar=('x0', 'y0', 'x1', 'y1'),
                   help='crop the query to this box (px); else the whole query is padded')
    p.add_argument('--radius', type=int, default=2)
    p.add_argument('--thick', type=int, default=1)
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


def main():
    a = parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
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

    # --- support: keypoints from JSON, cropped to its bbox (like training) ---
    ann = json.load(open(a.kp))
    kp_xy = np.array(ann['keypoints'], dtype=float)             # [21,2] px
    vis = (np.abs(kp_xy).sum(1) > 0).astype(float)             # (0,0) => invisible
    kp_np = np.concatenate([kp_xy, vis[:, None]], 1)           # [21,3]
    skeleton = [tuple(e) for e in ann.get('skeleton', SKELETON)]

    s_img = cv2.imread(a.support)
    crop, ox, oy = crop_char(s_img, kp_np)
    ch, cw = crop.shape[:2]
    kp_c = torch.tensor(kp_np[:, :2] - np.array([ox, oy])).float()
    kp_pad = kpts_to_pad(kp_c, ch, cw)
    kp3d = torch.cat([kp_pad, torch.zeros(kp_pad.shape[0], 1)], -1)
    w = torch.tensor(vis).float()[:, None]
    w3 = torch.cat([w, w, torch.zeros_like(w)], -1)
    t, tw = gen._msra_generate_target(data_cfg, kp3d.numpy(), w3.numpy(), sigma=1)
    s_t = preprocess(crop).flip(0)[None].to(a.device)
    s_tgt = torch.tensor(t).float()[None].to(a.device)
    s_w = torch.tensor(tw).float()[None].to(a.device)
    cen = kp3d[:, :2].mean(0)
    scl = kp3d[:, :2].max(0)[0] - kp3d[:, :2].min(0)[0]

    # --- query: optional bbox crop, else whole-image pad ---
    q_img = cv2.imread(a.query)
    if a.query_bbox is not None:
        x0, y0, x1, y1 = [int(v) for v in a.query_bbox]
        q_crop = q_img[y0:y1, x0:x1]
    else:
        q_crop = q_img
    q_t = preprocess(q_crop).flip(0)[None].to(a.device)

    data = {
        'img_s': [s_t], 'img_q': q_t,
        'target_s': [s_tgt], 'target_weight_s': [s_w],
        'target_q': None, 'target_weight_q': None, 'return_loss': False,
        'img_metas': [{'sample_skeleton': [skeleton], 'query_skeleton': skeleton,
                       'sample_joints_3d': [kp3d], 'query_joints_3d': kp3d,
                       'sample_center': [cen], 'query_center': cen,
                       'sample_scale': [scl], 'query_scale': scl,
                       'sample_rotation': [0], 'query_rotation': 0,
                       'sample_bbox_score': [1], 'query_bbox_score': 1,
                       'query_image_file': '', 'sample_image_file': ['']}]}
    with torch.no_grad():
        out = model(**data)
    pts = np.array(torch.as_tensor(out['points']).squeeze().cpu()).reshape(-1, 2)[:21]

    # draw only the keypoints the support actually defined (visible) — cleaner
    vmask = vis.astype(bool)
    q_draw = draw_pose(resize_pad_raw(q_crop), pts, vmask, a.radius, a.thick)
    s_draw = draw_pose(resize_pad_raw(crop), kp3d[:, :2].numpy(), vmask, a.radius, a.thick)

    stem = Path(a.query).stem
    cv2.imwrite(f'{a.outdir}/{stem}_pred.png', cv2.cvtColor(q_draw, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'{a.outdir}/{stem}_support.png', cv2.cvtColor(s_draw, cv2.COLOR_RGB2BGR))
    print(f'[done] {a.outdir}/{stem}_pred.png  +  {stem}_support.png')


if __name__ == '__main__':
    main()
