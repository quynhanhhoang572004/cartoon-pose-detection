r"""Support-guided test-time adaptation (TTA) for cartoon CAPE.

WHY THIS IS POSSIBLE HERE
-------------------------
CAPE's protocol hands the model a LABELED support set at test time, already in
the target domain (the new character, the cartoon style). Standard CAPE spends
that set on matching only and never updates a weight. Standard test-time
adaptation (TENT, SHOT) has the opposite problem: it updates weights but has no
labels at test time, so it falls back to entropy / pseudo-label surrogates.

Here we have both. Before predicting the real queries we take a few SUPERVISED
gradient steps on the support set itself, via LEAVE-ONE-SHOT-OUT: each support
image takes a turn as the query -- whose ground truth we already know -- with
the remaining shots as its support. Then we predict the real queries, then we
restore the original weights so the next character starts clean.

Because the support set is fixed for a character, we adapt ONCE per character
(not once per query): the adaptation would be identical for every query anyway.

Only normalization-layer parameters move by default (--mode norm), the usual
guard against overfitting a handful of shots.

  python test_tta.py \
    --config configs/cartoon/train_merged.py \
    --checkpoint work_dirs/train_supcon_100ep/best_PCK_epoch_70.pth \
    --coco /.../data/merged/coco_val.json \
    --img-dir /.../data/merged/images/val \
    --category 4 --shots 5 --steps 5 --lr 1e-4 --trials 3

Reports baseline (no TTA) vs TTA on the SAME support/query split, averaged over
--trials random splits, since a 5-shot result swings a lot with which 5 you get.
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmpose.models import build_posenet
from torchvision import transforms

from models import *  # noqa: F401,F403  (registers custom classes)
from models.datasets.pipelines.top_down_transform import TopDownGenerateTargetFewShot
from demo_batch import Resize_Pad, crop_char
# reuse the EXACT support/coordinate plumbing the pseudo-label path uses
from generate_pseudo import build_support_pool, pad_to_orig, predict

NUM_KPT = 21
NORM_LAYERS = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm,
               nn.InstanceNorm1d, nn.InstanceNorm2d)


# --------------------------------------------------------------------------- #
#  which weights are allowed to move during adaptation                        #
# --------------------------------------------------------------------------- #
def select_params(model, mode):
    """Freeze everything, then unfreeze the chosen subset. Returns that subset.

    norm       -- only normalization affine params (TENT-style; safest on 5 shots)
    norm_head  -- norm params + the keypoint head (stronger, overfits sooner)
    all        -- everything (ablation upper bound; expect overfitting)
    """
    for p in model.parameters():
        p.requires_grad_(False)

    chosen = []
    if mode in ('norm', 'norm_head'):
        for m in model.modules():
            if isinstance(m, NORM_LAYERS):
                for p in m.parameters(recurse=False):
                    p.requires_grad_(True)
                    chosen.append(p)
    if mode == 'norm_head':
        for p in model.keypoint_head.parameters():
            if not p.requires_grad:          # don't add the head's norms twice
                p.requires_grad_(True)
                chosen.append(p)
    if mode == 'all':
        for p in model.parameters():
            p.requires_grad_(True)
            chosen.append(p)

    if not chosen:
        raise SystemExit(f'--mode {mode} selected no parameters to adapt')
    return chosen


# --------------------------------------------------------------------------- #
#  one leave-one-shot-out episode, built for a LOSS pass (return_loss=True)   #
# --------------------------------------------------------------------------- #
def episode(pool, q_idx, sup_idx, skeleton):
    """Shot `q_idx` becomes the query (its GT is known); `sup_idx` are its support."""
    sub = [pool[j] for j in sup_idx]
    q = pool[q_idx]
    K = len(sub)
    return {
        'img_s': [e['img'] for e in sub],
        'target_s': [e['tgt'] for e in sub],
        'target_weight_s': [e['w'] for e in sub],
        'img_q': q['img'],
        'target_q': q['tgt'],
        'target_weight_q': q['w'],
        'return_loss': True,
        'img_metas': [{
            'sample_skeleton': [skeleton] * K, 'query_skeleton': skeleton,
            'sample_joints_3d': [e['kp3d'] for e in sub], 'query_joints_3d': q['kp3d'],
            'sample_center': [e['cen'] for e in sub], 'query_center': q['cen'],
            'sample_scale': [e['scl'] for e in sub], 'query_scale': q['scl'],
            'sample_rotation': [0] * K, 'query_rotation': 0,
            'sample_bbox_score': [1] * K, 'query_bbox_score': 1,
            'query_image_file': '', 'sample_image_file': [''] * K}]}


def adapt(model, pool, skeleton, a):
    """Take --steps supervised gradient steps on the support set alone.

    The query cycles through the shots, so with --steps == --shots every shot is
    used exactly once. Returns the per-step loss for logging.
    """
    params = select_params(model, a.mode)
    opt = torch.optim.Adam(params, lr=a.lr)
    K = len(pool)
    history = []
    for step in range(a.steps):
        q_idx = step % K
        sup_idx = [j for j in range(K) if j != q_idx]
        if a.adapt_shots:
            sup_idx = sup_idx[:a.adapt_shots]
        losses = model(**episode(pool, q_idx, sup_idx, skeleton))
        loss = sum(v for k, v in losses.items() if 'loss' in k)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss))
    for p in model.parameters():          # leave the model in a clean eval state
        p.requires_grad_(False)
    return history


# --------------------------------------------------------------------------- #
#  evaluation: PCK + EPE on the held-out queries of this character            #
# --------------------------------------------------------------------------- #
THRS = (0.05, 0.1, 0.2)


def evaluate(model, pool, queries, skeleton, a, preprocess):
    """Crop each query to its GT character bbox (as training does), predict with
    the full support set, compare in ORIGINAL image pixels."""
    correct = {t: 0 for t in THRS}
    n = 0
    epe = 0.0
    for an in queries:
        img = cv2.imread(str(Path(a.img_dir) / an['file_name']))
        if img is None:
            continue
        gt = np.array(an['keypoints']).reshape(-1, 3)
        gxy, gv = gt[:, :2], gt[:, 2] > 0
        if gv.sum() < 1:
            continue
        crop, ox, oy = crop_char(img, gt)
        ch, cw = crop.shape[:2]
        if ch < 5 or cw < 5:
            continue
        q_t = preprocess(crop).flip(0)[None].to(a.device)
        pts = predict(model, pool, q_t, skeleton, a.device)        # 256 pad space
        pts = pad_to_orig(pts, ch, cw) + np.array([ox, oy])        # -> original px

        bb = an.get('bbox')
        norm_sz = max(bb[2], bb[3]) if bb else float((gxy[gv].max(0) - gxy[gv].min(0)).max())
        for k in range(NUM_KPT):
            if gv[k]:
                d = float(np.linalg.norm(pts[k] - gxy[k]))
                n += 1
                epe += d
                for t in THRS:
                    correct[t] += (d / norm_sz < t)
    return ({t: correct[t] / max(n, 1) for t in THRS}, epe / max(n, 1), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--coco', required=True,
                    help='labeled COCO for the TARGET character (split into support + queries)')
    ap.add_argument('--img-dir', required=True)
    ap.add_argument('--category', type=int, required=True, help='character category id')
    ap.add_argument('--shots', type=int, default=5, help='labeled support images given at test time')
    # --- adaptation knobs ---
    ap.add_argument('--steps', type=int, default=5, help='gradient steps on the support set')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--mode', default='norm', choices=['norm', 'norm_head', 'all'],
                    help='which weights adapt (norm is the safe default on few shots)')
    ap.add_argument('--adapt-shots', type=int, default=0,
                    help='cap the support size inside leave-one-out episodes (0 = use all shots-1)')
    ap.add_argument('--trials', type=int, default=3,
                    help='random support/query splits to average (5-shot results swing a lot)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()

    cfg = Config.fromfile(a.config)
    imgsz = cfg.model.encoder_config.img_size
    model = build_posenet(cfg.model)
    load_checkpoint(model, a.checkpoint, map_location='cpu')
    model.eval().to(a.device)
    # pristine weights: restored before every trial and after every adaptation
    snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        Resize_Pad(imgsz, imgsz)])
    gen = TopDownGenerateTargetFewShot()
    data_cfg = dict(cfg.data_cfg)
    data_cfg['image_size'] = np.array([imgsz, imgsz])
    data_cfg['joint_weights'] = None
    data_cfg['use_different_joint_weights'] = False

    coco = json.load(open(a.coco))
    id2name = {im['id']: im['file_name'] for im in coco['images']}
    anns = [dict(an, file_name=id2name[an['image_id']])
            for an in coco['annotations'] if an['category_id'] == a.category]
    if len(anns) < a.shots + 1:
        raise SystemExit(f'need >= {a.shots + 1} labeled images, have {len(anns)}')
    skeleton = next(c['skeleton'] for c in coco['categories'] if c['id'] == a.category)
    print(f'[data] category {a.category}: {len(anns)} labeled images '
          f'-> {a.shots} support / {len(anns) - a.shots} query per trial')

    agg = {'base': {t: [] for t in THRS}, 'tta': {t: [] for t in THRS},
           'base_epe': [], 'tta_epe': []}

    for trial in range(a.trials):
        rng = random.Random(a.seed + trial)
        idx = list(range(len(anns)))
        rng.shuffle(idx)
        sup_anns = [anns[i] for i in idx[:a.shots]]
        qry_anns = [anns[i] for i in idx[a.shots:]]

        model.load_state_dict(snapshot)        # start every trial from pristine
        pool = build_support_pool(sup_anns, a.img_dir, gen, data_cfg, preprocess, a.device)
        if len(pool) < 2:
            print(f'  trial {trial}: only {len(pool)} support images readable, skipped')
            continue

        base_pck, base_epe, n = evaluate(model, pool, qry_anns, skeleton, a, preprocess)
        hist = adapt(model, pool, skeleton, a)
        tta_pck, tta_epe, _ = evaluate(model, pool, qry_anns, skeleton, a, preprocess)
        model.load_state_dict(snapshot)        # reset: next character starts clean

        print(f'  trial {trial}: n={n} kpts | loss {hist[0]:.4f} -> {hist[-1]:.4f} | '
              f'PCK@0.05 {base_pck[0.05]:.3f} -> {tta_pck[0.05]:.3f} | '
              f'EPE {base_epe:.1f} -> {tta_epe:.1f}')
        for t in THRS:
            agg['base'][t].append(base_pck[t])
            agg['tta'][t].append(tta_pck[t])
        agg['base_epe'].append(base_epe)
        agg['tta_epe'].append(tta_epe)

    if not agg['base_epe']:
        raise SystemExit('no trial completed')

    mean = lambda v: sum(v) / len(v)
    print('=' * 68)
    print(f'[TTA] category {a.category} | mode={a.mode} steps={a.steps} lr={a.lr} '
          f'shots={a.shots} | mean over {len(agg["base_epe"])} trials')
    print(f'{"metric":<14}{"no TTA":>10}{"with TTA":>11}{"delta":>10}')
    for t in THRS:
        b, s = mean(agg['base'][t]), mean(agg['tta'][t])
        print(f'{"PCK@" + str(t):<14}{b:>10.3f}{s:>11.3f}{s - b:>+10.3f}')
    b, s = mean(agg['base_epe']), mean(agg['tta_epe'])
    print(f'{"EPE (px)":<14}{b:>10.2f}{s:>11.2f}{s - b:>+10.2f}   (lower is better)')
    print('=' * 68)
    print('Support set is LABELED target-domain data available at test time; only')
    print(f'{a.mode} params moved, {a.steps} steps, weights reset after each trial.')


if __name__ == '__main__':
    main()
