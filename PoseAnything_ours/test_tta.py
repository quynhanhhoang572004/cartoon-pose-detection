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
import os
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
def select_params(model, mode, named=False):
    """Freeze everything, then unfreeze the chosen subset. Returns that subset.

    norm       -- only normalization affine params (TENT-style; safest on 5 shots)
    norm_head  -- norm params + the keypoint head (stronger, overfits sooner)
    all        -- everything (ablation upper bound; expect overfitting)

    With named=True the names come too, which is what lets the adaptation be
    saved as an adapter -- the handful of tensors that moved -- instead of a
    copy of the whole model.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    chosen = []                              # list of (name, param)
    if mode in ('norm', 'norm_head'):
        for mname, m in model.named_modules():
            if isinstance(m, NORM_LAYERS):
                for pname, p in m.named_parameters(recurse=False):
                    p.requires_grad_(True)
                    chosen.append((f'{mname}.{pname}' if mname else pname, p))
    if mode == 'norm_head':
        for pname, p in model.keypoint_head.named_parameters():
            if not p.requires_grad:          # don't add the head's norms twice
                p.requires_grad_(True)
                chosen.append((f'keypoint_head.{pname}', p))
    if mode == 'all':
        for pname, p in model.named_parameters():
            p.requires_grad_(True)
            chosen.append((pname, p))

    if not chosen:
        raise SystemExit(f'--mode {mode} selected no parameters to adapt')
    return chosen if named else [p for _, p in chosen]


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


def pool_loss(model, pool, skeleton, a):
    """Mean loss over ALL K leave-one-out episodes, no gradient.

    The per-step losses inside adapt() are each measured on a DIFFERENT held-out
    shot, so comparing the first to the last says nothing. This averages over the
    same K episodes every time, so before/after are comparable.
    """
    K = len(pool)
    tot = 0.0
    with torch.no_grad():
        for q_idx in range(K):
            sup_idx = [j for j in range(K) if j != q_idx]
            losses = model(**episode(pool, q_idx, sup_idx, skeleton))
            tot += float(sum(v for k, v in losses.items() if 'loss' in k))
    return tot / max(K, 1)


def adapt(model, pool, skeleton, a):
    """Take --steps supervised gradient steps on the support set alone.

    The query cycles through the shots, so with --steps == --shots every shot is
    used exactly once. Returns (loss_before, loss_after) measured on the same
    full set of leave-one-out episodes.
    """
    params = select_params(model, a.mode)
    opt = torch.optim.Adam(params, lr=a.lr)
    K = len(pool)
    before = pool_loss(model, pool, skeleton, a)
    for p in params:
        p.requires_grad_(True)
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
    after = pool_loss(model, pool, skeleton, a)
    return before, after


# --------------------------------------------------------------------------- #
#  evaluation: PCK + EPE on the held-out queries of this character            #
# --------------------------------------------------------------------------- #
THRS = (0.05, 0.1, 0.2)


def support_visible(pool):
    """Keypoints the model can actually predict: visible in EVERY support image.

    PoseAnything zeroes a support keypoint's embedding and masks it out of
    attention when any shot does not show it (head.py: support_keypoints *
    mask_s), so the coordinate it still emits for that slot is meaningless.
    pam.py builds mask_s as the product of all target_weight_s, and the
    training-time evaluator masks the same way (transformer_base_dataset.py:
    mask_sample is ANDed over the pair). Scoring those slots punishes the model
    for something it was never given.
    """
    m = np.ones(NUM_KPT, dtype=bool)
    for e in pool:
        m &= e['w'][0, :, 0].detach().cpu().numpy() > 0
    return m


def evaluate(model, pool, queries, skeleton, a, preprocess):
    """Crop each query to its GT character bbox (as training does), predict with
    the full support set, compare in ORIGINAL image pixels.

    Returns the support-masked metrics (comparable to the training-time PCK) and,
    alongside, the unmasked ones over every GT-visible keypoint.
    """
    correct = {t: 0 for t in THRS}
    correct_all = {t: 0 for t in THRS}
    n = n_all = 0
    epe = epe_all = 0.0
    sup_vis = support_visible(pool) if not a.score_all_kpts else np.ones(NUM_KPT, bool)
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
            if not gv[k]:
                continue
            d = float(np.linalg.norm(pts[k] - gxy[k]))
            n_all += 1
            epe_all += d
            for t in THRS:
                correct_all[t] += (d / norm_sz < t)
            if not sup_vis[k]:
                continue                       # model was never shown this joint
            n += 1
            epe += d
            for t in THRS:
                correct[t] += (d / norm_sz < t)
    masked = ({t: correct[t] / max(n, 1) for t in THRS}, epe / max(n, 1), n)
    unmasked = ({t: correct_all[t] / max(n_all, 1) for t in THRS},
                epe_all / max(n_all, 1), n_all)
    return masked, unmasked


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
    ap.add_argument('--save-adapted', metavar='PATH',
                    help="save trial 0's adapted weights, so generate_pseudo.py "
                         'can pseudo-label with the adapted teacher instead of '
                         'the base one (use the matching support json)')
    ap.add_argument('--score-all-kpts', action='store_true',
                    help='also score joints missing from some support shot, which '
                         'the model cannot predict (off by default, matching the '
                         'training-time evaluator)')
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
           'base_epe': [], 'tta_epe': [],
           'base_all': {t: [] for t in THRS}, 'base_all_epe': []}

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

        (base_pck, base_epe, n), (all_pck, all_epe, n_all) = \
            evaluate(model, pool, qry_anns, skeleton, a, preprocess)
        l_before, l_after = adapt(model, pool, skeleton, a)
        (tta_pck, tta_epe, _), _ = evaluate(model, pool, qry_anns, skeleton, a, preprocess)
        if a.save_adapted and trial == 0:
            # Save the ADAPTER: only the tensors adaptation was allowed to move.
            # In --mode norm that is the normalization affine params, a fraction
            # of a percent of the model, so this is ~1 MB instead of ~250 MB --
            # and one adapter per character, over the base model, is the honest
            # description of what TTA produces.
            # Trial 0 draws its support with Random(seed) over the same
            # annotation order as adapt_character.py's split, so the support set
            # recorded here IS that split's support.json.
            adapter = {n: p.detach().cpu().clone()
                       for n, p in select_params(model, a.mode, named=True)}
            n_ad = sum(v.numel() for v in adapter.values())
            n_all = sum(p.numel() for p in model.parameters())
            torch.save({'adapter': adapter, 'base_checkpoint': a.checkpoint,
                        'meta': {'tta': {'mode': a.mode, 'lr': a.lr, 'steps': a.steps,
                                         'shots': a.shots, 'seed': a.seed,
                                         'category': a.category},
                                 'support': [an['file_name'] for an in sup_anns]}},
                       a.save_adapted)
            mb = os.path.getsize(a.save_adapted) / 1e6
            print(f'  [saved] adapter -> {a.save_adapted}  ({mb:.1f} MB)')
            print(f'          {len(adapter)} tensors, {n_ad:,} params '
                  f'= {100 * n_ad / n_all:.2f}% of {n_all:,}')
            print(f'          support: {[an["file_name"] for an in sup_anns]}')
            print(f'          use it with:  --adapter {a.save_adapted}')
        model.load_state_dict(snapshot)        # reset: next character starts clean

        n_sup = int(support_visible(pool).sum())
        print(f'  trial {trial}: {n_sup}/{NUM_KPT} joints visible in all {len(pool)} shots'
              f' -> n={n} scored ({n_all} if unmasked) | '
              f'LOO loss {l_before:.4f} -> {l_after:.4f} | '
              f'PCK@0.05 {base_pck[0.05]:.3f} -> {tta_pck[0.05]:.3f} | '
              f'EPE {base_epe:.1f} -> {tta_epe:.1f}')
        for t in THRS:
            agg['base'][t].append(base_pck[t])
            agg['tta'][t].append(tta_pck[t])
            agg['base_all'][t].append(all_pck[t])
        agg['base_epe'].append(base_epe)
        agg['tta_epe'].append(tta_epe)
        agg['base_all_epe'].append(all_epe)

    if not agg['base_epe']:
        raise SystemExit('no trial completed')

    mean = lambda v: sum(v) / len(v)
    # 5-shot results swing a lot between support draws, so a bare mean over a
    # handful of trials is not reportable -- carry the spread with it.
    std = lambda v: float(np.std(v))
    print('=' * 68)
    print(f'[TTA] category {a.category} | mode={a.mode} steps={a.steps} lr={a.lr} '
          f'shots={a.shots} | mean over {len(agg["base_epe"])} trials')
    print(f'{"metric":<12}{"no TTA":>14}{"with TTA":>15}{"delta":>18}{"unmasked":>11}')
    for t in THRS:
        bv, sv = agg['base'][t], agg['tta'][t]
        dv = [s - b for b, s in zip(bv, sv)]        # paired: same support draw
        print(f'{"PCK@" + str(t):<12}'
              f'{mean(bv):>8.3f}+-{std(bv):<4.3f}'
              f'{mean(sv):>9.3f}+-{std(sv):<4.3f}'
              f'{mean(dv):>+12.3f}+-{std(dv):<4.3f}'
              f'{mean(agg["base_all"][t]):>11.3f}')
    bv, sv = agg['base_epe'], agg['tta_epe']
    dv = [s - b for b, s in zip(bv, sv)]
    print(f'{"EPE (px)":<12}'
          f'{mean(bv):>8.2f}+-{std(bv):<4.2f}'
          f'{mean(sv):>9.2f}+-{std(sv):<4.2f}'
          f'{mean(dv):>+12.2f}+-{std(dv):<4.2f}'
          f'{mean(agg["base_all_epe"]):>11.2f}   (lower is better)')
    print('delta is PAIRED (same support draw before/after), so its spread is the')
    print('one that matters -- the base/TTA spreads mostly reflect which 5 shots fell out.')
    print('=' * 68)
    print('Scored joints are those visible in the query AND in every support shot,')
    print('matching the training-time evaluator; "unmasked" scores every GT-visible')
    print('joint, including ones the model was never shown and cannot predict.')
    print('Support set is LABELED target-domain data available at test time; only')
    print(f'{a.mode} params moved, {a.steps} steps, weights reset after each trial.')


if __name__ == '__main__':
    main()
