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
import torch.nn.functional as F
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
    pts = np.array(torch.as_tensor(out['points']).squeeze().cpu()).reshape(-1, 2)[:21]
    return pts * 256.0     # model outputs [0,1] normalized -> convert to 256 pad space


# ============================================================================ #
#  EMBEDDING-CONSISTENCY (uses the SupCon-shaped keypoint embedding).           #
#  A pseudo-label at (x,y) is trusted only if the feature sampled there looks   #
#  like that keypoint TYPE, i.e. is close to the type's prototype (mean support #
#  embedding). This is the SEMANTIC check that complements support-invariance   #
#  (a confident-but-misplaced point is stable across supports but its embedding #
#  does NOT match its keypoint type -> caught here). NEEDS a shakeout run.      #
# ============================================================================ #
def compute_prototypes(model, pool):
    """Mean per-keypoint embedding over the labeled support pool -> [num_kpt, C].
    Replicates PoseHead's support-token extraction (heatmap-weighted feature)."""
    acc = wsum = None
    for e in pool:
        with torch.no_grad():
            _, fs = model.extract_features([e['img']], e['img'])   # feature_s
        feat = fs[0]                                               # [1, C, fh, fw]
        tgt = e['tgt']                                             # [1, num_kpt, sh, sw]
        feat_r = F.interpolate(feat, size=tgt.shape[-2:], mode='bilinear', align_corners=False)
        t = tgt / (tgt.flatten(2).sum(-1)[..., None, None] + 1e-8)
        emb = t.flatten(2) @ feat_r.flatten(2).permute(0, 2, 1)    # [1, num_kpt, C]
        vis = (e['w'] > 0).float()                                 # [1, num_kpt, 1] visible mask
        acc = emb * vis if acc is None else acc + emb * vis
        wsum = vis if wsum is None else wsum + vis
    proto = (acc / (wsum + 1e-8)).squeeze(0)                       # [num_kpt, C]
    return F.normalize(proto, dim=-1)


def query_embeddings(model, support_img, q_t, pts_norm):
    """Sample the query feature map at the predicted [0,1] keypoint coords.
    Returns L2-normalized [num_kpt, C]."""
    with torch.no_grad():
        feat_q, _ = model.extract_features([support_img], q_t)     # [1, C, h, w]
    grid = torch.tensor(pts_norm, dtype=torch.float32) * 2.0 - 1.0  # [num_kpt, 2] in [-1,1]
    grid = grid[None, None].to(feat_q.device)                      # [1, 1, num_kpt, 2]
    samp = F.grid_sample(feat_q, grid, align_corners=False)        # [1, C, 1, num_kpt]
    return F.normalize(samp[0, :, 0, :].t(), dim=-1)               # [num_kpt, C]


def bbox_from_pts(pts, W, H, pad=0.3):
    """Character bbox from predicted keypoints (+padding), clamped to the image."""
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    bw, bh = x1 - x0, y1 - y0
    x0, x1 = x0 - bw * pad, x1 + bw * pad
    y0, y1 = y0 - bh * pad, y1 + bh * pad
    return (max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1)))


def filter_query(model, pool, prototypes, q_t, skeleton, a, rng, emb=None):
    """Run the support-ensemble (+ optional embedding) filter on ONE query.
    Returns (mean_256 [21,2], visible [21] bool). `emb` overrides a.use_embedding."""
    use_emb = a.use_embedding if emb is None else emb
    eps_px = a.eps * 256.0
    preds = np.stack([predict(model, rng.sample(pool, a.shots), q_t, skeleton, a.device)
                      for _ in range(a.n_subsets)], 0)        # [N, 21, 2]
    mean = preds.mean(0)
    visible = np.linalg.norm(preds - mean[None], axis=2).max(0) < eps_px   # tier 1
    if use_emb:
        qemb = query_embeddings(model, pool[0]['img'], q_t, mean / 256.0)
        cos = (qemb * prototypes).sum(-1).cpu().numpy()
        visible = visible & ((1.0 - cos) < a.emb_thresh)      # tier 2
    return mean, visible


def pseudo_for_frame(model, pool, prototypes, frame, preprocess, skeleton, a, rng):
    """Full per-frame pipeline. With 2-stage self-crop (default): a rough pass on
    the whole frame gives a character bbox, then the ensemble runs on the CROP
    (character fills the frame like at training) -> far better pseudo-labels on
    raw unlabeled frames. Returns (pts_orig [21,2] in original px, visible [21])."""
    H, W = frame.shape[:2]
    x0, y0, crop = 0, 0, frame
    if not a.single_stage:
        q0 = preprocess(frame).flip(0)[None].to(a.device)
        m0, _ = filter_query(model, pool, prototypes, q0, skeleton, a, rng, emb=False)
        rough = pad_to_orig(m0, H, W)                         # rough keypoints, original px
        bx0, by0, bx1, by1 = bbox_from_pts(rough, W, H, pad=0.3)
        if bx1 - bx0 >= 5 and by1 - by0 >= 5:
            x0, y0, crop = bx0, by0, frame[by0:by1, bx0:bx1]
    ch, cw = crop.shape[:2]
    q_t = preprocess(crop).flip(0)[None].to(a.device)
    mean, visible = filter_query(model, pool, prototypes, q_t, skeleton, a, rng)
    pts = pad_to_orig(mean, ch, cw) + np.array([x0, y0])      # crop -> original px
    return pts, visible


def run_eval(model, pool, prototypes, skeleton, a, preprocess, rng):
    """Measure pseudo-label QUALITY against a labeled set: of the keypoints the
    filter KEEPS, how many are correct (PCK@0.2 vs GT), and how many we keep
    (coverage). Tells you whether the base model is a good enough teacher."""
    coco = json.load(open(a.eval_coco))
    id2name = {im['id']: im['file_name'] for im in coco['images']}
    anns = [an for an in coco['annotations'] if an['category_id'] == a.category]
    thrs = [0.05, 0.1, 0.2]
    correct = {t: 0 for t in thrs}
    kept_gtvis = gtvis = 0
    for an in anns:
        img = cv2.imread(str(Path(a.eval_img_dir) / id2name[an['image_id']]))
        if img is None:
            continue
        pts, visible = pseudo_for_frame(model, pool, prototypes, img, preprocess, skeleton, a, rng)
        gt = np.array(an['keypoints']).reshape(-1, 3)
        gxy, gv = gt[:, :2], gt[:, 2] > 0
        bb = an.get('bbox')
        norm_sz = max(bb[2], bb[3]) if bb else float((gxy[gv].max(0) - gxy[gv].min(0)).max())
        for k in range(21):
            if gv[k]:
                gtvis += 1
                if visible[k]:
                    kept_gtvis += 1
                    d = np.linalg.norm(pts[k] - gxy[k]) / norm_sz
                    for t in thrs:
                        correct[t] += (d < t)
    cov = kept_gtvis / max(gtvis, 1)
    print('=' * 60)
    print(f'[EVAL] pseudo-label quality on {len(anns)} labeled frames')
    for t in thrs:
        print(f'   accuracy of KEPT keypoints  PCK@{t:<4}: {correct[t] / max(kept_gtvis, 1):.3f}')
    print(f'   coverage (kept / GT-visible keypoints): {cov:.3f}')
    print(f'   settings: n_subsets={a.n_subsets} shots={a.shots} eps={a.eps} '
          f'embedding={a.use_embedding} emb_thresh={a.emb_thresh}')
    print('   -> for self-training to sharpen localization, PCK@0.05 of kept '
          'labels should also be high (not just @0.2)')
    print('=' * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--support-coco', required=True, help='COCO json holding the labeled support set')
    ap.add_argument('--support-img-dir', required=True)
    ap.add_argument('--category', type=int, required=True, help='character category id in the COCO')
    ap.add_argument('--unlabeled-dir', help='folder of unlabeled query frames (generate mode)')
    ap.add_argument('--out', help='output pseudo-label COCO json (generate mode)')
    # --- EVAL mode: measure pseudo-label quality against a LABELED set ---
    ap.add_argument('--eval-coco', help='labeled COCO with GT keypoints; enables EVAL mode')
    ap.add_argument('--eval-img-dir', help='images dir for --eval-coco')
    ap.add_argument('--n-subsets', type=int, default=3, help='N independent support subsets to cross-check')
    ap.add_argument('--shots', type=int, default=5, help='support images per subset')
    ap.add_argument('--eps', type=float, default=0.05,
                    help='agreement radius as a fraction of 256 (0.05 = ~13px)')
    ap.add_argument('--min-visible', type=int, default=8, help='min agreed keypoints to keep a frame')
    ap.add_argument('--single-stage', action='store_true',
                    help='disable 2-stage self-crop (pad the whole frame; worse on raw frames)')
    # --- second filter tier: embedding-consistency (uses SupCon embedding) ---
    ap.add_argument('--use-embedding', action='store_true',
                    help='also require the sampled feature to match the keypoint prototype')
    ap.add_argument('--emb-thresh', type=float, default=0.3,
                    help='max cosine DISTANCE (1-cos) between a point and its keypoint prototype')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    rng = random.Random(a.seed)

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

    prototypes = None
    if a.use_embedding:                     # [num_kpt, C] mean support embedding per keypoint
        prototypes = compute_prototypes(model, pool)
        print(f'[embedding] prototypes ready: {tuple(prototypes.shape)}')

    # EVAL MODE: measure pseudo-label quality against a labeled set, then stop
    if a.eval_coco:
        if not a.eval_img_dir:
            raise SystemExit('--eval-coco needs --eval-img-dir')
        run_eval(model, pool, prototypes, skeleton, a, preprocess, rng)
        return

    if not a.unlabeled_dir or not a.out:
        raise SystemExit('generate mode needs --unlabeled-dir and --out')
    frames = sorted(p for ext in ('*.jpg', '*.png') for p in Path(a.unlabeled_dir).glob(ext))
    out_images, out_anns = [], []
    iid = aid = 1
    kept = skipped = 0
    for fp in frames:
        q_img = cv2.imread(str(fp))
        if q_img is None:
            continue
        H, W = q_img.shape[:2]
        # 2-stage self-crop + support-ensemble (+ embedding) filter
        pts_orig, visible = pseudo_for_frame(model, pool, prototypes, q_img,
                                             preprocess, skeleton, a, rng)

        if visible.sum() < a.min_visible:
            skipped += 1
            continue

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
