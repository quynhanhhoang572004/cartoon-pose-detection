r"""Headless version of demo.py — no OpenCV mouse clicking.

Reads the support keypoints + skeleton from a JSON file, so it can run on a
server without a display. Everything else mirrors demo.py.

Support keypoints must be given in the ORIGINAL support-image pixel coordinates
(the same space your annotation tool exports); this script applies the identical
pad+resize transform demo.py uses, so you do NOT pre-scale them.

JSON format (--kp):
{
  "keypoints": [[x1, y1], [x2, y2], ...],   # pixel coords in the CLEAN support frame
  "skeleton":  [[0, 1], [0, 2], ...]         # 0-indexed pairs (optional; omit for no edges)
}

Usage:
  python demo_headless.py \
      --support clean_support_frame.png \    # CLEAN frame, NOT the one with overlay burned in
      --query   tom_and_jerry_e17_frame005800.jpg \
      --kp      support_keypoints.json \
      --config  configs/demo_b.py \
      --checkpoint cartoon_test_ckpt.pth \
      --outdir  output
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import load_checkpoint
from mmpose.core import wrap_fp16_model
from mmpose.models import build_posenet
from torchvision import transforms

from models import *  # noqa: F401,F403  (triggers registry)
from tools.visualization import plot_results


class Resize_Pad:
    def __init__(self, w=256, h=256):
        self.w = w
        self.h = h

    def __call__(self, image):
        _, w_1, h_1 = image.shape
        ratio_1 = w_1 / h_1
        if round(ratio_1, 2) != 1:
            if ratio_1 > 1:
                hp = int(w_1 - h_1) // 2
                image = F.pad(image, (hp, 0, hp, 0), 0, "constant")
                return F.resize(image, [self.h, self.w])
            else:
                wp = int(h_1 - w_1) // 2
                image = F.pad(image, (0, wp, 0, wp), 0, "constant")
                return F.resize(image, [self.h, self.w])
        return F.resize(image, [self.h, self.w])


def transform_keypoints_to_pad_and_resize(keypoints, image_size):
    """Map keypoints from original image space to the 256 padded space."""
    trans = keypoints.clone()
    h, w = image_size[:2]
    if w / h > 1:  # width bigger -> pad height
        hp = int(w - h) // 2
        trans[:, 1] = keypoints[:, 1] + hp
        trans *= (256. / w)
    else:          # height bigger -> pad width
        wp = int(h - w) // 2
        trans[:, 0] = keypoints[:, 0] + wp
        trans *= (256. / h)
    return trans


def _yolo_label_for(img_path):
    """Ultralytics convention: .../images/... -> .../labels/... , ext -> .txt"""
    base = os.path.splitext(img_path)[0] + '.txt'
    parts = base.replace('\\', '/').split('/')
    if 'images' in parts:
        parts[len(parts) - 1 - parts[::-1].index('images')] = 'labels'
    return os.path.normpath('/'.join(parts))


def _find_data_yaml(start_path):
    """Walk up from a label/image path looking for a data.yaml."""
    d = os.path.dirname(os.path.abspath(start_path))
    for _ in range(6):
        cand = os.path.join(d, 'data.yaml')
        if os.path.exists(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _read_yolo_keypoints(label_path, img_w, img_h):
    with open(label_path, 'r', encoding='utf-8') as f:
        line = next(l for l in f if l.strip())  # first instance
    kpt_vals = list(map(float, line.split()))[5:]  # skip cls + xywh
    pts = [[kpt_vals[3 * k] * img_w, kpt_vals[3 * k + 1] * img_h]
           for k in range(len(kpt_vals) // 3)]
    return torch.tensor(pts).float()


def _read_skeleton(data_yaml):
    import yaml
    with open(data_yaml, 'r', encoding='utf-8') as f:
        dy = yaml.safe_load(f)
    return [tuple(e) for e in dy.get('skeleton', [])]


def load_support_keypoints(args, img_w, img_h):
    """Return (kp_orig [K,2] in support-image pixels, skeleton list of tuples).

    Input modes (in priority order):
      --kp        JSON: {"keypoints": [[x,y],...], "skeleton": [[i,j],...]}
      --kp-yolo   explicit Ultralytics YOLO-pose .txt label
      (neither)   AUTO: derive the .txt label from --support path
                  (images/->labels/) and the skeleton from a nearby data.yaml
    """
    if args.kp:  # explicit JSON
        with open(args.kp, 'r', encoding='utf-8') as f:
            ann = json.load(f)
        return torch.tensor(ann['keypoints']).float(), \
            [tuple(e) for e in ann.get('skeleton', [])]

    # YOLO label: explicit path, else auto-derived from the support image path
    label_path = args.kp_yolo or _yolo_label_for(args.support)
    if not os.path.exists(label_path):
        raise FileNotFoundError(
            f'No support keypoints. Looked for YOLO label at:\n  {label_path}\n'
            f'Either the support image must live under an images/ dir with a '
            f'matching labels/*.txt, or pass --kp-yolo / --kp explicitly.')
    print(f'[support kp] reading {label_path}')

    kp_orig = _read_yolo_keypoints(label_path, img_w, img_h)
    data_yaml = args.data_yaml or _find_data_yaml(label_path)
    skeleton = _read_skeleton(data_yaml) if data_yaml else []
    if data_yaml:
        print(f'[skeleton] from {data_yaml}')
    else:
        print('[skeleton] none found (graph refinement runs without edges)')
    return kp_orig, skeleton


def parse_args():
    p = argparse.ArgumentParser(description='Pose Anything Headless Demo')
    p.add_argument('--support', required=True, help='CLEAN support image (no overlay)')
    p.add_argument('--query', required=True)
    p.add_argument('--kp', help='JSON with support keypoints (+skeleton)')
    p.add_argument('--kp-yolo', dest='kp_yolo',
                   help='YOLO-pose .txt label for the support frame')
    p.add_argument('--data-yaml', dest='data_yaml',
                   help='Ultralytics data.yaml (for skeleton, used with --kp-yolo)')
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--outdir', default='output')
    p.add_argument('--device', default='cpu', help='cpu or cuda:0')
    p.add_argument('--fuse-conv-bn', action='store_true')
    p.add_argument('--cfg-options', nargs='+', action=DictAction, default={})
    return p.parse_args()


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True
    os.makedirs(args.outdir, exist_ok=True)
    img_size = cfg.model.encoder_config.img_size

    support_img = cv2.imread(args.support)
    query_img = cv2.imread(args.query)
    if support_img is None or query_img is None:
        raise ValueError('Fail to read images')
    h, w = support_img.shape[:2]

    kp_orig, skeleton = load_support_keypoints(args, w, h)
    if len(skeleton) == 0:
        skeleton = [(0, 0)]
    # original pixel coords -> 256 padded space (matches demo.py clicking space)
    kp_src = transform_keypoints_to_pad_and_resize(kp_orig, (h, w))

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        Resize_Pad(img_size, img_size)])
    support_t = preprocess(support_img).flip(0)[None]
    query_t = preprocess(query_img).flip(0)[None]

    genHeatMap = TopDownGenerateTargetFewShot()  # noqa: F405
    data_cfg = cfg.data_cfg
    data_cfg['image_size'] = np.array([img_size, img_size])
    data_cfg['joint_weights'] = None
    data_cfg['use_different_joint_weights'] = False
    kp_src_3d = torch.cat((kp_src, torch.zeros(kp_src.shape[0], 1)), dim=-1)
    kp_src_3d_weight = torch.cat(
        (torch.ones_like(kp_src), torch.zeros(kp_src.shape[0], 1)), dim=-1)
    target_s, target_weight_s = genHeatMap._msra_generate_target(
        data_cfg, kp_src_3d, kp_src_3d_weight, sigma=1)
    target_s = torch.tensor(target_s).float()[None]
    target_weight_s = torch.tensor(target_weight_s).float()[None]

    data = {
        'img_s': [support_t],
        'img_q': query_t,
        'target_s': [target_s],
        'target_weight_s': [target_weight_s],
        'target_q': None,
        'target_weight_q': None,
        'return_loss': False,
        'img_metas': [{'sample_skeleton': [skeleton],
                       'query_skeleton': skeleton,
                       'sample_joints_3d': [kp_src_3d],
                       'query_joints_3d': kp_src_3d,
                       'sample_center': [kp_src.mean(dim=0)],
                       'query_center': kp_src.mean(dim=0),
                       'sample_scale': [kp_src.max(dim=0)[0] - kp_src.min(dim=0)[0]],
                       'query_scale': kp_src.max(dim=0)[0] - kp_src.min(dim=0)[0],
                       'sample_rotation': [0],
                       'query_rotation': 0,
                       'sample_bbox_score': [1],
                       'query_bbox_score': 1,
                       'query_image_file': '',
                       'sample_image_file': [''],
                       }]
    }

    model = build_posenet(cfg.model)
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    model = model.to(args.device)
    model.eval()

    # move tensors to device
    data['img_s'] = [t.to(args.device) for t in data['img_s']]
    data['img_q'] = data['img_q'].to(args.device)
    data['target_s'] = [t.to(args.device) for t in data['target_s']]
    data['target_weight_s'] = [t.to(args.device) for t in data['target_weight_s']]

    with torch.no_grad():
        outputs = model(**data)

    vis_s_image = support_t[0].detach().cpu().numpy().transpose(1, 2, 0)
    vis_q_image = query_t[0].detach().cpu().numpy().transpose(1, 2, 0)
    plot_results(vis_s_image, vis_q_image, kp_src_3d,
                 target_weight_s[0], None, target_weight_s[0],
                 skeleton, None,
                 torch.tensor(outputs['points']).squeeze(0),
                 out_dir=args.outdir)
    print(f'[done] result saved to {args.outdir}/')


if __name__ == '__main__':
    main()
