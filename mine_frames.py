"""Mine annotation-ready frames from a YouTube video into a SEPARATE folder,
keeping only frames where the target character is present and FULL-BODY.

Pipeline: yt-dlp (download) -> ffmpeg (extract at --fps) -> drop blurry frames
-> drop near-duplicates -> YOLO-World open-vocab detection of the character
(--prompt) -> keep only frames whose best box looks full-body (tall enough +
not a face close-up + not tiny). Keeps this batch isolated so you annotate it,
then merge into the current dataset later.

  pip install yt-dlp opencv-python numpy ultralytics   # ffmpeg on PATH
  python mine_frames.py \
    --url https://www.youtube.com/watch?v=WDrAKCf6_Sg \
    --outdir data/pink_panther_batch2 \
    --prefix pink_panther_b2 \
    --detector yolov8s-worldv2.pt \
    --prompt "pink panther,pink cartoon panther,tall pink cartoon character" \
    --fps 2.0

Output frames: <outdir>/<prefix>_00001.jpg, ... (ready to upload to CVAT).
Tune to get MORE frames: lower --min-aspect (e.g. 0.9), raise --max-area (0.85),
lower --conf (0.03), raise --fps (3). Fewer/cleaner: opposite.
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def download(url, workdir):
    out = os.path.join(workdir, 'video.mp4')
    # YouTube now blocks the default clients ("video not available") + skips many
    # formats under the SABR experiment; forcing these clients restores format 18
    # (360p combined), which is plenty for cartoon pose frames.
    subprocess.run([
        'yt-dlp',
        '--extractor-args', 'youtube:player_client=tv,ios,web,android',
        '-f', '18/best[height<=720]/best',
        '--merge-output-format', 'mp4', '-o', out, url], check=True)
    return out


def extract(video, workdir, fps):
    raw = os.path.join(workdir, 'raw')
    os.makedirs(raw, exist_ok=True)
    subprocess.run(['ffmpeg', '-i', video, '-vf', f'fps={fps}',
                    '-q:v', '2', os.path.join(raw, 'f_%06d.jpg')], check=True)
    return sorted(Path(raw).glob('*.jpg'))


def blur_score(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()


class ClipFilter:
    """CLIP zero-shot filter (handles cartoon far better than a real-image
    detector). Keeps a frame if it looks like the character AND like a full-body
    shot, by comparing the image to short text prompts."""

    def __init__(self, model_name, char, device):
        import clip
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        # prompt sets: index 0 is the "keep" class in each
        self.presence = [f'a full shot of {char}, a cartoon character',
                         'a cartoon background scene with no character',
                         'a different cartoon character']
        self.body = [f'the whole body of {char}, head to feet visible, standing',
                     'an extreme close-up of a face']
        self.tp = self._enc_text(self.presence)
        self.tb = self._enc_text(self.body)

    def _enc_text(self, prompts):
        import clip
        with torch.no_grad():
            t = clip.tokenize(prompts).to(self.device)
            f = self.model.encode_text(t)
            return f / f.norm(dim=-1, keepdim=True)

    def scores(self, img_bgr):
        """Return (char_prob, fullbody_prob)."""
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        im = self.preprocess(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            f = self.model.encode_image(im)
            f = f / f.norm(dim=-1, keepdim=True)
            p = (100 * f @ self.tp.T).softmax(-1)[0]
            b = (100 * f @ self.tb.T).softmax(-1)[0]
        return float(p[0]), float(b[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--prefix', default='frame')
    ap.add_argument('--fps', type=float, default=2.0)
    ap.add_argument('--blur', type=float, default=50.0)
    ap.add_argument('--dup', type=float, default=5.0)
    # CLIP zero-shot character + full-body filter (works on cartoon; set --char ""
    # to skip filtering and keep every sharp, non-duplicate frame)
    ap.add_argument('--char', default='the pink panther',
                    help='character name for CLIP prompts; "" disables filtering')
    ap.add_argument('--clip-model', default='ViT-B/32')
    ap.add_argument('--char-thresh', type=float, default=0.5,
                    help='min CLIP prob the frame shows the character')
    ap.add_argument('--body-thresh', type=float, default=0.45,
                    help='min CLIP prob the frame is full-body (vs face close-up)')
    a = ap.parse_args()

    Path(a.outdir).mkdir(parents=True, exist_ok=True)

    clip_filter = None
    if a.char:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        clip_filter = ClipFilter(a.clip_model, a.char, device)
        print(f'[clip] {a.clip_model} on {device}, character="{a.char}"')

    with tempfile.TemporaryDirectory() as work:
        video = download(a.url, work)
        frames = extract(video, work, a.fps)
        print(f'[extract] {len(frames)} raw frames')

        kept, last_small = 0, None
        n_blur = n_dup = n_nochar = n_notfull = 0
        for fp in frames:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            if blur_score(img) < a.blur:
                n_blur += 1
                continue
            small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32)
            if last_small is not None and np.abs(small - last_small).mean() < a.dup:
                n_dup += 1
                continue

            if clip_filter is not None:
                char_p, body_p = clip_filter.scores(img)
                if char_p < a.char_thresh:
                    n_nochar += 1
                    continue
                if body_p < a.body_thresh:
                    n_notfull += 1
                    continue

            last_small = small
            kept += 1
            cv2.imwrite(os.path.join(a.outdir, f'{a.prefix}_{kept:05d}.jpg'), img)

        print(f'[done] kept {kept}')
        print(f'       dropped: {n_blur} blurry, {n_dup} dup, '
              f'{n_nochar} no-character, {n_notfull} not-full-body')
        print(f'       -> {a.outdir}  (upload to CVAT, annotate, then merge)')


if __name__ == '__main__':
    main()
