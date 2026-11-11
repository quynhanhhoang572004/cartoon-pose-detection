"""Second-pass CLIP filter over an EXISTING folder of frames: drop images that
do not clearly show the character (pink-object / title-card false positives) and
drop half-body / close-up frames, keeping full-body shots. Rejected frames are
MOVED to <dir>/_rejected (nothing deleted) so you can review them.

  python refilter_clip.py --dir data/pink_panther_batch2 \
      --char "the pink panther" --char-thresh 0.55 --body-thresh 0.5
"""
import argparse
import shutil
from pathlib import Path

import cv2
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--char', default='the pink panther')
    ap.add_argument('--clip-model', default='ViT-B/32')
    ap.add_argument('--char-thresh', type=float, default=0.55,
                    help='min prob the frame really shows the character')
    ap.add_argument('--body-thresh', type=float, default=0.5,
                    help='min prob the frame is full-body (vs half/close-up)')
    ap.add_argument('--solo', action='store_true',
                    help='also keep only SOLO frames (one character), drop multi-character')
    ap.add_argument('--solo-thresh', type=float, default=0.5,
                    help='min prob the frame shows the character ALONE')
    a = ap.parse_args()

    import clip
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load(a.clip_model, device=device)
    print(f'[clip] {a.clip_model} on {device}')

    # index 0 = the class we keep; the rest are the false-positive traps
    presence = [f'a clear view of {a.char}',
                'a different cartoon character',
                'a cartoon background with no main character',
                'cartoon title text, a logo or credits on screen']
    body = [f'the whole body of {a.char} from head to feet, standing, full body',
            'a close-up or only the upper half, legs not visible']
    solo = [f'{a.char} alone, only one cartoon character in the scene',
            'two or more cartoon characters together in the scene']

    def enc(prompts):
        with torch.no_grad():
            t = clip.tokenize(prompts).to(device)
            f = model.encode_text(t)
            return f / f.norm(dim=-1, keepdim=True)

    tp, tb, ts = enc(presence), enc(body), enc(solo)

    d = Path(a.dir)
    rej = d / '_rejected'
    rej.mkdir(exist_ok=True)
    imgs = sorted(p for p in d.glob('*.jpg'))
    kept = n_nochar = n_half = n_multi = 0
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        im = preprocess(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            f = model.encode_image(im)
            f = f / f.norm(dim=-1, keepdim=True)
            pp = (100 * f @ tp.T).softmax(-1)[0]
            bb = (100 * f @ tb.T).softmax(-1)[0]
            ss = (100 * f @ ts.T).softmax(-1)[0]
        # keep only if the character class wins AND is confident
        if pp.argmax().item() != 0 or pp[0].item() < a.char_thresh:
            shutil.move(str(p), str(rej / p.name)); n_nochar += 1; continue
        if bb.argmax().item() != 0 or bb[0].item() < a.body_thresh:
            shutil.move(str(p), str(rej / p.name)); n_half += 1; continue
        if a.solo and (ss.argmax().item() != 0 or ss[0].item() < a.solo_thresh):
            shutil.move(str(p), str(rej / p.name)); n_multi += 1; continue
        kept += 1

    print(f'[done] kept {kept}  |  moved to _rejected: {n_nochar} no-character, '
          f'{n_half} half/close-up, {n_multi} multi-character')
    print(f'       kept frames stay in {d}, rejects in {rej}')


if __name__ == '__main__':
    main()
