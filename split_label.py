"""Split a mined frame folder into a small labeling set + the rest (unlabeled).

Moves N evenly-spaced frames (diverse poses across the video) into
<dir>/_to_label/ for CVAT annotation; everything left stays in <dir> as the
unlabeled pool for self-training.

  python split_label.py --dir data/goofy_batch1 -n 40
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('-n', type=int, default=40, help='frames to set aside for labeling')
    a = ap.parse_args()

    d = Path(a.dir)
    lab = d / '_to_label'
    lab.mkdir(exist_ok=True)
    imgs = sorted(p for p in d.glob('*.jpg'))
    if not imgs:
        raise SystemExit(f'no .jpg in {d}')
    step = max(1, len(imgs) // a.n)
    picked = imgs[::step][:a.n]                 # evenly spaced -> diverse poses
    for p in picked:
        shutil.move(str(p), str(lab / p.name))
    print(f'[split] {len(picked)} frames -> {lab}  (annotate these in CVAT)')
    print(f'        {len(imgs) - len(picked)} frames remain in {d}  (unlabeled, self-training)')


if __name__ == '__main__':
    main()
