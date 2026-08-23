"""Filter extracted video frames: drop blurry, over/under-exposed and near-duplicate images.

Usage:
    python tools/filter_frames.py data/frames
    python tools/filter_frames.py data/frames --blur 80 --hash-dist 6 --delete
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dhash(gray, size=8):
    """64-bit difference hash, returned as a python int."""
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    return int("".join("1" if b else "0" for b in diff.flatten()), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="folder with extracted frames")
    ap.add_argument("--out", type=Path, default=None,
                    help="where rejected frames go (default: <src>/_rejected)")
    ap.add_argument("--blur", type=float, default=60.0,
                    help="min Laplacian variance; lower = blurrier (default 60)")
    ap.add_argument("--dark", type=float, default=35.0, help="min mean brightness")
    ap.add_argument("--bright", type=float, default=225.0, help="max mean brightness")
    ap.add_argument("--hash-dist", type=int, default=5,
                    help="Hamming distance below which a frame counts as duplicate (0 = off)")
    ap.add_argument("--delete", action="store_true", help="delete instead of moving to --out")
    args = ap.parse_args()

    files = sorted(p for p in args.src.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"no images found in {args.src}")

    rejected_dir = args.out or args.src / "_rejected"
    if not args.delete:
        rejected_dir.mkdir(parents=True, exist_ok=True)

    kept_hashes = []
    stats = {"kept": 0, "blur": 0, "exposure": 0, "duplicate": 0, "unreadable": 0}

    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            reason = "unreadable"
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fm = cv2.Laplacian(gray, cv2.CV_64F).var()
            mean = float(np.mean(gray))
            h = dhash(gray)
            if fm < args.blur:
                reason = "blur"
            elif not (args.dark <= mean <= args.bright):
                reason = "exposure"
            elif args.hash_dist > 0 and any(hamming(h, k) <= args.hash_dist for k in kept_hashes):
                reason = "duplicate"
            else:
                reason = None
                kept_hashes.append(h)

        if reason is None:
            stats["kept"] += 1
            continue

        stats[reason] += 1
        if args.delete:
            path.unlink()
        else:
            shutil.move(str(path), str(rejected_dir / path.name))

    total = len(files)
    print(f"total {total} | kept {stats['kept']} | blur {stats['blur']} | "
          f"exposure {stats['exposure']} | duplicate {stats['duplicate']} | "
          f"unreadable {stats['unreadable']}")
    if not args.delete:
        print(f"rejected frames moved to {rejected_dir}")


if __name__ == "__main__":
    main()
