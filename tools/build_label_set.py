"""Gather curated per-character frames into one flat set ready for keypoint labelling.

Picks a diverse subset per character using farthest-point sampling over the cached CLIP
embeddings, so the frames you label cover distinct poses instead of near-duplicates.

Usage:
    python tools/build_label_set.py --out data/label_set --cap 150 \
        data/bugs_bunny data/pink_panther data/sylvester data/mickey_mouse
"""
import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

EXTS = {".jpg", ".jpeg", ".png"}


def load_embeddings(char_dir: Path):
    """Merge every cached {names, emb} npz in the character folder into one lookup."""
    table = {}
    for npz in sorted(char_dir.glob("*.npz")):
        d = np.load(npz, allow_pickle=True)
        if "names" not in d or "emb" not in d:
            continue
        for name, vec in zip(d["names"], d["emb"]):
            table.setdefault(str(name), vec)
    return table


def farthest_point(emb, k):
    """Greedy k-center: start from the medoid, repeatedly take the least-covered point."""
    n = len(emb)
    if k >= n:
        return list(range(n))
    centre = emb.mean(axis=0)
    centre /= max(np.linalg.norm(centre), 1e-9)
    picked = [int(np.argmax(emb @ centre))]
    # cosine distance to the nearest already-picked frame
    dist = 1.0 - emb @ emb[picked[0]]
    for _ in range(k - 1):
        nxt = int(np.argmax(dist))
        picked.append(nxt)
        dist = np.minimum(dist, 1.0 - emb @ emb[nxt])
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chars", nargs="+", type=Path, help="character folders (each with frames/)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=150, help="max frames per character")
    ap.add_argument("--subdir", default="frames", help="subfolder holding the curated frames")
    args = ap.parse_args()

    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for char_dir in args.chars:
        name = char_dir.name
        src = char_dir / args.subdir
        paths = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
        if not paths:
            print(f"{name}: no frames, skipped")
            continue

        table = load_embeddings(char_dir)
        have = [p for p in paths if p.name in table]
        missing = len(paths) - len(have)

        if len(have) < len(paths) * 0.5:
            # not enough cached vectors to rank by diversity; fall back to an even stride
            step = max(1, len(paths) // args.cap)
            chosen = paths[::step][:args.cap]
            how = "stride"
        else:
            emb = np.stack([table[p.name] for p in have])
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            chosen = [have[i] for i in farthest_point(emb, args.cap)]
            how = "diverse"

        for p in sorted(chosen):
            shutil.copy2(p, img_dir / p.name)
            rows.append({"file": p.name, "character": name, "source": str(p)})
        print(f"{name}: {len(paths)} available -> {len(chosen)} picked ({how}"
              f"{f', {missing} without embeddings' if missing else ''})")

    with open(args.out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "character", "source"])
        w.writeheader()
        w.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["character"]] = counts.get(r["character"], 0) + 1
    print(f"\ntotal {len(rows)} images -> {img_dir}")
    print("per character:", counts)


if __name__ == "__main__":
    main()
