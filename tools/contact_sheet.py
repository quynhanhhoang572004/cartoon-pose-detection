"""Build a grid contact sheet from a folder, or from a scores.csv ranking, for quick eyeballing.

Usage:
    python tools/contact_sheet.py data/bad_bunny/frames --out sheet.jpg
    python tools/contact_sheet.py data/bad_bunny/frames --csv data/bad_bunny/scores.csv \
        --slice top --n 36 --out top.jpg
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--csv", type=Path, default=None, help="scores.csv to order by")
    ap.add_argument("--slice", choices=["top", "bottom", "spread"], default="spread")
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--cell", type=int, default=320, help="cell width in px")
    args = ap.parse_args()

    if args.csv:
        with open(args.csv, newline="", encoding="utf-8") as f:
            rows = [(r["file"], float(r["score"])) for r in csv.DictReader(f)]
    else:
        rows = [(p.name, 0.0) for p in sorted(args.src.iterdir()) if p.suffix.lower() in EXTS]

    if args.slice == "top":
        rows = rows[:args.n]
    elif args.slice == "bottom":
        rows = rows[-args.n:]
    else:
        idx = np.linspace(0, len(rows) - 1, min(args.n, len(rows))).astype(int)
        rows = [rows[i] for i in idx]

    cw = args.cell
    ch = int(cw * 9 / 16)
    cols = args.cols
    grid_rows = int(np.ceil(len(rows) / cols))
    sheet = np.full((grid_rows * (ch + 24), cols * cw, 3), 30, np.uint8)

    for i, (name, score) in enumerate(rows):
        img = cv2.imread(str(args.src / name))
        if img is None:
            continue
        img = cv2.resize(img, (cw, ch))
        r, c = divmod(i, cols)
        y, x = r * (ch + 24), c * cw
        sheet[y:y + ch, x:x + cw] = img
        cv2.putText(sheet, f"{name[-9:-4]} {score:.2f}", (x + 4, y + ch + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"{len(rows)} frames -> {args.out}")


if __name__ == "__main__":
    main()
