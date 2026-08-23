"""Score frames by how likely they contain a target character, using CLIP.

Two scoring modes, combined when both are available:
  * text   -- softmax over positive vs negative prompts (zero-shot, no labels needed)
  * refs   -- cosine similarity to the mean embedding of example images (--refs folder)

Embeddings are cached next to the frames so re-scoring with new prompts/refs is instant.

Usage:
    # 1. zero-shot pass, writes scores.csv sorted best-first
    python tools/select_character.py data/bad_bunny/frames

    # 2. eyeball the ranking, then split the folder
    python tools/select_character.py data/bad_bunny/frames --move --thresh 0.6

    # 3. stronger: drop a few confirmed frames into a refs folder and re-score
    python tools/select_character.py data/bad_bunny/frames --refs data/bad_bunny/refs --move --thresh 0.6
"""
import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_POS = [
    "a cartoon frame showing Bugs Bunny, the grey and white rabbit",
    "a grey cartoon rabbit with long ears",
]
DEFAULT_NEG = [
    "a cartoon frame showing Daffy Duck, the black duck",
    "a cartoon frame showing Elmer Fudd, the bald hunter",
    "a cartoon frame showing Porky Pig",
    "a cartoon frame showing Yosemite Sam with a big moustache",
    "a cartoon background with no characters",
    "a title card with text",
]


def list_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)


def as_tensor(feats):
    """transformers>=5 returns a ModelOutput whose pooler_output holds the projected features."""
    if isinstance(feats, torch.Tensor):
        return feats
    return feats.pooler_output


def embed(paths, model, processor, device, batch=32, desc="frames"):
    vecs = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        imgs = [Image.open(p).convert("RGB") for p in chunk]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            f = as_tensor(model.get_image_features(**inputs))
        vecs.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
        print(f"  {desc}: {min(i + batch, len(paths))}/{len(paths)}", end="\r", flush=True)
    print()
    return np.concatenate(vecs) if vecs else np.zeros((0, 512), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="folder of frames to score")
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--pos", nargs="*", default=DEFAULT_POS, help="positive text prompts")
    ap.add_argument("--neg", nargs="*", default=DEFAULT_NEG, help="negative text prompts")
    ap.add_argument("--refs", type=Path, default=None,
                    help="folder of confirmed example images of the character")
    ap.add_argument("--ref-weight", type=float, default=0.5,
                    help="blend weight for the refs score when --refs is given (0..1)")
    ap.add_argument("--csv", type=Path, default=None, help="default: <src>/../scores.csv")
    ap.add_argument("--cache", type=Path, default=None, help="default: <src>/../clip_emb.npz")
    ap.add_argument("--move", action="store_true", help="split frames by --thresh")
    ap.add_argument("--other", type=Path, default=None,
                    help="where rejected frames go (default: <src>_other)")
    ap.add_argument("--thresh", type=float, default=0.5, help="keep frames scoring >= this")
    ap.add_argument("--top", type=int, default=None,
                    help="with --move, keep the N best instead of using --thresh")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    from transformers import CLIPModel, CLIPProcessor

    paths = list_images(args.src)
    if not paths:
        raise SystemExit(f"no images found in {args.src}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(paths)} frames | device={device} | model={args.model}")
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model)

    cache = args.cache or args.src.parent / "clip_emb.npz"
    names = [p.name for p in paths]
    img_emb = None
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        if list(d["names"]) == names:
            img_emb = d["emb"]
            print(f"loaded cached embeddings from {cache}")
    if img_emb is None:
        img_emb = embed(paths, model, processor, device, args.batch)
        np.savez(cache, names=np.array(names), emb=img_emb)
        print(f"cached embeddings to {cache}")

    prompts = list(args.pos) + list(args.neg)
    tok = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        txt = as_tensor(model.get_text_features(**tok))
    txt = torch.nn.functional.normalize(txt, dim=-1).cpu().numpy()

    logits = img_emb @ txt.T * float(model.logit_scale.exp())
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    score = probs[:, :len(args.pos)].sum(axis=1)

    if args.refs:
        ref_paths = list_images(args.refs)
        if not ref_paths:
            raise SystemExit(f"--refs given but no images in {args.refs}")
        ref_emb = embed(ref_paths, model, processor, device, args.batch, desc="refs")
        proto = ref_emb.mean(axis=0)
        proto /= np.linalg.norm(proto)
        sim = img_emb @ proto
        # map cosine sim (roughly 0.5..0.95 for a matching character) onto 0..1
        lo, hi = float(np.percentile(sim, 5)), float(np.percentile(sim, 95))
        sim_n = np.clip((sim - lo) / max(hi - lo, 1e-6), 0, 1)
        score = (1 - args.ref_weight) * score + args.ref_weight * sim_n
        print(f"blended {len(ref_paths)} refs at weight {args.ref_weight}")

    order = np.argsort(-score)
    csv_path = args.csv or args.src.parent / "scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "score"])
        for i in order:
            w.writerow([names[i], f"{score[i]:.4f}"])
    print(f"scores -> {csv_path}")
    print("score percentiles:", {p: round(float(np.percentile(score, p)), 3)
                                 for p in [10, 25, 50, 75, 90]})

    if not args.move:
        print("dry run; add --move to split the folder")
        return

    keep_idx = set(order[:args.top].tolist()) if args.top else \
        {int(i) for i in np.where(score >= args.thresh)[0]}
    other = args.other or args.src.parent / f"{args.src.name}_other"
    other.mkdir(parents=True, exist_ok=True)
    moved = 0
    for i, p in enumerate(paths):
        if i not in keep_idx:
            shutil.move(str(p), str(other / p.name))
            moved += 1
    print(f"kept {len(paths) - moved} in {args.src} | moved {moved} to {other}")


if __name__ == "__main__":
    main()
