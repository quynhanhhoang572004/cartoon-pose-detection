"""Adapt the model to ONE new character, end to end.

Everything we worked out on Goofy, for any character, in one command:

    python adapt_character.py --name goofy --category 7 \
        --config configs/cartoon/train_merged_supcon.py \
        --checkpoint work_dirs/selftrain_r4/best_PCK_epoch_40.pth \
        --coco ../data/coco_goofy.json --img-dir ../data/goofy_eval/images \
        --unlabeled-dir ../data/goofy_batch1 \
        --labeled-coco ../data/merged/coco_train.json \
        --out-dir ../data/adapt_goofy

Steps (each can be skipped once it has run):
    1 split    the labelled images into --shots support + the rest as query
    2 measure  PCK with and without test-time adaptation
    3 quality  how accurate the pseudo-labels would be, on the held-out query
    4 generate pseudo-labels for --unlabeled-dir
    5 merge    real labels + support shots + pseudo-labels into a train json
    6 print    the train and final-eval commands (training is long; run it yourself)

Nothing here trains. Step 6 hands you the command.

It never lets the query images into the training json -- they are the only clean
measurement there is.

On the teacher: the adapted model is the default, because what self-training
needs is CORRECT LABELS PER FRAME, i.e. coverage times accuracy. On Goofy the
adapter raised coverage 0.221 -> 0.258 while accuracy moved 0.611 -> 0.605,
which on ~130 kept keypoints is well under one standard error -- so it is ~16%
more correct labels at an accuracy we cannot distinguish. Step 3 re-measures
both on each new character; --pseudo-teacher base switches back.
"""
import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path


def sh(cmd, log=None):
    """Run a step, echo it, and tee the output so the numbers are kept."""
    print('\n$ ' + ' '.join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    out = r.stdout + ('\n' + r.stderr if r.returncode else '')
    print(out, flush=True)
    if log:
        Path(log).write_text(out, encoding='utf-8')
    if r.returncode:
        sys.exit(f'[adapt] step failed ({r.returncode}): {cmd[1]}')
    return out


def split(coco_path, category, shots, seed, out_dir):
    """Hold out --shots support images; everything else is query.

    Uses the same Random(seed) shuffle over the same annotation order as
    test_tta.py's trial 0, so the support set written here IS the one trial 0
    adapts on -- the pseudo-label teacher and the reported PCK then rest on the
    same shots instead of two different draws.
    """
    coco = json.load(open(coco_path))
    anns = [a for a in coco['annotations'] if a['category_id'] == category]
    if len(anns) < shots + 1:
        sys.exit(f'[adapt] category {category} has {len(anns)} labelled images, '
                 f'need at least {shots + 1}')
    idx = list(range(len(anns)))
    random.Random(seed).shuffle(idx)
    sup_ids = {anns[i]['id'] for i in idx[:shots]}
    by_img = {im['id']: im for im in coco['images']}

    paths = {}
    for tag, keep in (('support', True), ('query', False)):
        sel = [a for a in anns if (a['id'] in sup_ids) == keep]
        p = Path(out_dir) / f'{tag}.json'
        json.dump({'images': [by_img[a['image_id']] for a in sel],
                   'annotations': sel,
                   'categories': coco['categories']}, open(p, 'w'))
        print(f'  {tag:<8} {len(sel):4d} anns -> {p}')
        paths[tag] = str(p)
    return paths['support'], paths['query']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, help='character name, for filenames')
    ap.add_argument('--category', type=int, required=True, help='its COCO category id')
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True, help='the BASE model to adapt from')
    ap.add_argument('--coco', required=True, help='COCO with this character labelled')
    ap.add_argument('--img-dir', required=True, help='images for --coco')
    ap.add_argument('--unlabeled-dir', help='frames to pseudo-label (skips 4-5 if absent)')
    ap.add_argument('--labeled-coco', help='the existing training COCO, to merge into')
    ap.add_argument('--out-dir', required=True)

    ap.add_argument('--shots', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--trials', type=int, default=3, help='support draws to average in step 2')
    # TTA settings: these are the ones that worked on Goofy (+0.072 PCK@0.05).
    # mode=norm matters -- norm_head diverged, 0.470 -> 0.075.
    ap.add_argument('--tta-steps', type=int, default=20)
    ap.add_argument('--tta-lr', default='1e-3')
    ap.add_argument('--tta-mode', default='norm', choices=['norm', 'norm_head', 'all'])
    # pseudo-label filter
    ap.add_argument('--n-subsets', type=int, default=3)
    ap.add_argument('--filter-shots', type=int, default=3,
                    help='shots per subset; must be < --shots or the subsets repeat')
    ap.add_argument('--eps', default='0.03')
    ap.add_argument('--min-visible', type=int, default=6)
    ap.add_argument('--emb-thresh', default='0.25')
    ap.add_argument('--pseudo-teacher', default='tta', choices=['base', 'tta'],
                    help='tta lays the step-2 adapter over the base model: more '
                         'kept labels at indistinguishable accuracy. base skips it')

    ap.add_argument('--steps', default='1,2,3,4,5,6',
                    help='which steps to run, e.g. "4,5,6" to resume')
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()

    run = {int(s) for s in a.steps.split(',') if s.strip()}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    sup = str(out / 'support.json')
    qry = str(out / 'query.json')
    tta_ckpt = str(out / f'{a.name}_adapter.pth')
    pseudo = str(out / f'pseudo_{a.name}.json')
    merged = str(out / f'train_{a.name}.json')

    common = ['--config', a.config, '--category', a.category]

    if 1 in run:
        print(f'\n=== 1. split: {a.shots} support / rest query ===')
        split(a.coco, a.category, a.shots, a.seed, out)

    if 2 in run:
        print('\n=== 2. measure: with and without TTA ===')
        sh([py, 'test_tta.py', *common,
            '--checkpoint', a.checkpoint, '--coco', a.coco, '--img-dir', a.img_dir,
            '--shots', a.shots, '--trials', a.trials, '--seed', a.seed,
            '--steps', a.tta_steps, '--lr', a.tta_lr, '--mode', a.tta_mode,
            '--save-adapted', tta_ckpt, '--device', a.device],
           log=out / 'tta.txt')
        print(f'  -> {out / "tta.txt"}')

    # the adapter is a few hundred KB of norm affines, applied over the base
    # checkpoint -- not a second copy of the model
    teach = ['--checkpoint', a.checkpoint]
    if a.pseudo_teacher == 'tta':
        teach += ['--adapter', tta_ckpt]
    filt = ['--n-subsets', a.n_subsets, '--shots', a.filter_shots,
            '--eps', a.eps, '--min-visible', a.min_visible,
            '--use-embedding', '--emb-thresh', a.emb_thresh]

    if 3 in run:
        print('\n=== 3. pseudo-label quality, on the held-out query images ===')
        print('    Read: accuracy of KEPT keypoints must beat the model\'s own')
        print('    PCK@0.05 from step 2, or the labels add nothing.')
        sh([py, 'generate_pseudo.py', *common, *teach,
            '--support-coco', sup, '--support-img-dir', a.img_dir,
            '--eval-coco', qry, '--eval-img-dir', a.img_dir,
            *filt, '--device', a.device],
           log=out / 'quality.txt')
        print(f'  -> {out / "quality.txt"}')

    if 4 in run:
        if not a.unlabeled_dir:
            sys.exit('[adapt] step 4 needs --unlabeled-dir')
        print('\n=== 4. pseudo-label the unlabelled frames ===')
        sh([py, 'generate_pseudo.py', *common, *teach,
            '--support-coco', sup, '--support-img-dir', a.img_dir,
            '--unlabeled-dir', a.unlabeled_dir, '--out', pseudo,
            *filt, '--device', a.device])
        d = json.load(open(pseudo))
        n_frames = len(list(Path(a.unlabeled_dir).glob('*.jpg')))
        print(f'  {len(d["annotations"])} anns over {len(d["images"])} '
              f'of {n_frames} frames -> {pseudo}')

    if 5 in run:
        if not a.labeled_coco:
            sys.exit('[adapt] step 5 needs --labeled-coco')
        print('\n=== 5. merge: real labels + support shots + pseudo-labels ===')
        print('    (query.json is deliberately NOT merged in)')
        sh([py, str(Path(__file__).resolve().parent.parent / 'merge_coco.py'),
            '--out', merged, a.labeled_coco, sup, pseudo])

    if 6 in run:
        # no spaces: mmcv's --cfg-options splits its argv on whitespace, so
        # "[1, 2, 3]" arrives as three broken arguments
        ids = '[' + ','.join(str(i) for i in sorted({1, 2, 3, 4, 5, a.category})) + ']'
        print('\n=== 6. next: train, then measure once ===')
        print(f"""
# images: one flat dir holding the labelled images, this character's shots and
# its unlabelled frames, so a single img_prefix resolves every file_name.

CUDA_VISIBLE_DEVICES=0 python train.py \\
    --config configs/cartoon/selftrain.py \\
    --work-dir work_dirs/selftrain_{a.name} \\
    --cfg-options load_from="{a.checkpoint}" \\
      data.train.ann_file="{merged}" \\
      data.train.img_prefix="<FLAT_IMAGE_DIR>/" \\
      data.train.valid_class_ids="{ids}"

# Final number. --trials 1 --seed {a.seed} reuses the exact split from step 1:
# the support shots that are in training, and the query images that are not.
# Raising --trials here would draw other support sets, whose queries include
# images the model trained on.
python test_tta.py --config configs/cartoon/selftrain.py \\
    --checkpoint work_dirs/selftrain_{a.name}/latest.pth \\
    --coco {a.coco} --img-dir {a.img_dir} --category {a.category} \\
    --shots {a.shots} --trials 1 --seed {a.seed} --steps 0

# Compare against step 2's trial 0 line in {out / 'tta.txt'}:
#   base (no TTA) < TTA < self-trained  is the result you are hoping for.
""")


if __name__ == '__main__':
    main()
