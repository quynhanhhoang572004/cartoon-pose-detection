#!/usr/bin/env bash
# Train on the pseudo-labels for one character, then measure whether it helped.
#
#   bash train_pseudo.sh goofy 7                    # all the pseudo-labels
#   bash train_pseudo.sh goofy 7 balanced           # capped to match the others
#
# The third argument is a suffix on the merged json and on every output, so the
# two runs do not overwrite each other's work dir or before/after files.
#
# Expects, next to this script:
#   data/coco_selftrain_goofy.json   merge_coco.py output (real + support + pseudo)
#   data/merged/images/train/        the original labelled images
#   data/goofy_eval/images/          the 39 labelled frames of this character
#   data/goofy_batch1/               its unlabelled frames
set -euo pipefail

NAME=${1:-goofy}
CAT=${2:-7}
VARIANT=${3:-}                      # "" or e.g. "balanced"
TAG=$NAME${VARIANT:+_$VARIANT}

# ------------------------- EDIT THESE -------------------------
ROOT=/home/subnh3/projects/QuynhAnh/cartoon-pose-detection
OURS=$ROOT/PoseAnything_ours
BASE=$OURS/work_dirs/selftrain_r4/best_PCK_epoch_40.pth   # the teacher that made the labels
MERGED=$ROOT/data/coco_selftrain_$TAG.json
LBL_IMGS=$ROOT/data/merged/images/train                   # original labelled images
CHR_IMGS=$ROOT/data/${NAME}_eval/images                   # this character's labelled frames
UNLAB=$ROOT/data/${NAME}_batch1                           # its unlabelled frames
EVAL_COCO=$ROOT/data/coco_$NAME.json                      # ALL its labelled frames
EPOCHS=20
GPU=0
# --------------------------------------------------------------

IMGDIR=$ROOT/data/selftrain_images_$NAME      # shared: same images either way
WORK=$OURS/work_dirs/selftrain_$TAG
# the dataset never samples a category outside this list, so a missing $CAT
# means training completes, looking healthy, having learnt nothing new
IDS="[1,2,3,4,5,$CAT]"

echo "=== 0. checks ==="
for f in "$MERGED" "$BASE" "$EVAL_COCO"; do
  [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
done
python - "$MERGED" "$CAT" <<'PY'
import json, sys
from collections import Counter
d = json.load(open(sys.argv[1]))
c = Counter(a['category_id'] for a in d['annotations'])
print(f"  {len(d['images'])} images, {len(d['annotations'])} anns, per-cat {dict(c)}")
if c[int(sys.argv[2])] == 0:
    sys.exit(f"  category {sys.argv[2]} is NOT in the merged json -- did merge_coco.py "
             "print \"unknown category ... skipped\"?")
PY

echo "=== 1. one flat image dir ==="
mkdir -p "$IMGDIR"
cp -n "$LBL_IMGS"/*.jpg "$IMGDIR"/ 2>/dev/null || true
cp -n "$CHR_IMGS"/*.jpg "$IMGDIR"/ 2>/dev/null || true
cp -n "$UNLAB"/*.jpg    "$IMGDIR"/ 2>/dev/null || true
echo "  $(ls "$IMGDIR" | wc -l) images"

cd "$OURS"
python setup.py develop >/dev/null
export WANDB_MODE=offline

echo "=== 2. BEFORE: base model, fixed split, no adaptation ==="
# --trials 1 --seed 0 reuses the exact split the pseudo-labels were made with:
# the 5 support shots that ARE in training, and the query images that are NOT.
# More trials would draw other support sets, whose queries include images the
# model trained on.
python test_tta.py --config configs/cartoon/selftrain.py --checkpoint "$BASE" \
  --coco "$EVAL_COCO" --img-dir "$CHR_IMGS" --category "$CAT" \
  --shots 5 --trials 1 --seed 0 --steps 0 | tee "$ROOT/before_$TAG.txt" | tail -12

echo "=== 3. train ($EPOCHS epochs) ==="
CUDA_VISIBLE_DEVICES=$GPU python train.py \
  --config configs/cartoon/selftrain.py --work-dir "$WORK" \
  --cfg-options load_from="$BASE" total_epochs=$EPOCHS \
    data.train.ann_file="$MERGED" \
    data.train.img_prefix="$IMGDIR/" \
    data.train.valid_class_ids="$IDS"

echo "=== 4. AFTER: same split, same command, new weights ==="
python test_tta.py --config configs/cartoon/selftrain.py \
  --checkpoint "$WORK/latest.pth" \
  --coco "$EVAL_COCO" --img-dir "$CHR_IMGS" --category "$CAT" \
  --shots 5 --trials 1 --seed 0 --steps 0 | tee "$ROOT/after_$TAG.txt" | tail -12

echo
echo "================ $TAG ================"
echo "BEFORE:"; grep -E "^PCK@|^EPE" "$ROOT/before_$TAG.txt"
echo "AFTER:";  grep -E "^PCK@|^EPE" "$ROOT/after_$TAG.txt"
echo
echo "Read the 'no TTA' column. Self-training is supposed to move THAT -- it"
echo "changed the weights, so the gain should be there without adapting at all."
echo "If PCK@0.2 rises while PCK@0.05 falls, the pseudo-labels taught position"
echo "but blurred precision: regenerate them with a tighter --eps."
