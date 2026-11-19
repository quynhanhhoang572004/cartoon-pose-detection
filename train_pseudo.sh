#!/usr/bin/env bash
# Self-train on a character's pseudo-labels, for as many rounds as you ask,
# then measure whether it helped.
#
#   bash train_pseudo.sh goofy 7                 # 1 round
#   bash train_pseudo.sh goofy 7 3               # 3 rounds, curriculum
#   bash train_pseudo.sh goofy 7 1 balanced      # 1 round, capped merge
#
# A round is: fit an adapter on the current model -> pseudo-label with it ->
# merge -> train -> that becomes the model the next round teaches with. Later
# rounds get better labels because the teacher got better, which is the whole
# point of doing more than one.
#
# Run `adapt_character.py --steps 1` first: it writes the support/query split
# this reads, using the same shuffle test_tta.py measures with.
set -euo pipefail

NAME=${1:-goofy}
CAT=${2:-7}
ROUNDS=${3:-1}
VARIANT=${4:-}                       # "" or e.g. "balanced"; see MAX_PER_CAT

# ------------------------- EDIT THESE -------------------------
ROOT=/home/subnh3/projects/QuynhAnh/cartoon-pose-detection
OURS=$ROOT/PoseAnything_ours
BASE=$OURS/work_dirs/selftrain_r4/best_PCK_epoch_40.pth   # round 1 starts here
ADAPT=$ROOT/data/adapt_$NAME                              # adapt_character.py --out-dir
LBL_COCO=$ROOT/data/merged/coco_train.json                # the existing real labels
LBL_IMGS=$ROOT/data/merged/images/train
CHR_IMGS=$ROOT/data/${NAME}_eval/images                   # its labelled frames
UNLAB=$ROOT/data/${NAME}_frames                           # its unlabelled frames
EVAL_COCO=$ROOT/data/coco_$NAME.json                      # all its labelled frames
EPOCHS=20
GPU=0
MAX_PER_CAT=""                       # set e.g. 150 for the "balanced" variant
FILTER="--n-subsets 3 --shots 3 --eps 0.05 --min-visible 4 --use-embedding --emb-thresh 0.35"
TTA="--steps 20 --lr 1e-3 --mode norm"
# --------------------------------------------------------------

[ "$VARIANT" = "balanced" ] && MAX_PER_CAT=${MAX_PER_CAT:-150}
TAG=$NAME${VARIANT:+_$VARIANT}
IMGDIR=$ROOT/data/selftrain_images_$NAME
SUP=$ADAPT/support.json
# the dataset never samples a category outside this list, so a missing $CAT
# means training completes, looking healthy, having learnt nothing new
IDS="[1,2,3,4,5,$CAT]"

echo "=== 0. checks ==="
for f in "$BASE" "$SUP" "$LBL_COCO" "$EVAL_COCO" "$UNLAB" "$CHR_IMGS"; do
  [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
done
echo "  ok | $ROUNDS round(s)${MAX_PER_CAT:+, merge capped at $MAX_PER_CAT/cat}"

echo "=== 1. one flat image dir ==="
mkdir -p "$IMGDIR"
for d in "$LBL_IMGS" "$CHR_IMGS" "$UNLAB"; do
  cp -n "$d"/*.jpg "$IMGDIR"/ 2>/dev/null || true
done
echo "  $(ls "$IMGDIR" | wc -l) images"

cd "$OURS"
python setup.py develop >/dev/null
export WANDB_MODE=offline MPLBACKEND=Agg

# Every measurement uses --trials 1 --seed 0: the split adapt_character.py
# wrote, so the support shots are the ones in training and the query images are
# the ones that are not. Another seed draws a different support set whose
# queries include images the model trained on.
# test_tta.py prints the no-TTA and with-TTA columns from one call, so each of
# these fills a row of the table.
measure () {   # measure <checkpoint> <outfile>
  python test_tta.py --config configs/cartoon/selftrain.py --checkpoint "$1" \
    --coco "$EVAL_COCO" --img-dir "$CHR_IMGS" --category "$CAT" \
    --shots 5 --trials 1 --seed 0 $TTA | tee "$2" | tail -12
}

echo "=== 2. BEFORE: the base model ==="
measure "$BASE" "$ROOT/measure_${TAG}_r0.txt"

CKPT=$BASE
for r in $(seq 1 "$ROUNDS"); do
  echo "================ $TAG  round $r / $ROUNDS ================"
  WORK=$OURS/work_dirs/selftrain_${TAG}_r$r
  ADAPTER=$ADAPT/${NAME}_adapter_r$r.pth
  PSEUDO=$ADAPT/pseudo_${NAME}_r$r.json
  MERGED=$ADAPT/train_${TAG}_r$r.json

  # a) fit an adapter on the CURRENT model and teach with it. It is ~0.08% of
  #    the parameters, so this is cheap, and a better teacher keeps far more
  #    labels through the filter (on Goofy: 490 frames against 44).
  echo "--- a) adapter for round $r ---"
  python test_tta.py --config configs/cartoon/selftrain.py --checkpoint "$CKPT" \
    --coco "$EVAL_COCO" --img-dir "$CHR_IMGS" --category "$CAT" \
    --shots 5 --trials 1 --seed 0 $TTA --save-adapted "$ADAPTER" | tail -4

  echo "--- b) pseudo-label $(ls "$UNLAB"/*.jpg | wc -l) frames ---"
  python generate_pseudo.py --config configs/cartoon/selftrain.py \
    --checkpoint "$CKPT" --adapter "$ADAPTER" \
    --support-coco "$SUP" --support-img-dir "$CHR_IMGS" --category "$CAT" \
    --unlabeled-dir "$UNLAB" --out "$PSEUDO" $FILTER | tail -3

  echo "--- c) merge ---"   # query.json is deliberately NOT merged in
  python "$ROOT/merge_coco.py" --out "$MERGED" \
    ${MAX_PER_CAT:+--max-per-cat $MAX_PER_CAT} \
    "$LBL_COCO" "$SUP" "$PSEUDO" | tail -3

  # d) train from the BASE every round, not from the previous round's weights.
  #    The teacher improves round over round, so the labels improve; restarting
  #    from the base keeps a round's result attributable to its labels instead
  #    of compounding whatever the previous round drifted into. Set START=$CKPT
  #    below to compound instead, which is what the old run_selftrain.sh did.
  echo "--- d) train ($EPOCHS epochs) ---"
  START=$BASE
  CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config configs/cartoon/selftrain.py --work-dir "$WORK" \
    --cfg-options load_from="$START" total_epochs=$EPOCHS \
      data.train.ann_file="$MERGED" \
      data.train.img_prefix="$IMGDIR/" \
      data.train.valid_class_ids="$IDS"

  echo "--- e) measure round $r ---"
  measure "$WORK/latest.pth" "$ROOT/measure_${TAG}_r$r.txt"
  CKPT=$WORK/latest.pth            # next round teaches with this
done

echo
echo "================ $TAG ================"
for f in "$ROOT"/measure_${TAG}_r*.txt; do
  echo "--- $(basename "$f" .txt)"
  grep -E "^PCK@|^EPE" "$f"
done
echo
echo "Each line is:  no TTA | with TTA | delta | unmasked."
echo "Self-training should move the 'no TTA' column -- it changed the weights,"
echo "so the gain must be there without adapting at all. Whether '+TTA' still"
echo "gains on top is the question of whether the two compose."
echo "Rounds should improve while the teacher does, then flatten. A round that"
echo "goes backwards means the labels got worse: tighten --eps in FILTER."
echo "If PCK@0.2 rises while PCK@0.05 falls, the labels taught position but"
echo "blurred precision -- same fix."
