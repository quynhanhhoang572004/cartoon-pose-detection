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
# One command does all of it, including the support/query split on first run.
# Everything a round writes is tagged with the round number, so a run can be
# resumed or compared against an earlier one without overwriting it.
set -euo pipefail

NAME=${1:-goofy}
CAT=${2:-7}
ROUNDS=${3:-1}
VARIANT=${4:-}                       # "" or e.g. "balanced"; see MAX_PER_CAT

# ---- defaults, each overridable from the environment ----
# Every value below takes an env var, so the same script runs on the server and
# on Kaggle (where /kaggle/input is read-only and the paths differ) without
# anyone editing it and then committing their machine's paths.
#     ROOT=/kaggle/working CHR_IMGS=... bash train_pseudo.sh goofy 7 2
ROOT=${ROOT:-/home/subnh3/projects/QuynhAnh/cartoon-pose-detection}
OURS=${OURS:-$ROOT/PoseAnything_ours}
OUT=${OUT:-$ROOT}                                         # where results are written
BASE=${BASE:-$OURS/work_dirs/selftrain_r4/best_PCK_epoch_40.pth}   # round 1 starts here
ADAPT=${ADAPT:-$OUT/adapt_$NAME}                          # split + adapters + pseudo
LBL_COCO=${LBL_COCO:-$ROOT/data/merged/coco_train.json}   # the existing real labels
LBL_IMGS=${LBL_IMGS:-$ROOT/data/merged/images/train}
CHR_IMGS=${CHR_IMGS:-$ROOT/data/${NAME}_eval/images}      # its labelled frames
UNLAB=${UNLAB:-$ROOT/data/${NAME}_frames}                 # its unlabelled frames
EVAL_COCO=${EVAL_COCO:-$ROOT/data/coco_$NAME.json}        # all its labelled frames
# Validation during the rounds is optional: the val split holds only the
# source characters, so it says "did we forget them", not "did the new one
# improve". When it is missing -- which it is anywhere but the server -- train
# with --no-validate rather than dying on a path the run does not need.
VAL_COCO=${VAL_COCO:-$ROOT/data/merged/coco_val.json}
VAL_IMGS=${VAL_IMGS:-$ROOT/data/merged/images/val}
IMGDIR=${IMGDIR:-$OUT/selftrain_images_$NAME}             # one flat dir for img_prefix
EPOCHS=${EPOCHS:-20}
GPU=${GPU:-0}                 # which GPU the single-process steps use
GPUS=${GPUS:-1}               # processes for TRAINING only; >1 uses torch.distributed
PORT=${PORT:-29500}           # raise it if a previous run left the port bound
# Per-process batch, chosen so the EFFECTIVE batch stays 2 however many GPUs there
# are. Changing the effective batch changes training dynamics, and a 2-GPU round
# would then not be comparable with a 1-GPU one -- which defeats the point of
# running rounds at all. Override only deliberately.
SPG=${SPG:-$(( 2 / GPUS > 0 ? 2 / GPUS : 1 ))}
CUMULATIVE=${CUMULATIVE:-1}   # 1: each round trains on from the last (Noisy Student).
                              # 0: each round restarts from BASE.
MAX_PER_CAT=${MAX_PER_CAT:-}                 # set e.g. 150 for the "balanced" variant
FILTER=${FILTER:-"--n-subsets 3 --shots 3 --eps 0.05 --min-visible 4 --use-embedding --emb-thresh 0.35"}
TTA=${TTA:-"--steps 20 --lr 1e-3 --mode norm"}
# The interpreter to run everything with. On Kaggle the repo's stack lives in a
# separate venv, and relying on PATH is fragile -- name it instead:
#   PY=/kaggle/temp/env/bin/python bash train_pseudo.sh goofy 7 2
PY=${PY:-python}
# ---------------------------------------------------------

[ "$VARIANT" = "balanced" ] && MAX_PER_CAT=${MAX_PER_CAT:-150}
TAG=$NAME${VARIANT:+_$VARIANT}
SUP=$ADAPT/support.json
# the dataset never samples a category outside this list, so a missing $CAT
# means training completes, looking healthy, having learnt nothing new
IDS="[1,2,3,4,5,$CAT]"

echo "=== 0. checks ==="
for f in "$BASE" "$LBL_COCO" "$EVAL_COCO" "$UNLAB" "$CHR_IMGS"; do
  [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
done
"$PY" -c "import mmcv, mmpose, torch" 2>/dev/null || {
  echo "MISSING: \$PY ($PY) cannot import mmcv/mmpose/torch."
  echo "         Point PY at the interpreter that has them, e.g."
  echo "         PY=/kaggle/temp/env/bin/python bash train_pseudo.sh $NAME $CAT $ROUNDS"
  exit 1; }
echo "  ok | $ROUNDS round(s)${MAX_PER_CAT:+, merge capped at $MAX_PER_CAT/cat} | $PY"
echo "     $PY | train on $GPUS GPU(s) x $SPG = effective batch $((GPUS * SPG))"

cd "$OURS"
# Non-fatal: every call below runs from this directory, so `import models`
# resolves from cwd whether or not the package is registered. setup.py develop
# also drags in the repo's requirements.txt, whose unpinned opencv-python wants
# numpy>=2 and so fails the whole run over something it does not need.
"$PY" setup.py develop >/dev/null 2>&1 || echo "  (setup.py develop failed; cwd import is enough)"
export WANDB_MODE=offline MPLBACKEND=Agg

# The support/query split, if it is not there yet. Written once and reused by
# every round and every measurement: the shots in it are the ones that end up
# in training, and the queries are the ones that never do. Re-splitting between
# rounds would quietly move images across that line.
if [ ! -e "$SUP" ]; then
  echo "=== 0b. support/query split ==="
  mkdir -p "$ADAPT"
  "$PY" adapt_character.py --name "$NAME" --category "$CAT" \
    --config configs/cartoon/selftrain.py --checkpoint "$BASE" \
    --coco "$EVAL_COCO" --img-dir "$CHR_IMGS" --out-dir "$ADAPT" --steps 1
fi

echo "=== 1. one flat image dir ==="
mkdir -p "$IMGDIR"
for d in "$LBL_IMGS" "$CHR_IMGS" "$UNLAB"; do
  cp -n "$d"/*.jpg "$IMGDIR"/ 2>/dev/null || true
done
echo "  $(ls "$IMGDIR" | wc -l) images"

# Every measurement uses --trials 1 --seed 0: the split adapt_character.py
# wrote, so the support shots are the ones in training and the query images are
# the ones that are not. Another seed draws a different support set whose
# queries include images the model trained on.
# test_tta.py prints the no-TTA and with-TTA columns from one call, so each of
# these fills a row of the table.
measure () {   # measure <checkpoint> <outfile>
  "$PY" test_tta.py --config configs/cartoon/selftrain.py --checkpoint "$1" \
    --coco "$EVAL_COCO" --img-dir "$CHR_IMGS" --category "$CAT" \
    --shots 5 --trials 1 --seed 0 $TTA | tee "$2" | tail -12
}

echo "=== 2. BEFORE: the base model ==="
measure "$BASE" "$OUT/measure_${TAG}_r0.txt"

CKPT=$BASE
for r in $(seq 1 "$ROUNDS"); do
  echo "================ $TAG  round $r / $ROUNDS ================"
  WORK=$OUT/work_dirs/selftrain_${TAG}_r$r
  ADAPTER=$ADAPT/${NAME}_adapter_r$r.pth
  PSEUDO=$ADAPT/pseudo_${NAME}_r$r.json
  MERGED=$ADAPT/train_${TAG}_r$r.json

  # a) fit an adapter on the CURRENT model and teach with it. It is ~0.08% of
  #    the parameters, so this is cheap, and a better teacher keeps far more
  #    labels through the filter (on Goofy: 490 frames against 44).
  echo "--- a) adapter for round $r ---"
  "$PY" test_tta.py --config configs/cartoon/selftrain.py --checkpoint "$CKPT" \
    --coco "$EVAL_COCO" --img-dir "$CHR_IMGS" --category "$CAT" \
    --shots 5 --trials 1 --seed 0 $TTA --save-adapted "$ADAPTER" | tail -4

  echo "--- b) pseudo-label $(ls "$UNLAB"/*.jpg | wc -l) frames ---"
  "$PY" generate_pseudo.py --config configs/cartoon/selftrain.py \
    --checkpoint "$CKPT" --adapter "$ADAPTER" \
    --support-coco "$SUP" --support-img-dir "$CHR_IMGS" --category "$CAT" \
    --unlabeled-dir "$UNLAB" --out "$PSEUDO" $FILTER | tail -3

  echo "--- c) merge ---"   # query.json is deliberately NOT merged in
  "$PY" "$ROOT/merge_coco.py" --out "$MERGED" \
    ${MAX_PER_CAT:+--max-per-cat $MAX_PER_CAT} \
    "$LBL_COCO" "$SUP" "$PSEUDO" | tail -3

  # d) train on from the previous round's weights (CUMULATIVE=1, the default)
  #    or restart from the base each round (CUMULATIVE=0).
  #
  #    Cumulative is what Noisy Student does, and the reason is arithmetic:
  #    restarting makes round N just "base + EPOCHS epochs on round-N labels",
  #    so the only thing that ever improves is the labels. Once those saturate
  #    -- which they do after a round or two -- the later rounds buy nothing.
  #    Training on compounds instead.
  #
  #    What it costs is confirmation bias: the model reinforcing its own
  #    mistakes round after round. That is what the per-round measurement and
  #    the regression stop below are for.
  echo "--- d) train ($EPOCHS epochs) ---"
  [ "$CUMULATIVE" = "1" ] && START=$CKPT || START=$BASE
  echo "    starting from: $START"
  CFG_OPTS=(load_from="$START" total_epochs=$EPOCHS
            data.samples_per_gpu=$SPG
            data.train.ann_file="$MERGED"
            data.train.img_prefix="$IMGDIR/"
            data.train.valid_class_ids="$IDS")
  if [ -e "$VAL_COCO" ]; then
    CFG_OPTS+=(data.val.ann_file="$VAL_COCO" data.val.img_prefix="$VAL_IMGS/")
    NOVAL=""
  else
    echo "    no val set at $VAL_COCO -- training with --no-validate"
    NOVAL="--no-validate"
  fi
  if [ "$GPUS" -gt 1 ]; then
    echo "    $GPUS GPUs x $SPG per GPU = effective batch $((GPUS * SPG))"
    "$PY" -m torch.distributed.run --nproc_per_node="$GPUS" --master_port="$PORT"       train.py --config configs/cartoon/selftrain.py --work-dir "$WORK"       --launcher pytorch $NOVAL --cfg-options "${CFG_OPTS[@]}"
  else
    CUDA_VISIBLE_DEVICES=$GPU "$PY" train.py       --config configs/cartoon/selftrain.py --work-dir "$WORK"       $NOVAL --cfg-options "${CFG_OPTS[@]}"
  fi

  echo "--- e) measure round $r ---"
  measure "$WORK/latest.pth" "$OUT/measure_${TAG}_r$r.txt"

  # Stop if this round went backwards. Training on compounds whatever the
  # previous round became, so carrying a bad round forward poisons every round
  # after it -- and the useful artefact is the last round that improved, which
  # $CKPT still points at.
  prev=$(grep -oP '^PCK@0.05\s+\K[0-9.]+' "$OUT/measure_${TAG}_r$((r-1)).txt" 2>/dev/null || echo 0)
  now=$(grep -oP '^PCK@0.05\s+\K[0-9.]+' "$OUT/measure_${TAG}_r$r.txt" 2>/dev/null || echo 0)
  if awk "BEGIN{exit !($now < $prev - 0.005)}"; then
    echo "!! round $r PCK@0.05 $now < round $((r-1)) $prev -- the labels got worse."
    echo "!! stopping; keeping round $((r-1)). Tighten --eps in FILTER and rerun."
    break
  fi
  echo "    round $r: PCK@0.05 $prev -> $now"
  CKPT=$WORK/latest.pth            # next round teaches with this
done

echo
echo "================ $TAG ================"
for f in "$OUT"/measure_${TAG}_r*.txt; do
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
