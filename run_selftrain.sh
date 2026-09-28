#!/usr/bin/env bash
# Automatic self-training loop for cartoon CAPE.
#   generate pseudo-labels (support-ensemble + embedding filter) -> merge with
#   the real labels -> retrain from the current model -> repeat. Each round
#   teaches with the newest checkpoint (curriculum). NEEDS one shakeout run:
#   verify paths + that train.py accepts the --cfg-options overrides below.
set -euo pipefail

# ------------------------- EDIT THESE -------------------------
ROOT=/home/subnh3/projects/QuynhAnh/cartoon-pose-detection
OURS=$ROOT/PoseAnything_ours
LBL_COCO=$ROOT/data/merged/coco_train.json          # labeled support + supervision
LBL_IMGS=$ROOT/data/merged/images/train
UNLAB=$ROOT/data/unlabeled                          # bugs_bunny/ pink_panther/ sylvester/
CONFIG=$OURS/configs/cartoon/train_merged_supcon.py
CKPT=$OURS/work_dirs/train_supcon_100ep/best_PCK_epoch_70.pth   # base model (round 0)
ROUNDS=2
EPOCHS=30                                           # epochs per round
GPU=0
FILTER="--n-subsets 3 --shots 5 --eps 0.03 --min-visible 6 --use-embedding --emb-thresh 0.25"
declare -A CATS=( [bugs_bunny]=3 [pink_panther]=4 [sylvester]=5 )   # name -> category id
# --------------------------------------------------------------

cd "$OURS"
python setup.py develop

# the config's WandB hook uses an entity that 404s; keep runs offline (sync later
# with `wandb sync`), or comment the WandbLoggerHook out of the config.
export WANDB_MODE=offline

# One flat image dir holding the labeled images AND the unlabeled frames, so a
# single img_prefix resolves every file_name in the merged COCO. Built once.
IMGDIR=$ROOT/data/selftrain_images
mkdir -p "$IMGDIR"
cp -n "$LBL_IMGS"/*.jpg "$IMGDIR"/ 2>/dev/null || true
for char in "${!CATS[@]}"; do
  cp -n "$UNLAB/$char"/*.jpg "$IMGDIR"/ 2>/dev/null || true
done
echo "[setup] combined image dir: $(ls "$IMGDIR" | wc -l) images"

for r in $(seq 1 "$ROUNDS"); do
  echo "================ SELF-TRAINING ROUND $r ================"
  PSEUDO=$ROOT/data/pseudo_r$r ; mkdir -p "$PSEUDO"
  WORK=$OURS/work_dirs/selftrain_r$r

  # 1) pseudo-labels per character, using the NEWEST checkpoint as teacher
  PSEUDO_JSONS=""
  for char in "${!CATS[@]}"; do
    echo "--- pseudo-labelling $char (cat ${CATS[$char]}) ---"
    python generate_pseudo.py --config "$CONFIG" --checkpoint "$CKPT" \
      --support-coco "$LBL_COCO" --support-img-dir "$LBL_IMGS" \
      --category "${CATS[$char]}" \
      --unlabeled-dir "$UNLAB/$char" \
      --out "$PSEUDO/pseudo_$char.json" \
      $FILTER
    PSEUDO_JSONS="$PSEUDO_JSONS $PSEUDO/pseudo_$char.json"
  done

  # 2) merge real labels + all pseudo-labels into one training COCO
  MERGED=$ROOT/data/coco_selftrain_r$r.json
  python "$ROOT/merge_coco.py" --out "$MERGED" "$LBL_COCO" $PSEUDO_JSONS

  # 3) retrain from the current checkpoint on labels + pseudo
  CUDA_VISIBLE_DEVICES=$GPU python train.py --config "$CONFIG" --work-dir "$WORK" \
    --cfg-options load_from="$CKPT" total_epochs=$EPOCHS \
      data.train.ann_file="$MERGED" "data.train.img_prefix=$IMGDIR/"

  # 4) next round teaches with THIS round's best checkpoint
  CKPT=$(ls -t "$WORK"/best_PCK_* 2>/dev/null | head -1)
  echo "round $r done -> new teacher: $CKPT"
done

echo "ALL DONE. Final self-trained model: $CKPT"
echo "Evaluate it:  python test.py $CONFIG $CKPT"
