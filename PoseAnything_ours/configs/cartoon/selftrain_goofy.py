# Self-training on Goofy: real labels (5 characters + 5 Goofy support shots)
# plus the pseudo-labels the BASE teacher produced for the 614 unlabelled Goofy
# frames.
#
# The 34 held-out Goofy query images are NOT in here and must stay out -- they
# are the only clean measurement of whether this worked.
#
# Build the inputs first (from the repo root):
#   python merge_coco.py --out data/coco_selftrain_goofy.json \
#       data/merged/coco_train.json \
#       data/coco_goofy_support5.json \
#       data/pseudo_goofy.json
#   mkdir -p data/selftrain_images
#   cp -n data/merged/images/train/*.jpg data/selftrain_images/
#   cp -n data/goofy_eval/images/*.jpg   data/selftrain_images/   # the 5 shots
#   cp -n data/goofy_batch1/*.jpg        data/selftrain_images/   # the 614
#
# Then:
#   cd PoseAnything_ours && python setup.py develop
#   CUDA_VISIBLE_DEVICES=0 python train.py \
#       --config configs/cartoon/selftrain_goofy.py \
#       --work-dir work_dirs/selftrain_goofy
_base_ = 'train_merged_supcon.py'

ROOT = '/home/subnh3/projects/QuynhAnh/cartoon-pose-detection'   # EDIT
MERGED = f'{ROOT}/data/coco_selftrain_goofy.json'
IMGDIR = f'{ROOT}/data/selftrain_images/'

# goofy is category 7 and the base config stops at 5, so without this line the
# dataset never samples a single Goofy episode and the whole run is a no-op.
TRAIN_CLASS_IDS = [1, 2, 3, 4, 5, 7]

# start from the model the pseudo-labels were generated with
load_from = f'{ROOT}/PoseAnything_ours/work_dirs/selftrain_r4/best_PCK_epoch_40.pth'  # EDIT

# Fixed budget, and NO model selection on Goofy. The val split holds no Goofy,
# so save_best='PCK' would pick the checkpoint that is best on the OTHER
# characters; and selecting on the 34 held-out Goofy images would spend the one
# clean measurement we have. Train a set number of epochs, take latest.pth.
total_epochs = 20
lr_config = dict(policy='step', warmup='linear', warmup_iters=200,
                 warmup_ratio=0.001, step=[12, 17])
checkpoint_config = dict(interval=5)
evaluation = dict(interval=5, metric=['PCK', 'AUC', 'EPE'],
                  key_indicator='PCK', save_best=None,
                  gpu_collect=True, res_folder='')

data = dict(
    train=dict(ann_file=MERGED, img_prefix=IMGDIR,
               valid_class_ids=TRAIN_CLASS_IDS),
)

log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
