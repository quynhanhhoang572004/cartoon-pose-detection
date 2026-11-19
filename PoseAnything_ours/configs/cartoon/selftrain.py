# Self-training on ONE new character: the existing real labels, plus that
# character's few support shots, plus the pseudo-labels its teacher produced for
# its unlabelled frames.
#
# Character-agnostic on purpose -- the three things that change per character
# come in through --cfg-options, so the same config serves goofy, mickey, or
# whoever is next. `python adapt_character.py ... --steps 6` prints the command
# filled in for you.
#
#   CUDA_VISIBLE_DEVICES=0 python train.py \
#       --config configs/cartoon/selftrain.py \
#       --work-dir work_dirs/selftrain_<name> \
#       --cfg-options load_from="<base_ckpt>" \
#         data.train.ann_file="<merged.json>" \
#         data.train.img_prefix="<flat_image_dir>/" \
#         data.train.valid_class_ids="[1,2,3,4,5,7]"
#
# The LAST of those four is the one that silently ruins the run if forgotten:
# the base config stops at class 5, so a new character is never sampled and
# training completes, looking perfectly healthy, having learnt nothing about it.
_base_ = 'train_merged_supcon.py'

# Fixed budget, and NO model selection on the new character. The val split holds
# only the original characters, so save_best='PCK' picks the checkpoint that is
# best on THEM; and selecting on the new character's held-out query images would
# spend the one clean measurement there is. Train a set number of epochs, take
# latest.pth, measure once.
total_epochs = 20
lr_config = dict(policy='step', warmup='linear', warmup_iters=200,
                 warmup_ratio=0.001, step=[12, 17])
checkpoint_config = dict(interval=5)
evaluation = dict(interval=5, metric=['PCK', 'AUC', 'EPE'],
                  key_indicator='PCK', save_best=None,
                  gpu_collect=True, res_folder='')

# Pseudo-labels are noisier than hand labels, so keep the LR low: this run is
# meant to sharpen an already-trained model, not to move it far.
optimizer = dict(type='Adam', lr=1e-5)

log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
