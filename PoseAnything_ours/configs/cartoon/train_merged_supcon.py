# A/B run 2: attention aggregation + Contribution 3 (SupCon keypoint-identity
# contrastive loss). Inherits everything from train_merged.py (baseline attn,
# same data / valid_class_ids / schedule) and only flips SupCon on + renames the
# wandb run so the two curves sit side by side for the A/B.
#
#   cd PoseAnything_ours && python setup.py develop
#   CUDA_VISIBLE_DEVICES=0 python train.py \
#     --config configs/cartoon/train_merged_supcon.py \
#     --work-dir work_dirs/train_supcon_100ep
_base_ = 'train_merged.py'

# Contribution 3: turn on the SupCon loss inside PoseHead. mmcv deep-merges this
# into the base model.keypoint_head, so keypoint_agg='attn' etc. are preserved.
model = dict(
    keypoint_head=dict(
        with_contrast_loss=True,
        contrast_loss_weight=0.1,
        contrast_temp=0.1,
    )
)

# separate wandb run name (lists are replaced, not merged, so redefine in full).
log_config = dict(interval=10, hooks=[
    dict(type='TextLoggerHook'),
    dict(type='WandbLoggerHook',
         init_kwargs=dict(project='cartoon-cape',
                          entity='quynhanhhoang572004',
                          name='attn_supcon_100ep'),
         by_epoch=False),
])
