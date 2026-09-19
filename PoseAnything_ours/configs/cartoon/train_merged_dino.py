# DA experiment (Python-3.8-safe): swap the ImageNet-Swin backbone for DINO v1
# (self-supervised). DINOv2's `main` on torch.hub uses Python-3.10-only syntax
# (`float | None`) that the pose-anything env (Py 3.8) cannot parse, so we use
# DINO v1 instead. ViT-B/8 outputs 768-dim tokens (matches in_channels=768) and
# uses patch size 8, so the existing 256x256 / heatmap-64 setup works unchanged
# -> only the backbone differs from the Swin baseline (clean single-variable A/B).
#
#   cd PoseAnything_ours && python setup.py develop
#   CUDA_VISIBLE_DEVICES=0 python train.py \
#     --config configs/cartoon/train_merged_dino.py \
#     --work-dir work_dirs/train_dino_100ep
_base_ = 'train_merged.py'

# 'dino' (not 'dinov2') -> pam.py loads facebookresearch/dino:main, patch-8.
# 'dino_vitb8' is 768-dim; image stays 256 (from base) since 256/8 = 32.
model = dict(pretrained='dino_vitb8')

log_config = dict(interval=10, hooks=[
    dict(type='TextLoggerHook'),
    dict(type='WandbLoggerHook',
         init_kwargs=dict(project='cartoon-cape',
                          entity='quynhanhhoang572004',
                          name='dino_vitb8_100ep'),
         by_epoch=False),
])
