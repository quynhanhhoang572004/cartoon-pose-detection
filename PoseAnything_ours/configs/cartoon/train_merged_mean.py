# Ablation baseline: original mean aggregation (keypoint_agg='mean'). Inherits
# everything from train_merged.py and only switches attn -> mean. With 'mean' no
# shot_attn layer is created, so this matches the ORIGINAL PoseAnything head and
# can load a mean-trained checkpoint (e.g. PoseAnything/work_dirs/train_old).
#
#   python test.py configs/cartoon/train_merged_mean.py \
#     /abs/path/PoseAnything/work_dirs/train_old/best_PCK_epoch_30.pth
_base_ = 'train_merged.py'

model = dict(keypoint_head=dict(keypoint_agg='mean'))
