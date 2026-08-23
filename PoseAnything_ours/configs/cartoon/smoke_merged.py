# SMOKE TEST -- NEW architecture (variance-aware aggregation, keypoint_agg='attn').
# Runs on the MERGED dataset (Tom & Jerry + 4 cartoon characters).
# This config lives in PoseAnything_ours/, so keypoint_head sets keypoint_agg.
# Pair it with PoseAnything/configs/cartoon/smoke_merged.py (old arch) for the A/B.
#
#   cd PoseAnything_ours && python setup.py develop
#   python train.py --config configs/cartoon/smoke_merged.py --work-dir work_dirs/smoke_new
#   python test.py  configs/cartoon/smoke_merged.py work_dirs/smoke_new/latest.pth
#
# On load you will see exactly one 'missing key: keypoint_head.shot_attn.*'
# (the new learnable param) -- that is expected.

NUM_SHOTS = 5
# all 6 categories have data AFTER merging in Tom & Jerry. If T&J is NOT merged,
# drop 1,2 -> [3,4,5,6].
VALID_CLASS_IDS = [1, 2, 3, 4, 5, 6]
DATA_DIR = '/home/subnh3/projects/QuynhAnh/cartoon-pose-detection/data/merged'   # <-- EDIT: dir holding coco_train.json/coco_val.json + images/

log_level = 'INFO'
load_from = '/home/subnh3/projects/QuynhAnh/cartoon-pose-detection/PoseAnything/cartoon_test_ckpt.pth'
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric=['PCK', 'AUC', 'EPE'],
                  key_indicator='PCK', gpu_collect=True, res_folder='')
optimizer = dict(type='Adam', lr=1e-5)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', warmup='linear', warmup_iters=10,
                 warmup_ratio=0.001, step=[999])
total_epochs = 1
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])

channel_cfg = dict(num_output_channels=1, dataset_joints=1,
                   dataset_channel=[[0, ]], inference_channel=[0, ], max_kpt_num=100)

model = dict(
    type='PoseAnythingModel',
    pretrained='swinv2_small',
    encoder_config=dict(
        type='SwinTransformerV2', embed_dim=96, depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24], window_size=16, drop_path_rate=0.3,
        img_size=256, upsample="bilinear"),
    keypoint_head=dict(
        type='PoseHead',
        in_channels=768,
        keypoint_agg='attn',          # <-- NEW architecture
        transformer=dict(
            type='EncoderDecoder', d_model=256, nhead=8,
            num_encoder_layers=3, num_decoder_layers=3, graph_decoder='pre',
            dim_feedforward=768, dropout=0.1, similarity_proj_dim=256,
            dynamic_proj_dim=128, activation="relu", normalize_before=False,
            return_intermediate_dec=True),
        share_kpt_branch=False, num_decoder_layer=3,
        with_heatmap_loss=True, heatmap_loss_weight=2.0, support_order_dropout=-1,
        positional_encoding=dict(type='SinePositionalEncoding', num_feats=128, normalize=True)),
    train_cfg=dict(),
    test_cfg=dict(flip_test=False, post_process='default',
                  shift_heatmap=True, modulate_kernel=11))

data_cfg = dict(
    image_size=[256, 256], heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetRandomScaleRotation', rot_factor=15, scale_factor=0.15),
    dict(type='TopDownAffineFewShot'),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTargetFewShot', sigma=1),
    dict(type='Collect', keys=['img', 'target', 'target_weight'],
         meta_keys=['image_file', 'joints_3d', 'joints_3d_visible', 'center',
                    'scale', 'rotation', 'bbox_score', 'flip_pairs',
                    'category_id', 'skeleton']),
]
valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffineFewShot'),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTargetFewShot', sigma=1),
    dict(type='Collect', keys=['img', 'target', 'target_weight'],
         meta_keys=['image_file', 'joints_3d', 'joints_3d_visible', 'center',
                    'scale', 'rotation', 'bbox_score', 'flip_pairs',
                    'category_id', 'skeleton']),
]
test_pipeline = valid_pipeline

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='TransformerPoseDataset',
        ann_file=f'{DATA_DIR}/coco_train.json',
        img_prefix=f'{DATA_DIR}/images/train/',
        data_cfg=data_cfg, valid_class_ids=VALID_CLASS_IDS,
        max_kpt_num=channel_cfg['max_kpt_num'], num_shots=NUM_SHOTS,
        pipeline=train_pipeline),
    val=dict(
        type='TransformerPoseDataset',
        ann_file=f'{DATA_DIR}/coco_val.json',
        img_prefix=f'{DATA_DIR}/images/val/',
        data_cfg=data_cfg, valid_class_ids=VALID_CLASS_IDS,
        max_kpt_num=channel_cfg['max_kpt_num'], num_shots=NUM_SHOTS,
        num_queries=5, num_episodes=5, pipeline=valid_pipeline),
    test=dict(
        type='TestPoseDataset',
        ann_file=f'{DATA_DIR}/coco_val.json',
        img_prefix=f'{DATA_DIR}/images/val/',
        data_cfg=data_cfg, valid_class_ids=VALID_CLASS_IDS,
        max_kpt_num=channel_cfg['max_kpt_num'], num_shots=NUM_SHOTS,
        num_queries=5, num_episodes=5,
        pck_threshold_list=[0.05, 0.10, 0.15, 0.2, 0.25], pipeline=test_pipeline),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
shuffle_cfg = dict(interval=1)
