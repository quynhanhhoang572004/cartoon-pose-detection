# A/B config for the support-aggregation study (PoseAnything_ours).
#
# Flip the two knobs below and rerun to compare, everything else identical:
#   KEYPOINT_AGG : 'mean'      -> stock PoseAnything baseline
#                  'weighted'  -> param-free, energy/visibility weighted
#                  'attn'      -> learnable attention over shots (our method)
#   NUM_SHOTS    : 1 / 5 / 10 / 20  (aggregation only matters when > 1)
#
# Suggested runs (leave everything else fixed):
#   for agg in mean weighted attn: for k in 5 10 20 -> work_dirs/agg_${agg}_${k}shot
#   (K=1 is a sanity check: all three must give ~identical PCK.)
#
# Train:  python train.py --config configs/cartoon/graph_agg.py --work-dir work_dirs/agg_attn_5shot
# Test :  python test.py  configs/cartoon/graph_agg.py work_dirs/agg_attn_5shot/latest.pth
#
# NOTE the new 'shot_attn' layer (agg='attn') is not in the pretrained
# checkpoint, so expect exactly one 'missing key: keypoint_head.shot_attn.*'
# message on load. Anything else missing/unexpected means a wiring bug.

KEYPOINT_AGG = 'attn'   # <-- flip to 'mean' for the baseline
NUM_SHOTS = 5           # <-- 1 / 5 / 10 / 20

log_level = 'INFO'
load_from = 'cartoon_test_ckpt.pth'   # point at your MP-100 pretrained checkpoint
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
evaluation = dict(
    interval=10,
    metric=['PCK', 'NME', 'AUC', 'EPE'],
    key_indicator='PCK',
    gpu_collect=True,
    res_folder='')
optimizer = dict(type='Adam', lr=1e-5)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step', warmup='linear', warmup_iters=500,
    warmup_ratio=0.001, step=[70, 90])
total_epochs = 100
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])

channel_cfg = dict(
    num_output_channels=1,
    dataset_joints=1,
    dataset_channel=[[0, ]],
    inference_channel=[0, ],
    max_kpt_num=100)

model = dict(
    type='PoseAnythingModel',
    pretrained='swinv2_small',
    encoder_config=dict(
        type='SwinTransformerV2',
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=16,
        drop_path_rate=0.3,
        img_size=256,
        upsample="bilinear"),
    keypoint_head=dict(
        type='PoseHead',
        in_channels=768,
        keypoint_agg=KEYPOINT_AGG,      # <-- the only method-level change
        transformer=dict(
            type='EncoderDecoder',
            d_model=256,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            graph_decoder='pre',
            dim_feedforward=768,
            dropout=0.1,
            similarity_proj_dim=256,
            dynamic_proj_dim=128,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=True),
        share_kpt_branch=False,
        num_decoder_layer=3,
        with_heatmap_loss=True,
        heatmap_loss_weight=2.0,
        support_order_dropout=-1,
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=False,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetRandomScaleRotation', rot_factor=15, scale_factor=0.15),
    dict(type='TopDownAffineFewShot'),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor',
         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTargetFewShot', sigma=1),
    dict(type='Collect',
         keys=['img', 'target', 'target_weight'],
         meta_keys=['image_file', 'joints_3d', 'joints_3d_visible', 'center',
                    'scale', 'rotation', 'bbox_score', 'flip_pairs',
                    'category_id', 'skeleton']),
]

valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffineFewShot'),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor',
         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTargetFewShot', sigma=1),
    dict(type='Collect',
         keys=['img', 'target', 'target_weight'],
         meta_keys=['image_file', 'joints_3d', 'joints_3d_visible', 'center',
                    'scale', 'rotation', 'bbox_score', 'flip_pairs',
                    'category_id', 'skeleton']),
]

test_pipeline = valid_pipeline

data_root = 'data/mp100'
img_prefix = '/home/subnh3/projects/QuynhAnh/Cartoon-Pose-Guided-Image-Synthesis/model_phase1/data/images/train/'
val_img_prefix = '/home/subnh3/projects/QuynhAnh/Cartoon-Pose-Guided-Image-Synthesis/model_phase1/data/images/val/'

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='TransformerPoseDataset',
        ann_file=f'{data_root}/annotations/cartoon_train.json',
        img_prefix=img_prefix,
        data_cfg=data_cfg,
        valid_class_ids=None,
        max_kpt_num=channel_cfg['max_kpt_num'],
        num_shots=NUM_SHOTS,
        pipeline=train_pipeline),
    val=dict(
        type='TransformerPoseDataset',
        ann_file=f'{data_root}/annotations/cartoon_val.json',
        img_prefix=val_img_prefix,
        data_cfg=data_cfg,
        valid_class_ids=None,
        max_kpt_num=channel_cfg['max_kpt_num'],
        num_shots=NUM_SHOTS,
        num_queries=15,
        num_episodes=50,
        pipeline=valid_pipeline),
    test=dict(
        type='TestPoseDataset',
        ann_file=f'{data_root}/annotations/cartoon_val.json',
        img_prefix=val_img_prefix,
        data_cfg=data_cfg,
        valid_class_ids=None,
        max_kpt_num=channel_cfg['max_kpt_num'],
        num_shots=NUM_SHOTS,
        num_queries=15,
        num_episodes=100,
        pck_threshold_list=[0.05, 0.10, 0.15, 0.2, 0.25],
        pipeline=test_pipeline),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
shuffle_cfg = dict(interval=1)
