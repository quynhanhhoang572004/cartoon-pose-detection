# 1-shot fine-tune config for Tom & Jerry cartoon pose.
# Baseline (2) in the plan: init from the MP-100 pretrained checkpoint, then
# fine-tune on cartoon data. Compare its PCK against the zero-shot baseline.
#
# Before training:
#   1) convert data:
#        python tools/yolo2coco_pose.py --data-yaml <data.yaml> --split train \
#            --out data/mp100/annotations/cartoon_train.json
#        python tools/yolo2coco_pose.py --data-yaml <data.yaml> --split val \
#            --out data/mp100/annotations/cartoon_val.json
#   2) put the query/support images where img_prefix points (see `img_prefix`)
#   3) make sure `load_from` points at your MP-100 checkpoint
#
# Train:
#   python train.py --config configs/cartoon/graph_1shot.py --work-dir work_dirs/cartoon
# Test:
#   python test.py configs/cartoon/graph_1shot.py work_dirs/cartoon/latest.pth

log_level = 'INFO'
# init from the MP-100 Small-Swin 1-shot checkpoint (backbone + head)
load_from = 'cartoon_test_ckpt.pth'
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
optimizer = dict(
    type='Adam',
    lr=1e-5,  # low LR for fine-tuning
)

optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[70, 90])
total_epochs = 100  # fewer than the 200 used for from-scratch MP-100 training
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=1,
    dataset_joints=1,
    dataset_channel=[[0, ]],
    inference_channel=[0, ],
    max_kpt_num=100)  # your 21 kpts are padded to 100 + masked; keep 100

# model settings — Small SwinV2 (matches cartoon_test_ckpt.pth)
model = dict(
    type='PoseAnythingModel',
    # 'swinv2_small' (no .pth) only builds the arch; load_from fills the weights,
    # so no separate backbone file is needed.
    pretrained='swinv2_small',
    encoder_config=dict(
        type='SwinTransformerV2',
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=16,
        drop_path_rate=0.3,
        img_size=256,
        upsample="bilinear"
    ),
    keypoint_head=dict(
        type='PoseHead',
        in_channels=768,
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
# NOTE: yolo2coco_pose.py writes only file_name (no subdir), so img_prefix must
# be the folder that directly contains the images. Point it at your YOLO images
# dir. Adjust to your actual path.
img_prefix = '/home/subnh3/projects/QuynhAnh/Cartoon-Pose-Guided-Image-Synthesis/model_phase1/data/images/train/'
val_img_prefix = '/home/subnh3/projects/QuynhAnh/Cartoon-Pose-Guided-Image-Synthesis/model_phase1/data/images/val/'

data = dict(
    samples_per_gpu=8,   # lower to 4/2 if you hit GPU OOM
    workers_per_gpu=4,
    train=dict(
        type='TransformerPoseDataset',
        ann_file=f'{data_root}/annotations/cartoon_train.json',
        img_prefix=img_prefix,
        data_cfg=data_cfg,
        valid_class_ids=None,   # None = use all categories (tom + jerry)
        max_kpt_num=channel_cfg['max_kpt_num'],
        num_shots=1,
        pipeline=train_pipeline),
    val=dict(
        type='TransformerPoseDataset',
        ann_file=f'{data_root}/annotations/cartoon_val.json',
        img_prefix=val_img_prefix,
        data_cfg=data_cfg,
        valid_class_ids=None,
        max_kpt_num=channel_cfg['max_kpt_num'],
        num_shots=1,
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
        num_shots=1,
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
