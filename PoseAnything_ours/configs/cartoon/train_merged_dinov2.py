# DA experiment: swap the ImageNet-Swin backbone for DINOv2 (self-supervised,
# far more transferable to the cartoon domain -> the "domain adaptation" pillar).
# Standalone (NOT inherited) so the DINOv2-specific constraints are explicit.
#
# Key constraints handled below:
#   * DINOv2 ViT-B/14 outputs 768-dim tokens  -> keypoint_head.in_channels=768 (unchanged, matches).
#   * DINOv2 patch size = 14  -> input image size MUST be divisible by 14.
#     256 is NOT (256/14=18.28) so we use 224 = 14*16. heatmap 56 = 224/4.
#   * load_from stays: the Swin backbone weights won't match DINOv2 key names
#     (ignored as "unexpected keys"); the HEAD weights DO match and load, so the
#     Graph-Transformer head still starts from the MP-100 checkpoint. Backbone
#     comes from DINOv2's torch.hub weights.
#
#   cd PoseAnything_ours && python setup.py develop
#   CUDA_VISIBLE_DEVICES=0 python train.py \
#     --config configs/cartoon/train_merged_dinov2.py \
#     --work-dir work_dirs/train_dinov2_100ep

NUM_SHOTS = 5
VALID_CLASS_IDS = [1, 2, 3, 4, 5]   # mickey/minnie dropped
DATA_DIR = '/home/subnh3/projects/QuynhAnh/cartoon-pose-detection/data/merged'

log_level = 'INFO'
load_from = 'cartoon_test_ckpt.pth'   # head loads; Swin-backbone keys are ignored
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
evaluation = dict(interval=5, metric=['PCK', 'AUC', 'EPE'],
                  key_indicator='PCK', save_best='PCK',
                  gpu_collect=True, res_folder='')
optimizer = dict(type='Adam', lr=1e-5)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', warmup='linear', warmup_iters=500,
                 warmup_ratio=0.001, step=[70, 90])
total_epochs = 100
log_config = dict(interval=10, hooks=[
    dict(type='TextLoggerHook'),
    dict(type='WandbLoggerHook',
         init_kwargs=dict(project='cartoon-cape',
                          entity='quynhanhhoang572004',
                          name='dinov2_100ep'),
         by_epoch=False),
])

channel_cfg = dict(num_output_channels=1, dataset_joints=1,
                   dataset_channel=[[0, ]], inference_channel=[0, ], max_kpt_num=100)

model = dict(
    type='PoseAnythingModel',
    pretrained='dinov2_vitb14',       # <-- DINOv2 backbone (was 'swinv2_small')
    encoder_config=dict(              # ignored when pretrained contains 'dino',
        type='SwinTransformerV2', embed_dim=96, depths=[2, 2, 18, 2],   # kept
        num_heads=[3, 6, 12, 24], window_size=16, drop_path_rate=0.3,   # harmless
        img_size=224, upsample="bilinear"),
    keypoint_head=dict(
        type='PoseHead',
        in_channels=768,              # DINOv2 ViT-B/14 token dim == 768
        keypoint_agg='attn',          # our aggregation (supporting component)
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
    image_size=[224, 224], heatmap_size=[56, 56],   # 224 = 14*16 (DINOv2 patch)
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
