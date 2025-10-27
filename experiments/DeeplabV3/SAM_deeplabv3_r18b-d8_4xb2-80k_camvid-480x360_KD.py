_base_ = [
    '../../configs/_base_/models/upernet_vit-b16_ln_mln.py', '../../configs/_base_/datasets/camvid.py',
    '../../configs/_base_/default_runtime.py', '../../configs/_base_/schedules/schedule_80k.py'
]

crop_size = (480, 360)
data_preprocessor = dict(size=crop_size)
### 1. SAM config
arch = 'base'
SAM_dict = dict(
    type='SAM',
    image_encoder_cfg=dict(
        type='mmpretrain.ViTSAM',
        arch=arch,
        img_size=crop_size[0],
        patch_size=16,
        out_channels=256,
        out_indices=(2, 5, 8, 11),
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
    ),
    prompt_encoder_cfg=dict(
        type='PromptEncoder',
        embed_dim=256,
        image_embedding_size=(crop_size[0]//4, crop_size[1]//4),
        input_image_size=crop_size,
        mask_in_chans=16,
    ),
    mask_decoder_cfg=dict(
        type='MaskDecoder',
        num_multimask_outputs=3,
        transformer=dict(
            type='TwoWayTransformer',
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
            ),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    ))


### 2. training framework arch
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    #type='EncoderDecoderwithSAM_KD',
    type='EncoderDecoderwithSAMLoRA',
    data_preprocessor=data_preprocessor,
    backbone=None,
    SAM_config=SAM_dict,
    SAM_arch=arch,
    neck=dict(
        type='MultiLevelNeck',
        in_channels=[256, 256, 256, 256],
        out_channels=768,
        scales=[4, 2, 1, 0.5]),
    decode_head=dict(type='UPerAttHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)),
    auxiliary_head=None,
    Prompt_backbone=dict(
        type='ResNetV1c',
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet18_v1c'),
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    Prompt_head=dict(
        type='ASPPHead',
        in_channels=512,
        in_index=3,
        channels=128,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)),
    Prompt_auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)),
    test_cfg=dict(mode='slide', crop_size=(512, 1024), stride=(768, 768))
    #test_cfg=dict(mode='whole')
    )


work_dir = './experiments/DeeplabV3/CamVid_SAM_DeeplabV3_KD_results/'
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=500)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=500))

train_dataloader = dict(batch_size=4, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

### AdamW for finetuning
####################
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
####################

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-7,
        power=0.9,
        begin=0,
        end=80000,
        by_epoch=False)
]

find_unused_parameters=True
