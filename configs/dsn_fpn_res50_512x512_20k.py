_base_ = [
    '_base_/models/dsn_r50.py', '_base_/datasets/PVAda.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_20k.py'
]

model = dict(
    type='DSN',
    pretrained='pretrained/RN50.pt',
    # pretrained = None,
    backbone=dict(
        type='MyResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    backbone_t=dict(
        type='MyResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),

    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),

    cf_head=dict(
        type='CenterHead',
        in_channels=[48, 96, 192, 384],
        channels=2048,
        input_transform='resize_concat',
        in_index=(0, 1, 2, 3),
        dropout_ratio=-1,
        num_classes=6,
        align_corners=False),
    decode_head=dict(
        type='DAHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pam_channels=64,
        dropout_ratio=0.1,
        num_classes=6,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),

)

lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001,
                 paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
                                                'backbone_t': dict(lr_mult=0.1),
                                                 # 'decode_head': dict(lr_mult=0.1),
                                                 # 'text_encoder': dict(lr_mult=0.0),
                                                 'norm': dict(decay_mult=0.)}))

data = dict(samples_per_gpu=1)

