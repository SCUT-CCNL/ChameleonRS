# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='DSN',
    pretrained='pretrained/RN50.pt',
    class_names = 6,
    backbone=dict(
        type='MyResNet',
        style='pytorch'),

    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048 + 150],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='DAHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pam_channels=64,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.3)),
    discriminator=dict(
        type='Discriminator',
        in_channels=1024,
        channels=1,
        num_classes=1,
        dropout_ratio=0.1,
        align_corners=False,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.7)),
    recon_head=dict(
        type='ReconstructionHead',
        in_channels=2048,
        channels=3,
        num_classes=6,
        norm_cfg=norm_cfg),
    identity_head=dict(
        type='IdentityHead',
        in_channels=1,
        channels=1,
        num_classes=1,
        dropout_ratio=0.1,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    target_head=dict(
        type='IdentityHead',
        in_channels=1024,
        channels=1024,
        num_classes=6,
        dropout_ratio=0.1,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    sperate_head=dict(
        type='SperateHead',
        in_channels=1024,
        channels=256,
        num_classes=6),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

