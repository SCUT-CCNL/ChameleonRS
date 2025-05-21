# dataset settings
dataset_type = 'PVDataset_forAdap'
data_root = 'data'

IMG_MEAN = [ v*255 for v in [0.48145466, 0.4578275, 0.40821073]]
IMG_VAR = [ v*255 for v in [0.26862954, 0.26130258, 0.27577711]]
# IMG_MEAN = [0.48145466, 0.4578275, 0.40821073]
# IMG_VAR = [0.26862954, 0.26130258, 0.27577711]

img_norm_cfg = dict(mean=IMG_MEAN, std=IMG_VAR, to_rgb=True)
crop_size = (480,480)
train_pipeline = [
    dict(type='LoadImageFromFile_forAdap'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'B_img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512,512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        #img_dir='Potsdam/RGB/img_dir/train',
        #ann_dir='Potsdam/RGB/ann_dir/train',
        #split='Potsdam/RGB/train.txt',
        img_dir='Potsdam/IRRG/img_dir/train',
        ann_dir='Potsdam/IRRG/ann_dir/train',
        split='Potsdam/IRRG/train.txt',
        B_img_dir = 'Vaihingen/img_dir/train',
        B_split = 'Vaihingen/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='Vaihingen/img_dir/test',
        ann_dir='Vaihingen/ann_dir/test',
        split='Vaihingen/test.txt',
        # img_dir='Potsdam_RGB/img_dir/train',
        # ann_dir='Potsdam_RGB/ann_dir/train',
        # split='Potsdam_RGB/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='Vaihingen/img_dir/test',
        ann_dir='Vaihingen/ann_dir/test',
        split='Vaihingen/test.txt',
        pipeline=test_pipeline))
