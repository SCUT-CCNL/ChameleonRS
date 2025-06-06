# optimizer
optimizer = dict(type='SGD', lr=0.01, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000)
#checkpoint_config=dict(by_epoch=False, interval=500,max_keep_ckpts=3,save_best='mIoU')
evaluation = dict(interval=2000, metric='mIoU')
