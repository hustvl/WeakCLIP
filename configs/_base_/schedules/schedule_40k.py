# optimizer
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.00003)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='CosineAnnealing',  min_lr=1e-6, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=500)
evaluation = dict(interval=500, metric='mIoU')
