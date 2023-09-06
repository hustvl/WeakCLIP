
_base_ = [
    '_base_/models/weakclip_vitb.py',
    '_base_/datasets/pascal_voc12_aug_pseudo_mask_mct.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_20k.py'
]

lr_config = dict(warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6)


optimizer = dict(paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.0),
                                        'text_encoder': dict(lr_mult=0.0),
                                        # 'norm': dict(decay_mult=0.)
                                        }))

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    test=dict(
        ann_dir='SegmentationClassAug',
        split='ImageSets/Segmentation/trainaug.txt',
    )
)
