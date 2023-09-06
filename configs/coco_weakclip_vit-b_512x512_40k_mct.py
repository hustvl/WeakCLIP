
_base_ = [
    '_base_/models/weakclip_vitb.py',
    '_base_/datasets/ms_coco_wsss_mct.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_40k.py'
]


model = dict(
        neck=dict(
            in_channels=[768+91, 768+91, 768+91, 768+91]),
            decode_head=dict(num_classes=91))


lr_config = dict(warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6)


optimizer = dict(paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.0),
                                        'text_encoder': dict(lr_mult=0.0),
                                        # 'norm': dict(decay_mult=0.)
                                        }))
log_config = dict(interval=100)
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=6,
    test=dict(
        split='voc_format/train.txt',
    )
)
