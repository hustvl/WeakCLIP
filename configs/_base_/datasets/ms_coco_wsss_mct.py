_base_ = './ms_coco_wsss.py'

data = dict(
    train=dict(
        img_dir='images',
        ann_dir='voc_format/cocoPGTMCT',
        )
)
