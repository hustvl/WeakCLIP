_base_ = './pascal_voc12_aug.py'
# dataset settings
data = dict(
    train=dict(
        ann_dir=['SegmentationClassAugPseudoMaskMCT', 'SegmentationClassAugPseudoMaskMCT'],
))
