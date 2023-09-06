import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import SyncBatchNorm, BatchNorm2d, GroupNorm, LayerNorm
import numpy as np

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder

from .untils import tokenize
from .dgcnutils import dgcn_cacul_knn_matrix_, dgcn_crf_operation, dgcn_generate_supervision, dgcn_softmax, \
    dgcn_constrain_loss, dgcn_cal_seeding_loss, dgcn_get_cues_from_seg_gt


@SEGMENTORS.register_module()
class MyEncoderDecoder(EncoderDecoder):

    def __init__(self, backbone, decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 if_save_inference_logits=False,
                 npy_save_location='npy'):
        super(EncoderDecoderInferenceNpy, self).__init__(backbone,
                                                         decode_head,
                                                         neck=neck,
                                                         auxiliary_head=auxiliary_head,
                                                         train_cfg=train_cfg,
                                                         test_cfg=test_cfg,
                                                         pretrained=pretrained,
                                                         init_cfg=init_cfg)
        self.npy_save_location = npy_save_location
        self.if_save_inference_logits = if_save_inference_logits

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        if self.if_save_inference_logits:
            seg_logit_np = seg_logit.squeeze(0).cpu().numpy()
            save_path = './' + self.npy_save_location + '/' + img_metas[0][0]['ori_filename'][:-4] + '.npy'
            np.save(save_path, seg_logit_np)
        seg_pred = seg_logit.argmax(dim=1)
        if self.if_save_inference_argmax:
            save_path = './npyargmax/' + img_metas[0][0]['ori_filename'][:-4] + '.npy'
            np.save(save_path, seg_pred)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
