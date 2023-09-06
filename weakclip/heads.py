import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule
from mmseg.ops import Upsample, resize

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.decode_heads.fpn_head import FPNHead

from .dgcnutils import dgcn_cacul_knn_matrix_, dgcn_crf_operation, dgcn_generate_supervision, dgcn_softmax, \
    dgcn_constrain_loss, dgcn_cal_seeding_loss, dgcn_get_cues_from_seg_gt, dgcn_get_cues_from_seg_gt_tensor, generate_supervision_multi_threads, generate_supervision_by_so
# import maxflow_dgcn_cpp



@HEADS.register_module()
class IdentityHead(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.
    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.
    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, **kwargs):
        super(IdentityHead, self).__init__(
            input_transform=None, **kwargs)
        self.conv_seg = None

    def forward(self, inputs):
        return inputs

@HEADS.register_module()
class IdentityHeadDGCN(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.
    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.
    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, **kwargs):
        super(IdentityHeadDGCN, self).__init__(
            input_transform=None, **kwargs)
        self.conv_seg = None
        self.loss_decode_dict = kwargs.get('loss_decode', None)

    def forward(self, inputs):
        return inputs

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, imgs=None):
        device = inputs[0].device
        NUM_CLASSES = 21

        # generate seg_logits and seg_features
        # seg_logits' shape: torch.Size([B, 21, 128, 128])
        # seg_features' shape: torch.Size([B, 256, 128, 128])
        seg_logits = self.forward(inputs)

        # generate cues, dont set 'align_corners=False'
        gt_semantic_seg_scaled = torch.nn.functional.interpolate(gt_semantic_seg.float(), size=seg_logits.shape[2:],
                                                                 mode='nearest')
        # cues' shape: torch.Size([B, 21, 128, 128])
        cues = dgcn_get_cues_from_seg_gt_tensor(gt_semantic_seg_scaled, seg_logits.shape)
        labels = (cues.sum(dim=3).sum(dim=2) > 0).int()
        # calcute crf loss
        min_prob = torch.tensor(0.0000001, device=device)

        # follow identity head
        probs = dgcn_softmax(seg_logits, min_prob)
        # probs = seg_logits

        batchsize = probs.shape[0]
        probs_cpu = probs.cpu().detach().numpy()
        crf = dgcn_crf_operation(imgs.cpu().detach().numpy(), probs_cpu, img_metas)

        aux_crf_constrain_loss = dgcn_constrain_loss(probs, crf)

        aux_seeding_loss = dgcn_cal_seeding_loss(probs, cues)
        # print("seeding loss done!")

        # import pdb; pdb.set_trace()

        # loss_weight
        aux_seeding_loss = aux_seeding_loss * self.loss_decode_dict['loss_weight']
        aux_crf_constrain_loss = aux_crf_constrain_loss * self.loss_decode_dict['loss_weight']

        losses = {'aux_loss_boundary': aux_crf_constrain_loss, 'aux_loss_seeding': aux_seeding_loss}
        return losses


@HEADS.register_module()
class FPNHeadWithFeatures(FPNHead):

    def __init__(self, feature_strides, **kwargs):
        super(FPNHeadWithFeatures, self).__init__(feature_strides, **kwargs)

    def forward_with_features(self, inputs):
        x = self._transform_inputs(inputs)

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        output_seg = self.cls_seg(output)

        return output_seg, output

    def forward_train_with_features(self, inputs, img_metas, test_cfg):
        return self.forward_with_features(inputs)

    def forward_with_crf(self, inputs, img_metas, test_cfg):

        out = self.forward(inputs)

        min_prob = torch.tensor(0.0000001, device=out.device)
        probs = dgcn_softmax(out, min_prob)

        probs_cpu = probs.cpu().detach().numpy()
        crf = dgcn_crf_operation(imgs.cpu().detach().numpy(), probs_cpu, img_metas)


    def forward_test_with_crf(self, inputs, img_metas, test_cfg):
        pass


@HEADS.register_module()
class FPNHeadDGCN(FPNHeadWithFeatures):

    def __init__(self, feature_strides, if_dgcn=False, feature_size=None, if_dgcn_lite=False, if_gelu=False,
                 if_fewer_norm=False, if_fewer_act=False, dgcn_method='multi-threads', **kwargs):
        super(FPNHeadDGCN, self).__init__(feature_strides, **kwargs)

        self.if_dgcn = if_dgcn
        self.feature_size = feature_size
        self.if_dgcn_lite = if_dgcn_lite
        self.dgcn_method = dgcn_method

        for head in self.scale_heads._modules.values():
            # head._modules['0'].bn = nn.LayerNorm(256)
            # head._modules['0'].activate = nn.GELU()
            for module in head._modules.values():
                if type(module) == ConvModule:
                    if if_gelu:
                        module.activate = nn.GELU()

        if if_fewer_norm and if_fewer_norm:
            self.scale_heads._modules['2']._modules['0'].act_cfg = None
            self.scale_heads._modules['2']._modules['0'].activate = None
            self.scale_heads._modules['2']._modules['0'].with_activation = False

            self.scale_heads._modules['2']._modules['2'].norm_cfg = None
            self.scale_heads._modules['2']._modules['2'].bn = None
            self.scale_heads._modules['2']._modules['2'].with_norm = False

            self.scale_heads._modules['3']._modules['0'].act_cfg = None
            self.scale_heads._modules['3']._modules['0'].activate = None
            self.scale_heads._modules['3']._modules['0'].with_activation = False

            self.scale_heads._modules['3']._modules['2'].norm_cfg = None
            self.scale_heads._modules['3']._modules['2'].bn = None
            self.scale_heads._modules['3']._modules['2'].with_norm = False

            self.scale_heads._modules['3']._modules['2'].act_cfg = None
            self.scale_heads._modules['3']._modules['2'].activate = None
            self.scale_heads._modules['3']._modules['2'].with_activation = False

            self.scale_heads._modules['3']._modules['4'].norm_cfg = None
            self.scale_heads._modules['3']._modules['4'].bn = None
            self.scale_heads._modules['3']._modules['4'].with_norm = False
        # print("okk")
        # print("okk")
        # print("okk")


    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, imgs=None):
        if not self.if_dgcn:
            seg_logits = self.forward(inputs)
            losses = self.losses(seg_logits, gt_semantic_seg)
            return losses
        else:
            device = inputs[0].device
            NUM_CLASSES = 21

            # generate seg_logits and seg_features
            # seg_logits' shape: torch.Size([B, 21, 128, 128])
            # seg_features' shape: torch.Size([B, 256, 128, 128])
            seg_logits, seg_features = self.forward_train_with_features(inputs, img_metas, train_cfg)

            if self.feature_size != False:
                seg_logits = torch.nn.functional.interpolate(seg_logits.float(), size=self.feature_size,
                                                             mode='bilinear',
                                                             align_corners=(self.feature_size[0] % 2 == 1) and
                                                                           (self.feature_size[1] % 2 == 1))
                if isinstance(seg_features, list):
                    seg_feature_list = [torch.nn.functional.interpolate(seg_feature.float(),
                                                     size=self.feature_size,
                                                     mode='bilinear',
                                                     align_corners=(self.feature_size[0] % 2 == 1) and
                                                                   (self.feature_size[1] % 2 == 1))
                     for seg_feature in seg_features]
                    seg_features = torch.cat(seg_feature_list, 1)
                else:
                    seg_features = torch.nn.functional.interpolate(seg_features.float(), size=self.feature_size,
                                                                   mode='bilinear',
                                                                   align_corners=(self.feature_size[0] % 2 == 1) and
                                                                                 (self.feature_size[1] % 2 == 1))

            # generate cues, dont set 'align_corners=False'
            gt_semantic_seg_scaled = torch.nn.functional.interpolate(gt_semantic_seg.float(), size=seg_logits.shape[2:],
                                                                     mode='nearest')
            # cues' shape: torch.Size([B, 21, 128, 128])
            cues = dgcn_get_cues_from_seg_gt_tensor(gt_semantic_seg_scaled, seg_logits.shape)
            labels = (cues.sum(dim=3).sum(dim=2) > 0).int()
            knn_matrix = dgcn_cacul_knn_matrix_(seg_features)

            # calcute crf loss
            min_prob = torch.tensor(0.0000001, device=device)
            probs = dgcn_softmax(seg_logits, min_prob)

            batchsize = probs.shape[0]
            probs_cpu = probs.cpu().detach().numpy()
            crf = dgcn_crf_operation(imgs.cpu().detach().numpy(), probs_cpu, img_metas)
            crf_constrain_loss = dgcn_constrain_loss(probs, crf)

            # calculate peudo mask and seeding loss
            # supervision = generate_supervision_by_so(seg_features.cpu().detach().numpy(), labels, cues, None,
            #                                         probs.cpu().detach().numpy(), knn_matrix.cpu().detach().numpy())
            if self.dgcn_method == 'cython':
                supervision = dgcn_generate_supervision(seg_features.cpu().detach().numpy(), labels, cues.cpu(), None,
                                                        probs.cpu().detach().numpy(), knn_matrix.cpu().detach().numpy())
            # supervision = generate_supervision_multi_threads(seg_features.cpu().detach().numpy(), labels, cues, None,
            #                                          probs.cpu().detach().numpy(), knn_matrix.cpu().detach().numpy())

            if self.dgcn_method == 'multi-threads':
                markers_new_batch = torch.ones((batchsize, 41, 41), dtype=torch.long, device=cues.device) * NUM_CLASSES
                pos = torch.where(cues == 1)
                markers_new_batch[pos[0], pos[2], pos[3]] = pos[1]
                supervision = torch.from_numpy(
                    maxflow_dgcn_cpp.generate_supervision_multi_threads((markers_new_batch).float().cpu().numpy(),
                                                                        labels.cpu().numpy(),
                                                                        cues.cpu(),
                                                                        probs_cpu, knn_matrix.cpu().detach().numpy()))
            if self.if_dgcn_lite:
                seeding_loss = dgcn_cal_seeding_loss(probs, cues)
            else:
                seeding_loss = dgcn_cal_seeding_loss(probs, supervision)
            # print("seeding loss done!")

            losses = {'loss_boundary': crf_constrain_loss, 'loss_seeding': seeding_loss}
            return losses


@HEADS.register_module()
class SegHeadDGCN(FPNHeadDGCN):

    def __init__(self, feature_strides, if_dgcn=True, feature_size=None, seg_head_dilation=1, **kwargs):
        super(SegHeadDGCN, self).__init__(feature_strides, if_dgcn, feature_size, **kwargs)

        self.scale_heads = None
        if seg_head_dilation != 1:
            self.conv_seg = nn.Conv2d(kwargs['channels'], 21, kernel_size=3, stride=1, padding=seg_head_dilation, dilation=seg_head_dilation, bias=True)


    def forward_with_features(self, inputs):
        x = self._transform_inputs(inputs)

        # no scale head in origin version dgcn, feature-map after stage4 will be sent to segHead
        # output = self.scale_heads[0](x[0])
        # for i in range(1, len(self.feature_strides)):
        #     # non inplace
        #     output = output + resize(
        #         self.scale_heads[i](x[i]),
        #         size=output.shape[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)

        output_seg = self.cls_seg(x[-1])

        return output_seg, x

    def forward_test(self, inputs, img_metas, test_cfg):
        output_seg, _ = self.forward_with_features(inputs)

        return output_seg


