from mmseg.core import add_prefix
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class FPNDGCNBaseline(EncoderDecoder):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 if_dgcn=False):
        super(FPNDGCNBaseline, self).__init__(backbone,
                                              decode_head,
                                              neck,
                                              auxiliary_head,
                                              train_cfg,
                                              test_cfg,
                                              pretrained,
                                              init_cfg)
        self.if_dgcn = if_dgcn

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg, imgs=None):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        # decide if dgcn
        if imgs == None:
            # non-dgcn
            loss_decode = self.decode_head.forward_train(x, img_metas,
                                                         gt_semantic_seg,
                                                         self.train_cfg)
        else:
            # dgcn
            loss_decode = self.decode_head.forward_train(x, img_metas,
                                                         gt_semantic_seg,
                                                         self.train_cfg,
                                                         imgs)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg, img if self.if_dgcn else None)

        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses
