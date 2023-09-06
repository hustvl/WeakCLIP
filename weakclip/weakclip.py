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

from .untils import tokenize
from .dgcnutils import dgcn_cacul_knn_matrix_, dgcn_crf_operation, dgcn_generate_supervision, dgcn_softmax, \
    dgcn_constrain_loss, dgcn_cal_seeding_loss, dgcn_get_cues_from_seg_gt


@SEGMENTORS.register_module()
class WeakCLIP(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 class_names,
                 context_length,
                 context_feature='attention',
                 score_concat_index=3,
                 text_head=False,
                 text_encoder=None,
                 context_decoder=None,
                 neck=None,
                 tau=0.07,
                 auxiliary_head=None,
                 identity_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 token_embed_dim=512,
                 text_dim=1024, # no use
                 if_dgcn=False,
                 norm_eval=False,
                 fix_clip_things=False,
                 layer_norm_eval=False,
                 if_save_inference_logits=True,
                 if_save_inference_argmax=False,
                 with_flood_thresh=0.0,
                 if_fix_adapter=False,
                 if_decouple=False,
                 if_no_grad_vis=False,
                 if_dgcn_aux=False,
                 if_pyramid_queried_feature=False,
                 **args):
        super(WeakCLIP, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained


            assert text_encoder.get('pretrained') is None, \
                'both text encoder and segmentor set pretrained weight'

            if 'RN50' not in pretrained and 'RN101' not in pretrained and 'ViT-B' not in pretrained \
                    and 'ViT-L' not in pretrained:
                print('not CLIP pre-trained weight, using CLIP ViT-B-16')
                text_encoder.pretrained = 'pretrained/ViT-B-16.pt'
            else:
                text_encoder.pretrained = pretrained

        self.backbone = builder.build_backbone(backbone)
        self.text_encoder = builder.build_backbone(text_encoder)
        self.context_decoder = builder.build_backbone(context_decoder)
        self.context_length = context_length
        self.score_concat_index = score_concat_index

        assert context_feature in ['attention', 'backbone']
        self.context_feature = context_feature

        self.text_head = text_head
        self.tau = tau

        if neck is not None:
            self.neck = builder.build_neck(neck)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.with_identity_head = False
        self.identity_head = None
        self._init_identity_head(identity_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names]) # shape: torch.Size([21, 5])
        self.num_classes = len(self.texts)
        self.if_decouple = if_decouple


        context_length = self.text_encoder.context_length - self.context_length # 13 - 5

        # token embed dim is same as the width of text encoder
        token_embed_dim = self.text_encoder.embed_dim  # 512 for base; 768 for large

        self.contexts = nn.Parameter(torch.randn(1, context_length, token_embed_dim)) # shape: torch.Size([1, 8, 512])
        nn.init.trunc_normal_(self.contexts)

        # text dim is equal to the width of text encoder embed
        text_dim = self.text_encoder.embed_dim  # 512 for base; 768 for large

        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-1)

        if self.if_decouple:
            self.beta = nn.Parameter(torch.ones(text_dim) * 1e-1)

        self.if_dgcn = if_dgcn
        self.norm_eval = norm_eval
        self.fix_clip_things = fix_clip_things
        self.layer_norm_eval = layer_norm_eval
        self.if_save_inference_logits = if_save_inference_logits
        self.with_flood_thresh = with_flood_thresh
        self.if_save_inference_argmax = if_save_inference_argmax
        self.if_no_grad_vis = if_no_grad_vis
        self.if_dgcn_aux = if_dgcn_aux
        self.if_pyramid_queried_feature = if_pyramid_queried_feature

        assert self.with_decode_head

        self.train()

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)
    
    def _init_identity_head(self, identity_head):
        """Initialize ``auxiliary_head``"""
        if identity_head is not None:
            self.with_identity_head = True
            self.identity_head = builder.build_head(identity_head)

    def extract_feat(self, img):
        """Extract features from images."""
        if self.if_no_grad_vis:
            with torch.no_grad():
                x = self.backbone(img)
        else:
            x = self.backbone(img)
        return x

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


    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def _identity_head_forward_train(self, x, img_metas, gt_semantic_seg, imgs=None):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if imgs == None:
            loss_aux = self.identity_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
        else:
            loss_aux = self.identity_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg, imgs)
        losses.update(add_prefix(loss_aux, 'aux_identity'))
        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def after_extract_feat(self, x):
        # origin feature map form stage 1~4
        x_orig = list(x[0:4])
        # the last part of x is the [x_mean, x], x's shape is B * C(2048 for RN50) * W(512/32) * H(512/32)
        # global_fea is the x_mean, and the visual_embeddings is the feature point at stage4's output
        # global_feat's shape: torch.Size([4, 1024])
        # visual_embeddings' shape: torch.Size([4, 1024, 16, 16])
        global_feat, visual_embeddings = x[4]

        # B: 4 | C: 1024 | H: 16 | W: 16
        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            # concat the x_mean with the feature point and flatten these
            # visual_context's shape: torch.Size([4, 257, 1024])
            visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)  # B, N, C

        # contexts is the learnable context with exact class names
        # (B, K, C) | torch.Size([4, 21, 1024])
        text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        if self.if_decouple:
            text_diff, visual_diff = self.context_decoder(text_embeddings, visual_context)
            visual_context = visual_context + self.beta * visual_diff
            visual_embeddings = visual_context[:, 1:, :].permute(0, 2, 1).reshape(B, C, H, W)
        else:
            # update text_embeddings by visual_context!
            # (B, 1, C) | torch.Size([4, 21, 1024])
            text_diff = self.context_decoder(text_embeddings, visual_context) # ViTB: visual_context's shape: 8, 1025, 512 ; text_embeddings's shape: 8, 21, 512
        # (B, K, C) | torch.Size([4, 21, 1024])
        text_embeddings = text_embeddings + self.gamma * text_diff

        # compute score map and concat
        B, K, C = text_embeddings.shape
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        # score_map's shape: torch.Size([4, 21, 16, 16])
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text)

        if self.if_pyramid_queried_feature:
            for i in range(len(x_orig)):
                x_orig_i_shape = x_orig[i].shape
                # interpolate score_map
                score_map_i = torch.nn.functional.interpolate(score_map, size=(x_orig_i_shape[2], x_orig_i_shape[3]), mode='bilinear',
                                                align_corners=True)
                # print("size of x_org", str(i), ": ", x_orig[i].shape)
                x_orig[i] = torch.cat([x_orig[i], score_map_i], dim=1)
        else:
            x_orig[self.score_concat_index] = torch.cat([x_orig[self.score_concat_index], score_map], dim=1)
        return text_embeddings, x_orig, score_map

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

        # x is tuple,
        # x[0] shape: 8, 768, 128, 128
        # x[1] shape: 8, 768, 64, 64
        # x[2] shape: 8, 768, 32, 32
        # x[3] shape: 8, 768, 16, 16
        # x[4] is the normal output from the last ViT's block
        # x[4][0] shape: 8, 512
        # x[4][1] shape: 8, 512, 32, 32
        x = self.extract_feat(img)
        _x_orig = [x[i] for i in range(4)]
        # x_orig contains score_map in its x_org[3]
        text_embeddings, x_orig, score_map = self.after_extract_feat(x)

        if self.with_neck:
            x_orig = list(self.neck(x_orig))
            _x_orig = x_orig

        losses = dict()
        if self.text_head:
            x = [text_embeddings,] + x_orig
        else:
            x = x_orig

        # img's shape: torch.Size([4, 3, 512, 512])
        # score map's shape: torch.Size([4, 21, 16, 16])
        # gt_semantic_seg's shape: torch.Size([4, 1, 512, 512])

        # img_metas[0]: {'filename': '/data/zhulianghui/data/VOC2012/VOCdevkit/VOC2012/JPEGImages/2011_000192.jpg',
        # 'ori_filename': '2011_000192.jpg', 'ori_shape': (375, 500, 3), 'img_shape': (512, 512, 3),
        # 'pad_shape': (512, 512, 3), 'scale_factor': array([1.866    , 1.8666667, 1.866    , 1.8666667], dtype=float32),
        # 'flip': True, 'flip_direction': 'horizontal', 'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32),
        # 'std': array([58.395, 57.12 , 57.375], dtype=float32), 'to_rgb': True}}

        # gt_semantic_seg's shape: torch.Size([4, 1, 512, 512])
        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                          gt_semantic_seg, img if self.if_dgcn else None)
        losses.update(loss_decode)


        if self.with_identity_head:
            loss_identity = self._identity_head_forward_train(
                score_map/self.tau, img_metas, gt_semantic_seg, img if self.if_dgcn_aux else None)
            losses.update(loss_identity)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                _x_orig, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        # # change: for IN1K pretrained
        # loss_total = losses['decode.loss_boundary'] + losses['decode.loss_seeding'] + losses['aux_identity.loss_ce']
        # # loss_total = losses['decode.loss_boundary'] + losses['decode.loss_seeding']
        # if self.with_flood_thresh > loss_total:
        #     losses['decode.loss_boundary'] = losses['decode.loss_boundary'] * -1.0
        #     losses['decode.loss_seeding'] = losses['decode.loss_seeding'] * -1.0
        #     losses['aux_identity.loss_ce'] = losses['aux_identity.loss_ce'] * -1.0

        return losses

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        # x = self.extract_feat(img)

        x = self.extract_feat(img)
        _x_orig = [x[i] for i in range(4)]
        text_embeddings, x_orig, score_map = self.after_extract_feat(x)

        if self.with_neck:
            x_orig = list(self.neck(x_orig))

        if self.text_head:
            x = [text_embeddings,] + x_orig
        else:
            x = x_orig
        # print('text_embedding=', text_embeddings[0])
        out = self._decode_head_forward_test(x, img_metas)
        # print('cls_map=', out[0,:,40, 40])
        
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out


    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        
        if  torch.isnan(seg_logit).any():
            print('########### find NAN #############')

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        save_dir = 'prob_npy'
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, img_metas[0][0]['ori_filename'][:-4] + '.npy')

        if self.if_save_inference_logits and os.path.exists(save_path):
            print('########### find {} #############'.format(save_path))
            return list(np.load(save_path, allow_pickle=True).item()["pred"])
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        if self.if_save_inference_logits:
            seg_logit_np = seg_logit.squeeze(0).cpu().numpy()
            keys = np.unique(seg_pred)
            seg_logit_np = seg_logit_np[keys]

            np.save(save_path, {"prob": seg_logit_np, "keys": keys, "pred": seg_pred})
        if self.if_save_inference_argmax:
            save_path = './tmpargmax/' + img_metas[0][0]['ori_filename'][:-4] + '.npy'
            np.save(save_path, seg_pred)
        seg_pred = list(seg_pred)
        return seg_pred

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(WeakCLIP, self).train(mode)
        # todo: configure what it mean
        # self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, BatchNorm2d):
                    for param in m.parameters():
                        param.requires_grad = False
                    m.eval()
                if self.layer_norm_eval:
                    if isinstance(m, LayerNorm):
                        for param in m.parameters():
                            param.requires_grad = False
                        m.eval()

        if mode and self.fix_clip_things:
            pass

