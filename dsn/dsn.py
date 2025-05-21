import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor
import torch.autograd as autograd
from mmcv.cnn import Scale
import numpy as np
from mmseg.models.utils import SelfAttentionBlock as _SelfAttentionBlock

class GradientReversalLayer(autograd.Function):
    @staticmethod
    def forward(ctx, input, lambda_):
        # 前向传播直接返回输入
        ctx.lambda_ = lambda_
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时反转梯度
        lambda_ = ctx.lambda_
        grad_input = -lambda_ * grad_output
        return grad_input, None  # 第二个返回值是 None，因为 lambda_ 是常数，不需要求导

# 定义 GRL 层的模块化封装
class GRLLayer(nn.Module):
    def __init__(self):
        super(GRLLayer, self).__init__()
        #self.lambda_ = lambda_

    def forward(self, x,lambda_):
        return GradientReversalLayer.apply(x, lambda_)
@SEGMENTORS.register_module()
class DSN(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 backbone_t,
                 sperate_head,
                 decode_head,
                 recon_head,
                 cf_head,
                 class_names,
                 neck=None,
                 target_head=None,
                 auxiliary_head=None,
                 identity_head=None,
                 discriminator=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 **args):
        super(DSN, self).__init__(init_cfg)
        # if pretrained is not None:
        # backbone_t.pretrained = 'pretrained/backbone.pth'
        # backbone.pretrained = 'pretrained/backbone.pth'

        self.backbone = builder.build_backbone(backbone)
        self.backbone_t = builder.build_backbone(backbone_t)
        self.sperate_head = builder.build_head(sperate_head)
        self.recon_head = builder.build_head(recon_head)
        self.grl = GRLLayer()
        self.cf_head = builder.build_head(cf_head)

        self._init_decode_head(decode_head)
        self.auxiliary_head = builder.build_head(auxiliary_head)

        self.with_identity_head = False
        self.identity_head = None
        self._init_identity_head(identity_head,target_head)
        # self.decode_head.init_weights('pretrained/decode_RGB.pth')
        # self.auxiliary_head.init_weights('pretrained/auxiliary_head_RGB.pth')
        #self._init_t_identity_head(identity_head)

        self.discriminator = builder.build_head(discriminator)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg


        self.num_classes = class_names
        self.dann_iter = 0
        self.mfam_uter = 0
        self.cont_iter =3000
        self.iteration = 0

        assert self.with_decode_head


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

    def _init_identity_head(self, identity_head,target_head):
        """Initialize ``auxiliary_head``"""
        if identity_head is not None:
            self.with_identity_head = True
            self.identity_head = builder.build_head(identity_head)
            self.target_head = builder.build_head(target_head)


    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        return x
    def extract_feat_t(self, img):
        """Extract features from images."""
        x = self.backbone_t(img)
        return x

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):

        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _t_decode_head_forward_train(self, x, img_metas, gt_semantic_seg):

        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.target_head.forward_train([x], img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode_t'))
        return losses



    def _discriminator_forward_train(self, x, img_metas, gt_semantic_seg):

        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.discriminator.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)
        losses.update(add_prefix(loss_decode, 'diff_s'))
        return losses

    def _t_discriminator_forward_train(self, x, img_metas, gt_semantic_seg):

        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.discriminator.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'diff_t'))
        return losses

    def _recon_head_forward_train(self, x, img):

        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()

        loss_recon = self.recon_head.forward_train(x, img)
        losses.update(add_prefix(loss_recon, 'recon'))
        return losses

    def _t_recon_head_forward_train(self, x, img):

        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()

        loss_recon_t = self.recon_head.forward_train(x, img)

        losses.update(add_prefix(loss_recon_t, 'recon_t'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits


    def _identity_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        loss_aux = self.identity_head.forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(add_prefix(loss_aux, 'aux_identity'))

        return losses

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        loss_aux = self.auxiliary_head.forward_train(x, img_metas,gt_semantic_seg,self.train_cfg)
        losses.update(add_prefix(loss_aux, 'aux'))
        return losses

    def _t_auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        loss_aux = self.identity_head.forward_train([x], img_metas,gt_semantic_seg,self.train_cfg)
        losses.update(add_prefix(loss_aux, 't_aux'))
        return losses


    def _s_alignment(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()

        loss_align = self.target_head.forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(add_prefix(loss_align, 's2t_align'))

        return losses
    def _t_alignment(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()

        loss_align = self.target_head.forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(add_prefix(loss_align, 't_align'))

        return losses


    def _t_domain(self, tensor1, tensor2):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        loss_textalign = self.sperate_head.forward_train(tensor1, tensor2)
        # print(loss_textalign)
        losses.update(add_prefix(loss_textalign, 'target'))

        return losses

    def mse(self, tensor1, tensor2,epsilon=1e-7):
        assert tensor1.shape == tensor2.shape, "The shapes of the tensors must match."

        # 获取张量的维度
        B, C, H, W = tensor1.shape

        # 将张量展平为 [B, C, H*W]
        tensor1_flat = tensor1.view(B, C, -1)  # 变为 [B, C, H*W]
        tensor2_flat = tensor2.view(B, C, -1)  # 变为 [B, C, H*W]



        # 计算每个像素位置的余弦相似度
        dot_product = torch.einsum('bci,bdi->bc', tensor1_flat, tensor2_flat)  # 计算点积
        norm_tensor1 = torch.norm(tensor1_flat, dim=2) + epsilon  # 计算L2范数
        norm_tensor2 = torch.norm(tensor2_flat, dim=2) + epsilon  # 计算L2范数

        # 计算余弦相似度
        cosine_similarity = dot_product / (norm_tensor1 * norm_tensor2)  # 形状为 [B, C]

        # 计算损失，目标是使余弦相似度尽可能小
        loss = torch.mean(cosine_similarity ** 2)  # 使用余弦相似度的平方作为损失
        print(loss)
        return loss


    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set requires_grad for all the networks.

        Args:
            nets (nn.Module | list[nn.Module]): A list of networks or a single
                network.
            requires_grad (bool): Whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def cosine_similarity(self,features):

        norms = torch.norm(features, p=2, dim=1, keepdim=True)

        normalized_features = features / norms

        similarity_matrix = torch.mm(normalized_features, normalized_features.t())

        return similarity_matrix
    def forward_train(self, img, img_metas, B_img, gt_semantic_seg):
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
        # print(img.shape)
        # print("###################################")
        # print(B_img.shape)
        if not hasattr(self, 'iteration'):
            self.iteration = 0
        curr_iter = self.iteration
        #获取源域特征
        if curr_iter==0:
            self.backbone_t.load_state_dict(self.backbone.state_dict())
        x = self.extract_feat(img)
        visual_embeddings= x[3]

        #获取目标域特征
        x_t = self.extract_feat(B_img)
        x_t_pri = self.extract_feat_t(B_img)
        B,C,H,W = visual_embeddings.shape
        #进行像素-文本匹配



        losses = dict()
        '''
        进行源域的全监督训练
        1. 第四层进行双层注意力机制之后进行解码，进行全监督训练，主解码器的输入通道数是1024，输出是numclass
        2. 第三层输入简单的全卷积网络，进行全监督训练，辅助解码器的输入通道数是1024，输出是numclass
        '''

        loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        losses.update(loss_decode)


        t_logits = self._decode_head_forward_test(x_t, img_metas)


        '''
        进行传统的领域对抗训练
        '''
        if curr_iter>=self.dann_iter:
            p = float((curr_iter-self.dann_iter)/ (10000-self.dann_iter))
            p = 2. / (1. + np.exp(-10 * p)) - 1
            reversd_x_t = self.grl(x_t[-1], p)
            reversd_x = self.grl(x[-1], p)
            i_H,i_W = img.shape[-2:]
            source_label = torch.ones(B,1,i_H,i_W,dtype=torch.long).to(visual_embeddings.device)
            target_label = torch.zeros(B,1,i_H,i_W,dtype=torch.long).to(visual_embeddings.device)
            # traditional
            loss_s_domain = self._discriminator_forward_train(reversd_x, img_metas, source_label)
            losses.update(loss_s_domain)
            loss_t_domain = self._t_discriminator_forward_train(reversd_x_t, img_metas, target_label)
            losses.update(loss_t_domain)

            loss_aux = self._auxiliary_head_forward_train(x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)



            loss_recon = self._t_recon_head_forward_train(0.5*x_t[-1]+0.5*x_t_pri[-1], B_img)
            losses.update(loss_recon)

            center_feature = self.cf_head.forward(x_t[-1], self.auxiliary_head.forward(x_t))
            center_feature_pri = self.cf_head.forward(x_t_pri[-1], self.auxiliary_head.forward(x_t))
            loss_seperate = self._t_domain(center_feature,center_feature_pri)
            losses.update(loss_seperate)



        if curr_iter>=self.mfam_uter:
            t_pse_decode = self.generatepse(F.softmax(t_logits, dim=1), 0.5).unsqueeze(1)
            loss_aux_t = self._t_auxiliary_head_forward_train(self.auxiliary_head.forward(x_t), img_metas, t_pse_decode)
            losses.update(loss_aux_t)

        if curr_iter>=self.cont_iter:
            '''
            目标域的特征和源域原型特征进行对齐
            1. 先得到源域的原型特征
            2. 目标域的特征和源域的原型进行相似度度量
            3. 使用CE损失进行约束
            '''
            t_pse_decode = self.generatepse(F.softmax(t_logits, dim=1), 0.5).unsqueeze(1)
            # t_pse_decode = F.softmax(t_logits, dim=1)

            # self.set_requires_grad(self.backbone, False)
            # self.set_requires_grad(self.backbone_t,False)
            x_t_pri[-1] = 0.5*x_t[-1]+0.5*x_t_pri[-1]
            loss_decode_t = self._t_decode_head_forward_train(self._decode_head_forward_test(x_t_pri, img_metas), img_metas, t_pse_decode)
            losses.update(loss_decode_t)

        if hasattr(self, 'iteration'):
            self.iteration += 1
        return losses

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        # out = self._decode_head_forward_test(x, img_metas)
        #print(self.iteration)
        
        if self.iteration<=self.cont_iter and self.iteration!=0:
            out = self._decode_head_forward_test(x, img_metas)
        else:
            x_pri = self.extract_feat_t(img)
            x_pri[-1]=0.5*x[-1]+0.5*x_pri[-1]
            out = self._decode_head_forward_test(x_pri, img_metas)
            print("###################")
        # out = score_map
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

        if torch.isnan(seg_logit).any():
            print('########### find NAN #############')

        return seg_logit
    def generatepse(self,tensor,psehold):
        # print(tensor.shape)
        # print(tensor)
        values, indices = torch.topk(tensor, k=2, dim=1)  # Get the top-2 values along the channel dimension
        max_values = values[:, 0]
        second_max_values = values[:, 1]

        diff = max_values - second_max_values  # Compute the difference between maximum and second maximum values

        mask = diff > psehold  # Create a mask where the difference is greater than 0.5
        # print(mask)

        # pseudo_labels = indices[:, 0].unsqueeze(1).unsqueeze(2)  # Create pseudo labels with the index of maximum value
        # pseudo_labels = pseudo_labels.expand(*tensor.shape[:-1])  # Adjust dimensions to match 'tensor'
        pseudo_labels = torch.argmax(tensor, 1)
        # print(pseudo_labels)
        pseudo_labels[mask == 0] = 255
        # print(pseudo_labels)

        # print(pseudo_labels)
        # print(pseudo_labels.shape)
        zero_count = (pseudo_labels == 255).sum(dim=[1, 2], keepdim=True)
        # print(zero_count)
        # print(pseudo_labels.shape)

        return pseudo_labels

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
                output = output.flip(dims=(3,))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2,))

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
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)
        return seg_pred
