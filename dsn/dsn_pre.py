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
class DSNPre(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 recon_head=None,
                 **args):
        super(DSNPre, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = 'open-mmlab://resnet50_v1c'
            # backbone.pretrained = 'pretrained/backbone_RGB.pth'



        self.backbone = builder.build_backbone(backbone)
        self.backbone_t = builder.build_backbone(backbone)
        self.recon_head = builder.build_head(recon_head)



        # assert context_feature in ['attention', 'backbone']
        # self.context_feature = context_feature
        # self.tau = tau

        # if neck is None:
        #     self.neck = builder.build_neck(neck)

        self._init_decode_head(decode_head)
        self.auxiliary_head = builder.build_head(auxiliary_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg


        self.num_classes = 6

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


    def _init_t_identity_head(self, identity_head):
        """Initialize ``auxiliary_head``"""
        if identity_head is not None:
            self.with_identity_head = True
            self.t_identity_head = builder.build_head(identity_head)

    def _init_discriminator(self, discriminator):
        """Initialize ``auxiliary_head``"""
        self.discriminator = builder.build_head(discriminator)



    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
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
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode_t'))
        return losses

    def _seg_head_forward_train(self, x, img_metas, gt_semantic_seg):

        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.seg_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'auxseg_s'))
        return losses

    def _t_seg_head_forward_train(self, x, img_metas, gt_semantic_seg):

        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.seg_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'auxseg_t'))
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

    def _contra_head_forward_train(self, x, pred, text):

        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_con = self.cf_head.forward_train(x, pred, text)

        losses.update(add_prefix(loss_con, 'contra_s'))
        return losses

    def _t_contra_head_forward_train(self, x, pred, text):

        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_con_t = self.cf_head.forward_train(x, pred, text)

        losses.update(add_prefix(loss_con_t, 'contra_t'))
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
        loss_aux = self.auxiliary_head.forward_train(x, img_metas,gt_semantic_seg,self.train_cfg)
        losses.update(add_prefix(loss_aux, 't_aux'))
        return losses

    # def _aux_identity_head_forward_train(self, x, img_metas, gt_semantic_seg):
    #     """Run forward function and calculate loss for auxiliary head in
    #     training."""
    #     losses = dict()
    #     loss_aux = self.aux_identity_head.forward_train(
    #         x, img_metas, gt_semantic_seg, self.train_cfg)
    #     losses.update(add_prefix(loss_aux, 'aux_identity_global'))
    #
    #     return losses
    #
    # def _t_aux_identity_head_forward_train(self, x, img_metas, gt_semantic_seg):
    #     """Run forward function and calculate loss for auxiliary head in
    #     training."""
    #     losses = dict()
    #     loss_aux = self.aux_identity_head.forward_train(
    #         x, img_metas, gt_semantic_seg, self.train_cfg)
    #     losses.update(add_prefix(loss_aux, 't_aux_identity_global'))
    #
    #     return losses
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

    def _t_s_align(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()

        loss_align = self.target_head.forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(add_prefix(loss_align, 't2s_align'))

        return losses

    def _s_s_align(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()

        loss_align = self.identity_head.forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(add_prefix(loss_align, 's2s_align'))

        return losses

    def _s_decode(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()

        loss_align = self.target_head.forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(add_prefix(loss_align, 's_decode'))

        return losses

    # def _t_domain(self, x,img_metas, gt_semantic_seg):
    #     """Run forward function and calculate loss for auxiliary head in
    #     training."""
    #
    #     losses = dict()
    #     loss_aux = self.discriminator.forward_train(
    #         x, img_metas, gt_semantic_seg, self.train_cfg)
    #     losses.update(add_prefix(loss_aux, 't_domain_loss'))
    #
    #     return losses

    def _s_domain(self, tensor1,tensor2):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        loss_textalign = self.sperate_head.forward_train(tensor1,tensor2)
        # print(loss_textalign)
        losses.update(add_prefix(loss_textalign, 'source'))

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

    def get_text_embeddings(self, x, x_t):
        global_feat, visual_embeddings = x[4]
        global_feat_t, visual_embeddings_t = x_t[4]
        B, C, H, W = visual_embeddings.shape
        visual_context = (global_feat + global_feat_t).reshape(B, C, 1).permute(0, 2, 1)
        # (B, K, C)
        text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        text_embeddings = text_embeddings + self.gamma * text_diff
        return text_embeddings

    def weighted_add_text_features_to_feature_map(self,F, P, text_features):
        """
        根据预测概率图对每个像素的文本特征进行加权求和，然后与特征图相加。

        参数:
            F: 特征图，形状为 [B, C, H, W]
            P: 预测概率图，形状为 [B, K, H, W]
            text_features: 文本特征，形状为 [B, K, C]

        返回:
            结果特征图，形状为 [B, C, H, W]
        """
        B, C, H, W = F.shape
        B, K, C_text = text_features.shape
        P=torch.softmax(P,dim=1)
        P = P.permute(0, 2, 3, 1)  # [B, H, W, K]

        text_features = text_features.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, K, C]

        P = P.unsqueeze(-1)  # [B, H, W, K, 1]

        weighted_text_features = P * text_features  # [B, H, W, K, C]

        weighted_text_features = weighted_text_features.sum(dim=3)  # [B, H, W, C]

        weighted_text_features = weighted_text_features.permute(0, 3, 1, 2)  # [B, C, H, W]

        result = F + weighted_text_features

        return result

    def add_text_features_to_feature_map(self,P, text_features):

        B,K,C = text_features.shape
        B,K,H,W = P.shape
        P= torch.softmax(P,dim=1)
        # output = text_features[torch.arange(B).unsqueeze(1).unsqueeze(2), P]  # [B, H, W, C]
        # output = output.permute(0, 3, 1, 2)

        P = P.view(B, K, -1)

        text_features = text_features.transpose(1, 2)
        result = torch.matmul(text_features, P)
        output = result.view(B, C, H, W)

        return output

    def get_pseudo_labels(self,tensor1, tensor2, tensor3, threshold):
        """
        从三个张量中生成伪标签。

        参数:
            tensor1, tensor2, tensor3: 形状为 [B, K, H, W] 的张量，表示语义分割预测图。
            threshold: 用于判断最高概率和次高概率差值的阈值。

        返回:
            pseudo_labels: 形状为 [B, H, W] 的张量，表示每个像素点的伪标签。
        """
        # 确保输入张量的形状一致
        assert tensor1.shape == tensor2.shape == tensor3.shape, "输入张量的形状必须一致"
        tensor1 = F.softmax(tensor1, dim=1)
        tensor2 = F.softmax(tensor2, dim=1)
        tensor3 = F.softmax(tensor3, dim=1)
        # print(tensor1)

        B, K, H, W = tensor1.shape
        device = tensor1.device

        # 初始化伪标签张量，初始值为 -1（表示无效标签）
        pseudo_labels = torch.full((B, H, W), 255, dtype=torch.long, device=device)

        # 遍历每个张量，计算最高概率和次高概率的差值
        for tensor in [tensor1, tensor2, tensor3]:
            # 获取每个像素点的最高概率和对应的类别索引
            max_probs, max_classes = tensor.max(dim=1)

            # 获取每个像素点的次高概率
            tensor_clone = tensor.clone()
            tensor_clone.scatter_(1, max_classes.unsqueeze(1), -float('inf'))  # 将最高概率位置设为负无穷
            second_max_probs, _ = tensor_clone.max(dim=1)

            # 计算最高概率和次高概率的差值
            diff = max_probs - second_max_probs

            # 检查差值是否大于阈值
            valid_mask = diff > threshold

            # 对于满足条件的像素点，更新伪标签
            update_mask = valid_mask & (pseudo_labels == 255)  # 只更新尚未分配伪标签的像素点
            pseudo_labels[update_mask] = max_classes[update_mask]

        return pseudo_labels
    def getprototype(self,features,labels,num_classes):

        B, C, H, W = features.shape
        assert labels.shape == (B, H, W), "标签的形状必须与特征图的空间维度匹配"

        # 初始化类别中心特征
        class_centers = torch.zeros(B, num_classes, C, device=features.device)

        # 遍历每个类别
        for class_idx in range(num_classes):
            mask = (labels == class_idx).unsqueeze(1)  # 形状为 [B, 1, H, W]

            masked_features = features * mask  # 形状为 [B, C, H, W]

            class_sum = masked_features.sum(dim=(2, 3))  # 沿 H 和 W 维度求和，形状为 [B, C]

            class_count = mask.sum(dim=(2, 3))  # 沿 H 和 W 维度求和，形状为 [B, 1]

            class_count = torch.clamp(class_count, min=1.0)

            class_mean = class_sum / class_count  # 形状为 [B, C]

            class_centers[:, class_idx, :] = class_mean

        return class_centers
    def after_extract_feat(self, x,text_embeddings,mode = False):
        #x_orig = list(x[0:4])
        # print(domain_label.shape)

        visual_embeddings = x
        #设置三个尺度的池化结果，分别是1、3、5
        # f1x1 = self.conv1x1(visual_embeddings)
        # f3x3 = self.conv3x3(visual_embeddings)
        # f5x5 = self.conv5x5(visual_embeddings)

        #得到三个尺度的匹配结果
        # visual1 = F.normalize(f1x1, dim=1, p=2)
        # visual3 = F.normalize(f3x3, dim=1, p=2)
        # visual5 = F.normalize(f5x5, dim=1, p=2)
        visual = F.normalize(x, dim=1, p=2)

        text = F.normalize(text_embeddings, dim=2, p=2)
        # score_map1 = torch.einsum('bchw,bkc->bkhw', visual1, text)
        # score_map2 = torch.einsum('bchw,bkc->bkhw', visual3, text)
        # score_map3 = torch.einsum('bchw,bkc->bkhw', visual5, text)
        score_map = torch.einsum('bchw,bkc->bkhw', visual, text)

        # pse = self.get_pseudo_labels(score_map1/self.tau, score_map2/self.tau, score_map3/self.tau,0.4)
        # score_map = (score_map1+score_map2+score_map3)/3
        pse = self.generatepse(F.softmax(score_map/self.tau,dim=1),0.5)
        # x_orig[self.score_concat_index] = torch.cat([visual_embeddings, score_map], dim=1)
        # print(score_map.shape)
        # outfeature = 0.75*visual_embeddings + 0.25*self.add_text_features_to_feature_map(score_map,text_embeddings)
        # outfeature = visual_embeddings
        return [score_map/self.tau],pse.unsqueeze(1)

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
    def forward_train(self, img, img_metas,B_img, gt_semantic_seg):
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
        if not hasattr(self, 'iteration'):
            self.iteration = 0
        curr_iter = self.iteration
        #获取源域特征
        x = self.extract_feat(img)
        # x_t=self.extract_feat(B_img)

        #获取目标域特征
        #进行像素-文本匹配

        losses = dict()
        loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        losses.update(loss_decode)
        loss_aux = self._auxiliary_head_forward_train(x, img_metas, gt_semantic_seg)
        losses.update(loss_aux)
        # if curr_iter>=6000:
        #     t_logits = self._decode_head_forward_test(x_t, img_metas)
        #     t_pse_decode = self.generatepse(F.softmax(t_logits, dim=1), 0.5).unsqueeze(1)
        #     loss_aux_t = self._t_auxiliary_head_forward_train(x_t, img_metas, t_pse_decode)
        #     losses.update(loss_aux_t)

        if hasattr(self, 'iteration'):
            self.iteration += 1

        return losses

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x,img_metas)

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
