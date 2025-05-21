import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule
from mmseg.ops import Upsample, resize
import torch.nn.functional as F
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from .loss import MSELoss,ContrastiveLoss
from mmseg.models.utils import SelfAttentionBlock as _SelfAttentionBlock
from mmseg.models.decode_heads.cascade_decode_head import BaseCascadeDecodeHead
from mmseg.core import add_prefix
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
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
        self.ignore_index = 255

    def forward(self, inputs):
        return inputs
    def losses(self, seg_logit, seg_label):
        """Compute ``pam_cam``, ``pam``, ``cam`` loss."""
        loss = dict()
        if len(seg_logit) == 1:
            loss.update(
                add_prefix(
                    super(IdentityHead, self).losses(seg_logit[0], seg_label),
                    'ce'))
        else:
            score1,score2, score3 = seg_logit

            loss.update(
                add_prefix(
                    super(IdentityHead, self).losses(score1, seg_label),
                    'score1x1'))
            loss.update(add_prefix(
                    super(IdentityHead, self).losses(score2, seg_label),
                    'score3x3'))
            loss.update(add_prefix(
                super(IdentityHead, self).losses(score3, seg_label),
                'score5x5'))
        return loss

@HEADS.register_module()
class TargetHead(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.
    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.
    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, **kwargs):
        super(TargetHead, self).__init__(
            input_transform=None, **kwargs)
        # self.conv_seg = None
        self.ignore_index = 255
        self.conv_seg = nn.Sequential(
            nn.Conv2d(self.channels, self.channels//4, kernel_size=1),
            nn.SyncBatchNorm(self.channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels//4, self.num_classes, kernel_size=1)
        )

    def forward(self, inputs):
        return self.cls_seg(inputs)
@HEADS.register_module()
class SperateHead(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.
    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.
    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, **kwargs):
        super(SperateHead, self).__init__(
            input_transform=None, **kwargs)
        self.cosine_loss = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, inputs1,inputs2):
        return inputs1,inputs2
    def forward_train(self, inputs1, inputs2):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = self.losses(inputs1, inputs2)
        return losses

    def losses(self, tensor1, tensor2,epsilon = 1e-7):
        """Compute segmentation loss."""
        loss = dict()
        # 计算余弦相似度
        tensor1 = tensor1 / (torch.norm(tensor1, p=2, dim=1, keepdim=True) + 1e-6)
        tensor2 = tensor2 / (torch.norm(tensor2, p=2, dim=1, keepdim=True) + 1e-6)
        similarity = self.cosine_loss(tensor1, tensor2)

        # 计算损失，目标是使余弦相似度尽可能小
        loss_sperate = 0.2 * torch.mean(similarity**2)

        loss['loss_seperate'] = loss_sperate
        return loss
def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))
@HEADS.register_module()
class ReconstructionHead(BaseDecodeHead):
    def __init__(self,in_channels, **kwargs):
        super(ReconstructionHead, self).__init__(in_channels,**kwargs)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4,kernel_size=1),
            nn.SyncBatchNorm(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        # self.conv1 = ConvModule(
        #     in_channels,
        #     in_channels//4,
        #     1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 8, kernel_size=3, padding=1),
            nn.SyncBatchNorm(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        # self.conv2 = ConvModule(
        #     in_channels//4,
        #     in_channels // 8,
        #     3,
        #     padding=1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg)
        
        self.conv3 = nn.Conv2d(in_channels // 8, 3,kernel_size=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def forward_train(self, inputs, img):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, img)
        return losses

    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        # if self.sampler is not None:
        #     seg_weight = self.sampler.sample(seg_logit, seg_label)
        # else:
        #     seg_weight = None

        real = seg_label.flatten(start_dim=1)
        pred = seg_logit.flatten(start_dim=1)
        # alpha = (real - pred).mean(dim=1, keepdim=True)
        loss_mse = 0.5 * F.mse_loss(pred, real, reduction='mean')
        loss['loss_mse'] = loss_mse
        return loss
# Copyright (c) OpenMMLab. All rights reserved.

@HEADS.register_module()
class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 mask_ratio=0.0,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()
        self.mask_ratio = mask_ratio

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding # 加上了位置编码的信息

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature 去掉全局信息，得到图像信息

        patches = self.head(features) # 用head得到patchs
        mask = torch.zeros_like(patches)
        mask[T:] = 1  # mask其他的像素全部设为 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches) # 得到 重构之后的 img
        mask = self.patch2img(mask)

        return img, mask
    def forward_train(self, features,backward_indexes, img):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        predicted_img,mask = self.forward(features,backward_indexes)
        losses = self.losses(predicted_img,mask, img)
        return losses

    def losses(self, predicted_img,mask, img):
        """Compute segmentation loss."""
        loss = dict()

        loss = torch.mean((predicted_img - img) ** 2 * mask) / self.mask_ratio
        loss['loss_mse'] = loss
        return loss

@HEADS.register_module()
class Discriminator(BaseDecodeHead):
    def __init__(self, in_channels,**kwargs):
        super(Discriminator, self).__init__(in_channels,**kwargs)
        self.in_channels = in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.SyncBatchNorm(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 128, kernel_size=1),
            nn.SyncBatchNorm(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 32, kernel_size=1),
            nn.SyncBatchNorm(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 2, kernel_size=1)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class SpatialGatherModule(nn.Module):
    """Aggregate the context features according to the initial predicted
    probability distribution.

    Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, scale):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale

    def forward(self, feats, probs):
        """Forward function."""
        batch_size, num_classes, height, width = probs.size()
        channels = feats.size(1)
        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1)
        # [batch_size, height*width, num_classes]
        feats = feats.permute(0, 2, 1)
        # [batch_size, channels, height*width]
        probs = F.softmax(self.scale * probs, dim=2)
        # [batch_size, channels, num_classes]
        ocr_context = torch.matmul(probs, feats)
        # ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return ocr_context


class ObjectAttentionBlock(_SelfAttentionBlock):
    """Make a OCR used SelfAttentionBlock."""

    def __init__(self, in_channels, channels, scale, conv_cfg, norm_cfg,
                 act_cfg):
        if scale > 1:
            query_downsample = nn.MaxPool2d(kernel_size=scale)
        else:
            query_downsample = None
        super(ObjectAttentionBlock, self).__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=query_downsample,
            key_downsample=None,
            key_query_num_convs=2,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            in_channels * 2,
            in_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, query_feats, key_feats):
        """Forward function."""
        context = super(ObjectAttentionBlock,
                        self).forward(query_feats, key_feats)
        output = self.bottleneck(torch.cat([context, query_feats], dim=1))
        if self.query_downsample is not None:
            output = resize(query_feats)

        return output


@HEADS.register_module()
class CenterHead(BaseCascadeDecodeHead):
    """Object-Contextual Representations for Semantic Segmentation.

    This head is the implementation of `OCRNet
    <https://arxiv.org/abs/1909.11065>`_.

    Args:
        ocr_channels (int): The intermediate channels of OCR block.
        scale (int): The scale of probability map in SpatialGatherModule in
            Default: 1.
    """

    def __init__(self, scale=1, **kwargs):
        super(CenterHead, self).__init__(**kwargs)
        self.scale = scale
        self.spatial_gather_module = SpatialGatherModule(self.scale)

        self.bottleneck = ConvModule(
            self.channels,
            self.channels,
            3,
            padding=1)

    def forward(self, inputs, prev_output):
        """Forward function."""
        #x = self._transform_inputs(inputs)
        # feats = self.bottleneck(inputs)
        context = self.spatial_gather_module(inputs, prev_output)
        

        return context
    def forward_train(self, inputs, prev_output,text_embeddings):
        
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        centerfeature = self.forward(inputs,prev_output)
        
        losses = self.losses(centerfeature, text_embeddings)
        return losses
        
    def losses(self, centerfeature, text_embeddings):
        """Compute segmentation loss."""
        loss = dict()

        con_loss = ContrastiveLoss()
        loss_con = 0.5 * con_loss(centerfeature, text_embeddings)

        
        loss['loss_con'] = loss_con
        return loss



