# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from torch import nn

from mmseg.core import add_prefix
from ..builder import HEADS
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from .decode_head import BaseDecodeHead


class PAM(_SelfAttentionBlock):
    """Position Attention Module (PAM)

    Args:
        in_channels (int): Input channels of key/query feature.
        channels (int): Output channels of key/query transform.
    """

    def __init__(self, in_channels, channels):
        super(PAM, self).__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=1,
            key_query_norm=False,
            value_out_num_convs=1,
            value_out_norm=False,
            matmul_norm=False,
            with_out=False,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)

        self.gamma = Scale(0)

    def forward(self, x):
        """Forward function."""
        out = super(PAM, self).forward(x, x)

        out = self.gamma(out) + x
        return out


class CAM(nn.Module):
    """Channel Attention Module (CAM)"""

    def __init__(self):
        super(CAM, self).__init__()
        self.gamma = Scale(0)

    def forward(self, x):
        """Forward function."""
        batch_size, channels, height, width = x.size()
        proj_query = x.view(batch_size, channels, -1)
        proj_key = x.view(batch_size, channels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new, dim=-1)
        proj_value = x.view(batch_size, channels, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, channels, height, width)

        out = self.gamma(out) + x
        return out


@HEADS.register_module()
class DAHead(BaseDecodeHead):
    """Dual Attention Network for Scene Segmentation.

    This head is the implementation of `DANet
    <https://arxiv.org/abs/1809.02983>`_.

    Args:
        pam_channels (int): The channels of Position Attention Module(PAM).
    """

    def __init__(self, pam_channels, **kwargs):
        super(DAHead, self).__init__(**kwargs)
        self.pam_channels = pam_channels
        self.pam_in_conv = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.pam = PAM(self.channels, pam_channels)
        self.pam_out_conv = ConvModule(
            self.channels,
            self.channels,
            3,
            padding=1,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.pam_conv_seg = nn.Conv2d(
            self.channels, self.num_classes, kernel_size=1)

        self.cam_in_conv = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.cam = CAM()
        self.cam_out_conv = ConvModule(
            self.channels,
            self.channels,
            3,
            padding=1,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.cam_conv_seg = nn.Conv2d(
            self.channels, self.num_classes, kernel_size=1)
        self.my_conv_seg = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.SyncBatchNorm(512),
            # nn.LeakyReLU(0.2,inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.SyncBatchNorm(256),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(256, self.num_classes, 1)
        )
        self.down = nn.Conv2d(512, 64, kernel_size=1)


    def pam_cls_seg(self, feat):
        """PAM feature classification."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.pam_conv_seg(feat)
        return output

    def my_cls_seg(self, feat):
        """PAM feature classification."""
        # if self.dropout is not None:
        #     feat = self.dropout(feat)
        output = self.my_conv_seg(feat)
        # output = self.conv_seg(output)
        return output

    def cam_cls_seg(self, feat):
        """CAM feature classification."""
        # if self.dropout is not None:
        #     feat = self.dropout(feat)
        output = self.cam_conv_seg(feat)
        return output

    def forward(self, inputs):
        """Forward function."""
        #x = self._transform_inputs(inputs)
        x = inputs[-1]
        # print(x.shape)
        # x = torch.cat([inputs[-1],inputs[-2]],dim=1)
        # low_level_feat = F.interpolate(self.layer2(inputs[1]),size=inputs[0].shape[2:],mode='bilinear',align_corners=True)
        pam_feat = self.pam_in_conv(x)
        pam_feat = self.pam(pam_feat)
        pam_feat = self.pam_out_conv(pam_feat)

        pam_out = self.pam_cls_seg(pam_feat)

        cam_feat = self.cam_in_conv(x)
        cam_feat = self.cam(cam_feat)
        cam_feat = self.cam_out_conv(cam_feat)
        cam_out = self.cam_cls_seg(cam_feat)

        # feat_sum = torch.cat([pam_feat , cam_feat],dim=1)
        feat_sum = pam_feat+cam_feat
        # expand_pam = F.interpolate(feat_sum, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        # expand_pam = torch.cat([feat_sum, self.down(inputs[1])], dim=1)
        pam_cam_out = self.cls_seg(feat_sum)


        return pam_cam_out,pam_out,cam_out

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, only ``pam_cam`` is used."""
        # print(self.forward(inputs)[0].shape)
        return self.forward(inputs)[0]

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        dict1 = torch.load(pretrained)
        self.load_state_dict(dict1, False)

    def losses(self, seg_logit, seg_label):
        """Compute ``pam_cam``, ``pam``, ``cam`` loss."""
        pam_cam_seg_logit,pam_seg_logit, cam_seg_logit = seg_logit
        # print(seg_label)
        loss = dict()
        loss.update(
            add_prefix(
                super(DAHead, self).losses(pam_cam_seg_logit, seg_label),
                'final'))
        # loss.update(
        #     add_prefix(
        #         super(DAHead, self).losses(pam_seg_logit, seg_label), 'pam'))
        # loss.update(
        #     add_prefix(
        #         super(DAHead, self).losses(cam_seg_logit, seg_label), 'cam'))
        return loss

