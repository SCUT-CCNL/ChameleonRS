# Copyright (c) OpenMMLab. All rights reserved.
from .bisenetv1 import BiSeNetV1
from .bisenetv2 import BiSeNetV2
from .cgnet import CGNet
from .fast_scnn import FastSCNN
from .hrnet import HRNet
from .icnet import ICNet
from .mit import MixVisionTransformer
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1c, ResNetV1d
from .resnext import ResNeXt
from .swin import SwinTransformer
from .timm_backbone import TIMMBackbone
from .unet import UNet
from .vit import VisionTransformer

from .mae import MAE
from .beit import BEiT
__all__ = [
    'ResNet', 'ResNetV1c', 'ResNetV1d', 'ResNeXt', 'HRNet', 'FastSCNN',
    'ResNeSt', 'MobileNetV2', 'UNet', 'CGNet', 'MobileNetV3',
    'VisionTransformer', 'SwinTransformer', 'MixVisionTransformer',
    'BiSeNetV1', 'BiSeNetV2', 'ICNet', 'TIMMBackbone','BEiT', 'MAE'
]

