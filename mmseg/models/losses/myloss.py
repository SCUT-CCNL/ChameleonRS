import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
@LOSSES.register_module()
class CosineLoss(nn.Module):
    def __init__(self, epsilon=1e-7, scale_factor=0.1):
        super(CosineLoss, self).__init__()
        self.epsilon = epsilon
        self.scale_factor = scale_factor

    def forward(self, pred, target):
        """
        计算余弦相似度损失

        Args:
            pred (Tensor): 模型的预测，形状为 (N, C, H, W)
            target (Tensor): 目标图像，形状为 (N, C, H, W)

        Returns:
            Tensor: 计算得到的余弦损失
        """
        assert pred.shape == target.shape, "The shapes of pred and target must match."

        # 展平张量 [B, C, H*W]
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)  # (N, C, H*W)
        target_flat = target.view(target.size(0), target.size(1), -1)  # (N, C, H*W)

        # 计算点积和 L2 范数
        dot_product = torch.einsum('bci,bdi->bc', pred_flat, target_flat)
        norm_pred = torch.norm(pred_flat, dim=2) + self.epsilon
        norm_target = torch.norm(target_flat, dim=2) + self.epsilon

        # 计算余弦相似度
        cosine_similarity = dot_product / (norm_pred * norm_target)

        # 计算余弦损失，并乘以缩放因子
        loss = torch.mean((1 - cosine_similarity) ** 2) * self.scale_factor

        return loss