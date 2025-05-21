import torch
import torch.nn as nn
from mmseg.models.builder import LOSSES
import torch.nn.functional as F


def mse_loss(pred, target):
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.pow(pred - target, 2).mean()
    return loss

@LOSSES.register_module()
class MSELoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=0.01):
        super(MSELoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * mse_loss(pred, target)
        return loss
        
@LOSSES.register_module()
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, class_centers, text_features):
        
        features1 = F.normalize(class_centers.squeeze(-1), p=2, dim=-1)
        features2 = F.normalize(text_features.transpose(1, 2), p=2, dim=-1)
    
        # Calculate cosine similarity (similarity matrix)
        similarity_matrix = torch.matmul(features1.transpose(1, 2), features2)  # [B, 6, 6]
        
        batch_size, num_classes, _ = similarity_matrix.shape
        labels = torch.arange(num_classes, device=similarity_matrix.device).repeat(batch_size)  # [B*6,]
        
        # Apply temperature scaling
        similarity_matrix /= self.temperature
        
        # Compute the loss using the cross-entropy loss
        loss = 0.5 * F.cross_entropy(similarity_matrix.view(-1, num_classes), labels)
        
        
        return loss