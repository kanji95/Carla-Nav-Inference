import torch
import torch.nn as nn
import torch.nn.functional as F


#PyTorch
# ALPHA = 0.8
# GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE

        return focal_loss


class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, smooth=1, alpha=0.8, ce_ratio=0.4, beta=0.5, eps=1e-7):
        super(ComboLoss, self).__init__()
        
        self.smooth = smooth
        self.alpha = alpha
        self.ce_ratio = ce_ratio
        self.beta = beta
        self.eps = eps

    def forward(self, inputs, targets):

        # if inputs.max() > 1 or inputs.min() < 0:
        #     import pdb; pdb.set_trace()
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()    
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        inputs = torch.clamp(inputs,min=self.eps, max=1.0 - self.eps)
        out = - (self.alpha * ((targets * torch.log(inputs)) + ((1 - self.alpha) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (self.ce_ratio * weighted_ce) - ((1 - self.ce_ratio) * dice)

        # if torch.isnan(out).any():
        #     import pdb; pdb.set_trace()
        
        assert not torch.isnan(out).any()
        assert not torch.isnan(dice).any()
        assert not torch.isnan(combo).any()

        return combo