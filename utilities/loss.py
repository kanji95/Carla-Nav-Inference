import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# class FocalLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(FocalLoss, self).__init__()

#     def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):

#         #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)

#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)

#         #first compute binary cross-entropy
#         BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
#         BCE_EXP = torch.exp(-BCE)
#         focal_loss = alpha * (1-BCE_EXP)**gamma * BCE

#         return focal_loss


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
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
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

def _pointwise_loss(lambd, input, target, size_average=True, reduce=True):
    d = lambd(input, target)
    if not reduce:
        return d
    return torch.mean(d) if size_average else torch.sum(d)
        
class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()

    def KLD(self, inp, trg):
        inp = inp/torch.sum(inp)
        trg = trg/torch.sum(trg)
        eps = 1e-7

        return torch.sum(trg*torch.log(eps+torch.div(trg,(inp+eps))))

    def forward(self, inp, trg):
        return _pointwise_loss(lambda a, b: self.KLD(a, b), inp, trg)

class ClassLevelLoss(nn.Module):
    def __init__(self, loss_func, beta=0.6):
        super(ClassLevelLoss, self).__init__()
        
        self.combo_loss = ComboLoss()
        self.bce_loss = nn.BCELoss()
        self.kld_loss = KLDLoss()
        self.beta = beta
        self.loss_func = loss_func

    # B, C, H, W
    def forward(self, inputs, targets):
        # print(inputs.shape, targets.shape)
        if "bce" in self.loss_func:
            return self.beta * self.bce_loss(inputs[:, 0], targets[:, 0]) + (1 - self.beta) * self.bce_loss(inputs[:, 1], targets[:, 1])
        elif "kldiv" in self.loss_func:
            return self.beta * self.kld_loss(inputs[:, 0], targets[:, 0]) + (1 - self.beta) * self.kld_loss(inputs[:, 1], targets[:, 1])
        elif "combo" in self.loss_func:
            return self.beta * self.combo_loss(inputs[:, 0], targets[:, 0]) + (1 - self.beta) * self.combo_loss(inputs[:, 1], targets[:, 1])
        else:
            raise NotImplementedError(f"{self.loss_func} not implemented!") 
