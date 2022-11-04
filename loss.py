

import segmentation_models_pytorch.utils.losses as vanilla_losses
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

""" Consistency loss from https://arxiv.org/pdf/2205.15428.pdf
"""
class ConsistencyTrainingLoss(nn.Module):

    def __init__(self, adaptive=True):
        super(ConsistencyTrainingLoss, self).__init__()
        self.epsilon = 1e-5
        self.adaptive = adaptive
        self.jaccard = vanilla_losses.JaccardLoss()

    def forward(self, new_mask, old_mask, new_seg, old_seg, iou_weight=None):
        def difference(mask1, mask2):
            return mask1 * (1 - mask2) + mask2 * (1 - mask1)

        vanilla_jaccard = vanilla_losses.JaccardLoss()(old_seg, old_mask)

        sil = torch.sum(
            difference(
                difference(new_mask, old_mask),
                difference(new_seg, old_seg))
        ) / torch.sum(torch.clamp(new_mask + old_mask + new_seg + old_seg, 0, 1) + self.epsilon)

        return (1 - iou_weight) * vanilla_jaccard + iou_weight * sil


class SIL(nn.Module):
    def __init__(self):
        super(SIL, self).__init__()
        self.epsilon = 1e-5

    def forward(self, new_mask, old_mask, new_seg, old_seg):
        def difference(mask1, mask2):
            return mask1 * (1 - mask2) + mask2 * (1 - mask1)

        perturbation_loss = torch.sum(
            difference(
                difference(new_mask, old_mask),
                difference(new_seg, old_seg))
        ) / torch.sum(torch.clamp(new_mask + old_mask + new_seg + old_seg, 0, 1) + self.epsilon)  # normalizing factor
        return perturbation_loss