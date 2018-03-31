import torch
import torch.nn.functional as F


def soft_dice_loss_with_logits(inputs, targets, reduce=True):
    return soft_dice_loss(inputs.sigmoid(), targets, reduce)


def soft_dice_loss(inputs, targets, reduce=True):
    num = targets.size(0)
    m1 = inputs.contiguous().view(num, -1)
    m2 = targets.contiguous().view(num, -1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
    score = 1 - score
    return score.mean() if reduce else score
