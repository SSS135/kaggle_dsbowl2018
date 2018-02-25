import torch


def dice_loss(inputs, targets, reduce=True):
    num = targets.size(0)
    m1 = inputs.view(num, -1)
    m2 = targets.view(num, -1)
    intersection = (m1 * m2)
    score = 2. * intersection.sum(1) / (m1.sum(1) + m2.sum(1) + 1e-6)
    score = 1 - score
    return score.mean() if reduce else score


def soft_dice_loss(inputs, targets, reduce=True):
    num = targets.size(0)
    m1 = inputs.view(num, -1)
    m2 = targets.view(num, -1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
    score = 1 - score
    return score.mean() if reduce else score


def clipped_mse_loss(pred, target, cmin, cmax, reduce=True):
    loss_nonclip = F.mse_loss(pred, target, reduce=False)
    loss_clip = F.mse_loss(pred.clamp(cmin, cmax), target, reduce=False)
    loss = torch.max(loss_nonclip, loss_clip)
    return loss.mean() if reduce else loss
