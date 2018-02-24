import numpy as np
from itertools import count


def iou(pred, target, bin_threshold=0.5):
    pred = pred.data > bin_threshold
    target = target.data > 0
    union = (pred + target).clamp(max=1).sum()
    if union == 0:
        return 0
    else:
        return (pred * target).sum() / union


def threshold_iou(iou, tmin=0.5, tmax=0.95, steps=10):
    return (iou > np.linspace(tmin, tmax, steps)).mean()


def mean_threshold_iou(pred, target, bin_threshold=0.5, tmin=0.5, tmax=0.95, steps=10):
    pred = (pred.data > bin_threshold).long()
    target = target.data.mul(255).round_().long()
    steps = np.linspace(tmin, tmax, steps)
    ious = []
    for obj_idx in count():
        mask = target == obj_idx
        if mask.sum() == 0:
            break
        iou = (pred * mask).sum() / (pred + mask).clamp(max=1).sum()
        ious.append(iou > steps)
    return np.mean(ious)


