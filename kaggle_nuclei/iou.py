import numpy as np
import torch


def iou(pred, target):
    assert pred.dim() > 2 and pred.shape == target.shape
    assert type(pred) == type(target) and type(pred).__name__.find('ByteTensor') != -1, (type(pred), type(target))
    bs = pred.shape[0]
    pred, target = pred.view(bs, -1), target.view(bs, -1)
    union = (pred | target).long().sum(-1)
    intersection = (pred & target).long().sum(-1)
    metric = intersection.float() / union.clamp(min=1).float()
    metric[union == 0] = 1
    return metric


def threshold_iou(iou, tmin=0.5, tmax=0.95, steps=10):
    return (iou.unsqueeze(1) > torch.linspace(tmin, tmax, steps).type_as(iou)).float().mean(1)


# @numba.jit
def mean_threshold_object_iou(pred_labels, target_labels, tmin=0.5, tmax=0.95, threshold_count=10):
    pred_status, target_status = object_iou(pred_labels, target_labels)
    thresholds = np.linspace(tmin, tmax, threshold_count).reshape(-1, 1)

    pred_mask = pred_status > thresholds
    target_mask = target_status > thresholds

    TP = target_mask.sum(-1)
    FP = np.invert(target_mask).sum(-1)
    FN = np.invert(pred_mask).sum(-1)

    div = TP + FP + FN
    ious = TP / div.clip(min=1)
    ious[div == 0] = 1

    return ious.mean()


def object_iou(pred_labels, target_labels):
    pred_count, target_count = pred_labels.max(), target_labels.max()
    pred_status, target_status = np.zeros(pred_count), np.zeros(target_count)
    for t_idx in range(1, target_count + 1):
        t_label = target_labels == t_idx
        if t_label.sum() == 0:
            continue
        p_overlap_idx = np.unique(pred_labels * t_label)
        p_overlap_idx = p_overlap_idx[p_overlap_idx != 0]

        if len(p_overlap_idx) == 0:
            continue

        p_label_tiles = pred_labels == p_overlap_idx.reshape(-1, 1, 1)
        iou = (p_label_tiles & t_label).sum(axis=(1, 2)) / (p_label_tiles | t_label).sum(axis=(1, 2))
        pred_status[p_overlap_idx - 1] = np.maximum(pred_status[p_overlap_idx - 1], iou)
        target_status[t_idx - 1] = iou.max()
    return pred_status, target_status


