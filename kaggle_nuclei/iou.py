import numba
import numpy as np


def iou(pred, target, bin_threshold=0.5):
    pred = pred.data > bin_threshold
    target = target.data > 0
    union = (pred | target).sum()
    if union == 0:
        return 1
    else:
        return (pred & target).sum() / union


def threshold_iou(iou, tmin=0.5, tmax=0.95, steps=10):
    return (iou > np.linspace(tmin, tmax, steps)).mean()


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


