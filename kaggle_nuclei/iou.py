import numpy as np
from itertools import count
import numba


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


@numba.jit
def mean_threshold_object_iou(pred_labels, target_labels, tmin=0.5, tmax=0.95, threshold_count=10):
    pred_status, target_status = object_iou(pred_labels, target_labels)
    thresholds = np.linspace(tmin, tmax, threshold_count)
    ious = []
    for threshold in thresholds:
        pred_mask = pred_status > threshold
        target_mask = target_status > threshold
        TP = target_mask.sum()
        FP = np.invert(target_mask).sum()
        FN = np.invert(pred_mask).sum()
        ious.append(TP / (TP + FP + FN))
    return np.mean(ious)


@numba.jit
def object_iou(pred_labels, target_labels):
    pred_count, target_count = pred_labels.max(), target_labels.max()
    pred_status, target_status = np.zeros(pred_count), np.zeros(target_count)
    for t_idx in range(1, target_count + 1):
        t_label = target_labels == t_idx
        p_overlap_idx = np.unique(pred_labels * t_label)
        for p_idx in p_overlap_idx:
            if p_idx == 0:
                continue
            p_label = pred_labels == p_idx
            iou = (p_label & t_label).sum() / (p_label | t_label).sum()
            pred_status[p_idx - 1] = max(pred_status[p_idx - 1], iou)
            target_status[t_idx - 1] = max(target_status[t_idx - 1], iou)
    return pred_status, target_status


