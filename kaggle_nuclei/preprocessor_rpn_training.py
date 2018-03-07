import copy
import math
import sys

import torch
import torch.nn.functional as F
import torch.utils.data
from optfn.cosine_annealing import CosineAnnealingRestartLR
from optfn.gadam import GAdam
from optfn.param_groups_getter import get_param_groups
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support
from torch.autograd import Variable
from tqdm import tqdm
from torch import nn

from .dataset import NucleiDataset
from .feature_pyramid_network import FPN
from .iou import threshold_iou, iou
from .losses import dice_loss, soft_dice_loss, clipped_mse_loss
from .unet import UNet
from .ms_d_net import MSDNet
import numpy as np
import numpy.random as rng


def matthews_corrcoef_checked(pred, target, default):
    return matthews_corrcoef(pred, target) if pred.sum() != 0 and target.sum() != 0 else default


def binary_focal_loss_with_logits(pred, target, lam=2, reduce=True):
    pred = F.sigmoid(pred)
    p = pred * target + (1 - pred) * (1 - target)
    loss = -(1 - p).pow(lam) * p.log()
    return loss.mean() if reduce else loss


def train_preprocessor_rpn(train_data, epochs=15, pretrain_epochs=7, model=None, return_predictions_at_epoch=None):
    samples_per_image = 64

    dataset = NucleiDataset(train_data, supersample=1)
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=4, pin_memory=True)

    model = FPN(3).cuda() if model is None else model.cuda()
    optimizer = GAdam(get_param_groups(model), lr=2e-4, nesterov=0.75, weight_decay=5e-4,
                      avg_sq_mode='tensor', amsgrad=False)

    scheduler = CosineAnnealingRestartLR(optimizer, len(dataloader), 2)
    pad = dataloader.dataset.padding
    best_model = model
    best_score = -math.inf

    sys.stdout.flush()

    for epoch in range(epochs):
        with tqdm(dataloader) as pbar:
            model.freeze_pretrained_layers(epoch < pretrain_epochs)
            mask_fscore_ma, score_fscore_ma, mask_matthews_ma, score_matthews_ma = 0, 0, 0, 0

            for i, data in enumerate(pbar):
                img, labels, sdf, obj_sizes = [x.cuda() for x in data]
                x_train = torch.autograd.Variable(img)

                optimizer.zero_grad()
                model_out = model(x_train, pad)
                train_pairs = get_train_pairs(
                    labels, sdf, obj_sizes, model_out, model.mask_pixel_sizes, samples_per_image)

                if train_pairs is None:
                    continue

                pred_masks, target_masks, pred_scores, target_scores = train_pairs

                target_masks = Variable(target_masks)
                target_scores = Variable(target_scores)

                # print(pred_masks.shape, target_masks.shape, pred_scores.shape, target_scores.shape)
                # print([(v.mean(), v.std(), v.min(), v.max()) for v in (pred_masks.data, target_masks.data, pred_scores.data, target_scores.data)])

                if return_predictions_at_epoch is not None and return_predictions_at_epoch == epoch:
                    return pred_masks.data.cpu(), target_masks.data.cpu(), x_train.data.cpu(), labels.cpu(), best_model

                mask_loss = binary_focal_loss_with_logits(pred_masks, target_masks)
                score_loss = binary_focal_loss_with_logits(pred_scores, target_scores)
                loss = mask_loss + 0.1 * score_loss

                loss.backward()
                optimizer.step()
                scheduler.step()

                pred_mask_np = (pred_masks.data > 0).cpu().numpy().reshape(-1)
                target_mask_np = target_masks.data.byte().cpu().numpy().reshape(-1)
                pred_score_np = (pred_scores.data > 0).cpu().numpy().reshape(-1)
                target_score_np = target_scores.data.byte().cpu().numpy().reshape(-1)

                mask_matthews = matthews_corrcoef_checked(pred_mask_np, target_mask_np, mask_fscore_ma)
                score_matthews = matthews_corrcoef_checked(pred_score_np, target_score_np, score_fscore_ma)
                _, _, mask_fscore, _ = precision_recall_fscore_support(
                    pred_mask_np, target_mask_np, average='binary', warn_for=[])
                _, _, score_fscore, _ = precision_recall_fscore_support(
                    pred_score_np, target_score_np, average='binary', warn_for=[])

                bc = 1 - 0.99 ** (i + 1)
                mask_fscore_ma = 0.99 * mask_fscore_ma + 0.01 * mask_fscore
                score_fscore_ma = 0.99 * score_fscore_ma + 0.01 * score_fscore
                mask_matthews_ma = 0.99 * mask_matthews_ma + 0.01 * mask_matthews
                score_matthews_ma = 0.99 * score_matthews_ma + 0.01 * score_matthews
                pbar.set_postfix(E=epoch, MF=mask_fscore_ma / bc, SF=score_fscore_ma / bc,
                                 MM=mask_matthews_ma / bc, SM=score_matthews_ma / bc, refresh=False)

            score = mask_fscore_ma
            if mask_fscore_ma > best_score:
                best_score = score
                best_model = copy.deepcopy(model)
    return best_model


def get_train_pairs(
        labels, sdf, obj_sizes, net_out, pixel_sizes, samples_count,
        neg_to_pos_ratio=3, pos_sdf_threshold=0.6, neg_sdf_threshold=-0.3,
        pos_size_limits=(0.4, 0.75), neg_size_limits=(0.15, 1.5)):
    outputs = []
    for sample_idx in range(labels.shape[0]):
        net_out_sample = [(m[sample_idx, 0], s[sample_idx, 0]) for m, s in net_out]
        o = get_train_pairs_single(
            labels[sample_idx, 0], sdf[sample_idx, 0], obj_sizes[sample_idx, 0], net_out_sample,
            pixel_sizes, samples_count, neg_to_pos_ratio, pos_sdf_threshold, neg_sdf_threshold,
            pos_size_limits, neg_size_limits
        )
        outputs.extend(o)

    if len(outputs) == 0:
        return None

    pred_masks, target_masks, pred_scores, target_scores = [torch.cat(o, 0) for o in zip(*outputs)]
    return pred_masks.unsqueeze(1), target_masks.unsqueeze(1).float(), pred_scores, target_scores


def get_train_pairs_single(labels, sdf, obj_sizes, net_out, pixel_sizes, samples_count,
                           neg_to_pos_ratio, pos_sdf_threshold, neg_sdf_threshold,
                           pos_size_limits, neg_size_limits):
    resampled_layers = resample_data(labels, sdf, obj_sizes, pixel_sizes)
    outputs = []
    for layer_idx, layer_data in enumerate(zip(net_out, resampled_layers)):
        (out_masks, out_scores), (res_labels, res_sdf, res_sizes) = layer_data

        num_samples_left = samples_count - len(outputs)
        num_layers_left = len(net_out) - layer_idx
        num_layer_total_samples = round(num_samples_left / num_layers_left)
        num_layer_pos_samples = math.ceil(num_layer_total_samples / (neg_to_pos_ratio + 1))

        o = generate_samples_for_layer(
            out_masks, out_scores, res_labels, res_sdf, res_sizes,
            num_layer_pos_samples, neg_to_pos_ratio,
            pos_sdf_threshold, neg_sdf_threshold,
            pos_size_limits, neg_size_limits
        )
        if o is not None:
            outputs.append(o)
    return outputs


def resample_data(labels, sdf, obj_sizes, pixel_sizes):
    assert labels.shape == sdf.shape
    assert labels.dim() == 2
    resampled = []
    for px_size in pixel_sizes:
        assert labels.shape[-1] % px_size == 0
        if px_size == 1:
            res_labels, res_sdf, res_sizes = labels, sdf, obj_sizes
        else:
            res_labels = labels[px_size // 2 - 1::px_size, px_size // 2 - 1::px_size]
            res_sizes = obj_sizes[px_size // 2 - 1::px_size, px_size // 2 - 1::px_size] / px_size
            res_sdf = F.avg_pool2d(Variable(sdf.view(1, 1, *sdf.shape), volatile=True), px_size, px_size).data[0, 0]
        resampled.append((res_labels, res_sdf, res_sizes))
    return resampled


def generate_samples_for_layer(out_masks, out_scores, labels, sdf, obj_sizes,
                               max_pos_samples_count, neg_to_pos_ratio,
                               pos_sdf_threshold, neg_sdf_threshold,
                               pos_size_limits, neg_size_limits):
    def upscaled_indexes(mask, max_count):
        idx = mask.nonzero() * stride + border
        perm = torch.randperm(len(idx))[:max_count].type_as(idx)
        idx = idx[perm]
        idx = idx[:, 0] * labels.shape[-1] + idx[:, 1]
        return idx, perm

    def center_crop(image, centers):
        centers_y = centers / labels.shape[-1]
        centers_x = centers - centers_y * labels.shape[-1]
        centers = torch.stack([centers_y, centers_x], 1).cpu()
        assert centers.shape == (centers_x.shape[0], 2), centers.shape
        crops = []
        for c in centers:
            crop = image[c[0] - border: c[0] + border, c[1] - border: c[1] + border]
            crops.append(crop)
        return torch.stack(crops, 0)

    border = FPN.mask_size // 2
    stride = FPN.mask_size // FPN.mask_kernel_size
    mask_centers_slice = (
        slice(border, -border + 1, stride),
        slice(border, -border + 1, stride))
    sdf_centers = sdf[mask_centers_slice]
    size_centers = obj_sizes[mask_centers_slice] / FPN.mask_size

    assert sdf_centers.shape == out_masks.shape[:2], (sdf_centers.shape, out_masks.shape)

    pos_centers_fmap = (sdf_centers > pos_sdf_threshold) & \
                       (size_centers > pos_size_limits[0]) & \
                       (size_centers < pos_size_limits[1])
    neg_centers_fmap = (sdf_centers < neg_sdf_threshold) | \
                       (size_centers < neg_size_limits[0]) | \
                       (size_centers > neg_size_limits[1])

    # TODO: allow zero negative centers
    if pos_centers_fmap.sum() == 0 or neg_centers_fmap.sum() == 0:
        return None

    pos_centers, pos_centers_perm = upscaled_indexes(pos_centers_fmap, max_pos_samples_count)
    neg_centers, neg_centers_perm = upscaled_indexes(neg_centers_fmap, len(pos_centers) * neg_to_pos_ratio)

    pos_centers_fmap_idx = pos_centers_fmap.view(-1).nonzero().squeeze()
    neg_centers_fmap_idx = neg_centers_fmap.view(-1).nonzero().squeeze()
    pred_pos_scores_idx = pos_centers_fmap_idx[pos_centers_perm]
    pred_neg_scores_idx = neg_centers_fmap_idx[neg_centers_perm]
    pred_pos_scores = out_scores.take(Variable(pred_pos_scores_idx))
    pred_neg_scores = out_scores.take(Variable(pred_neg_scores_idx))

    pred_scores = torch.cat([pred_pos_scores, pred_neg_scores])
    target_scores = out_scores.data.new(pred_scores.shape[0]).fill_(0)
    target_scores[:pred_pos_scores.shape[0]] = 1

    label_crops = center_crop(labels, pos_centers)
    pos_center_label_nums = labels.take(pos_centers)
    target_masks = label_crops == pos_center_label_nums.view(-1, 1, 1)
    pred_masks = out_masks.view(-1, FPN.mask_size, FPN.mask_size).index_select(0, Variable(pred_pos_scores_idx))

    return pred_masks, target_masks, pred_scores, target_scores
