import copy
import math
import sys

import torch
import torch.nn.functional as F
import torch.utils.data
from optfn.cosine_annealing import CosineAnnealingRestartParam
from optfn.gadam import GAdam
from optfn.param_groups_getter import get_param_groups
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support
from torch.autograd import Variable
from tqdm import tqdm
from torch import nn

from ..dataset import NucleiDataset, train_pad, train_size
from .feature_pyramid_network import FPN
from ..iou import threshold_iou, iou
from ..losses import dice_loss, soft_dice_loss, clipped_mse_loss
import numpy as np
import numpy.random as rng
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.batchnorm import _BatchNorm
import time
from collections import OrderedDict
from tensorboardX import SummaryWriter
from ..roi_align import roi_align, pad_boxes


box_padding = 0.2


# def binary_cross_entropy_with_logits(x, z, reduce=True):
#     bce = x.clamp(min=0) - x * z + x.abs().neg().exp().add(1).log()
#     return bce.mean() if reduce else bce


def binary_focal_loss_with_logits(input, target, lam=2):
    weight = (target - F.sigmoid(input)).abs().pow(lam)
    ce = F.binary_cross_entropy_with_logits(input, target, weight=weight)
    return ce


# def mse_focal_loss(pred, target, lam=2, reduce=True):
#     mse = F.mse_loss(pred, target, reduce=False)
#     loss = (pred - target).clamp(-1, 1).abs().pow(lam) * mse
#     return loss.mean() if reduce else loss


def copy_state_dict(model):
    return copy.deepcopy(OrderedDict((k, v.cpu()) for k, v in model.state_dict().items()))


def batch_to_instance_norm(model):
    for module in model.modules():
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                norm = nn.InstanceNorm2d(child.num_features, affine=child.affine)
                norm.weight = child.weight
                norm.bias = child.bias
                setattr(module, name, child)


def train(train_data, epochs=15, pretrain_epochs=7, saved_model=None, return_predictions_at_epoch=None):
    dataset = NucleiDataset(train_data)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1, pin_memory=True)

    model_gen = FPN(3).cuda() if saved_model is None else saved_model.cuda()
    batch_to_instance_norm(model_gen)
    model_gen.freeze_pretrained_layers(False)
    # model_disc = GanD(4).cuda()

    # model_gen.apply(weights_init)
    # model_disc.apply(weights_init)

    # optimizer_gen = torch.optim.SGD(get_param_groups(model_gen), lr=0.02, momentum=0.9, weight_decay=1e-4)
    optimizer_gen = GAdam(get_param_groups(model_gen), lr=2e-4, betas=(0.9, 0.999), avg_sq_mode='weight',
                          amsgrad=False, nesterov=0.5, weight_decay=1e-4)
    # optimizer_disc = GAdam(get_param_groups(model_disc), lr=1e-4, betas=(0.9, 0.999), avg_sq_mode='tensor',
    #                       amsgrad=False, nesterov=0.5, weight_decay=1e-4, norm_weight_decay=False)

    # scheduler_gen = CosineAnnealingRestartParam(optimizer_gen, len(dataloader), 2)
    # scheduler_disc = CosineAnnealingRestartLR(optimizer_disc, len(dataloader), 2)

    best_state_dict = copy_state_dict(model_gen)
    best_score = -math.inf

    # one = Variable(torch.cuda.FloatTensor([0.95]))
    # zero = Variable(torch.cuda.FloatTensor([0.05]))

    # summary = SummaryWriter()
    # summary.add_text('hparams', f'epochs {epochs}; pretrain {pretrain_epochs}; '
    #                             f'batch size {dataloader.batch_size}; img size {train_size}; img pad {train_pad}')

    sys.stdout.flush()

    for epoch in range(epochs):
        with tqdm(dataloader) as pbar:
            model_gen.freeze_pretrained_layers(epoch < pretrain_epochs)
            score_fscore_sum, t_iou_sum, f_iou_sum, box_iou_sum = 0, 0, 0, 0

            batch_masks = np.zeros(len(model_gen.mask_pixel_sizes))

            for batch_idx, data in enumerate(pbar):
                img, labels, sdf = [x.cuda() for x in data]
                x_train = Variable(img)
                sdf_train = Variable(sdf * 0.5 + 0.5)
                mask_train = Variable((labels > 0).float())

                cont_train = 1 - sdf ** 2
                cont_train = (cont_train.clamp(0.9, 1) - 0.9) * 10
                cont_train = Variable(cont_train)

                optimizer_gen.zero_grad()

                model_out_layers, model_out_img = model_gen(x_train, train_pad)
                train_pairs = get_train_pairs(
                    model_gen, labels, sdf, img[:, :, train_pad:-train_pad, train_pad:-train_pad], model_out_layers)

                if train_pairs is None:
                    continue

                pred_masks, target_masks, pred_boxes, target_boxes, pred_scores, target_scores, img_crops, layer_idx = train_pairs

                pred_masks = model_gen.predict_masks(pred_masks)

                bm_idx, bm_count = np.unique(layer_idx, return_counts=True)
                batch_masks[bm_idx] += bm_count

                if return_predictions_at_epoch is not None and return_predictions_at_epoch == epoch:
                    return pred_masks.data.cpu(), target_masks.cpu(), img_crops.cpu(), \
                           x_train.data.cpu(), labels.cpu(), model_out_img.data.cpu(), model_gen

                target_masks = Variable(target_masks)
                target_scores = Variable(target_scores)
                target_boxes = Variable(target_boxes)
                # img_crops = Variable(img_crops)
                #
                # # real
                #
                # # for p in model_disc.parameters():
                # #     p.data.clamp_(-0.01, 0.01)
                #
                # optimizer_disc.zero_grad()
                #
                # read_disc_in = torch.cat([img_crops, target_masks], 1)
                # real_d, real_features = model_disc(read_disc_in)
                # loss_real = 0
                # # loss_real += -real_d.mean()
                # loss_real += F.binary_cross_entropy_with_logits(real_d, one.expand_as(real_d))
                # # loss_real += 0.5 * (1 - real_d.clamp(max=1)).pow_(2).mean()
                # # loss_real += 0.5 * (1 - real_d).pow_(2).mean()
                # loss_real.backward()
                #
                # # fake
                #
                # fake_disc_in = torch.cat([img_crops, F.sigmoid(pred_masks)], 1)
                # fake_d, fake_features = model_disc(fake_disc_in.detach())
                # loss_fake = 0
                # # loss_fake += fake_d.mean()
                # loss_fake += F.binary_cross_entropy_with_logits(fake_d, zero.expand_as(fake_d))
                # # loss_fake += 0.5 * (-1 - fake_d.clamp(min=-1)).pow_(2).mean()
                # # loss_fake += 0.5 * (0 - fake_d).pow_(2).mean()
                # loss_fake.backward()
                #
                # # gradient_penalty = calc_gradient_penalty(model_disc, read_disc_in.data, fake_disc_in.data)
                # # gradient_penalty.backward()
                #
                # optimizer_disc.step()
                #
                # # gen
                #
                # gen_d, fake_features = model_disc(fake_disc_in)
                # loss_gen = 0
                # # loss_gen += -gen_d.mean()
                # # loss_gen += binary_focal_loss_with_logits(gen_d, one.expand_as(gen_d))
                # # loss_gen += 0.5 * (1 - gen_d.div(3).clamp(min=-1)).pow_(2).mean()
                # # loss_gen += 0.5 * (1 - gen_d).pow_(2).mean()
                # loss_gen += F.mse_loss(fake_features, real_features.detach())
                # # loss_gen += F.mse_loss(gen_d, real_d.detach())

                img_mask_out, img_sdf_out, img_cont_out = model_out_img.split(1, 1)
                # img_mask_out = F.sigmoid(img_mask_out)
                img_mask_loss = binary_focal_loss_with_logits(img_mask_out, mask_train)
                img_sdf_loss = binary_focal_loss_with_logits(img_sdf_out, sdf_train)
                img_cont_loss = binary_focal_loss_with_logits(img_cont_out, cont_train)

                mask_loss = binary_focal_loss_with_logits(pred_masks, target_masks)
                score_loss = binary_focal_loss_with_logits(pred_scores, target_scores)
                box_loss = F.mse_loss(pred_boxes, target_boxes)
                loss = mask_loss + box_loss + score_loss + (img_mask_loss + img_sdf_loss + img_cont_loss) / 3

                loss.backward()
                optimizer_gen.step()
                optimizer_gen.zero_grad()

                # scheduler_gen.step()
                # scheduler_disc.step()

                box_iou = aabb_iou(pred_boxes.data, target_boxes.data).mean()

                pred_score_np = (pred_scores.data > 0).cpu().numpy().reshape(-1)
                target_score_np = (target_scores.data > 0.5).byte().cpu().numpy().reshape(-1)

                _, _, score_fscore, _ = precision_recall_fscore_support(
                    pred_score_np, target_score_np, average='binary', warn_for=[])

                f_iou = iou(pred_masks.data > 0, target_masks.data > 0.5)
                t_iou = threshold_iou(f_iou)

                iter = batch_idx + 1
                score_fscore_sum += score_fscore
                f_iou_sum += f_iou.mean()
                t_iou_sum += t_iou.mean()
                box_iou_sum += box_iou
                pbar.set_postfix(
                    _E=epoch, SF=score_fscore_sum / iter,
                    BIoU=box_iou_sum / iter,
                    MIoU=f_iou_sum / iter, MIoU_T=t_iou_sum / iter,
                    MPS=np.round(batch_masks / iter / img.shape[0], 1), refresh=False)

            score = t_iou_sum
            if score > best_score:
                best_score = score
                best_state_dict = copy_state_dict(model_gen)
    model_gen.load_state_dict(best_state_dict)
    return model_gen


def get_train_pairs(
        model, labels, sdf, img, net_out,
        pos_sdf_threshold=0.1, neg_sdf_threshold=-0.1,
        pos_iou_limit=0.4, neg_iou_limit=0.3,
        pos_samples=32, neg_to_pos_ratio=3):
    outputs = []
    for sample_idx in range(labels.shape[0]):
        net_out_sample = [(m[sample_idx], s[sample_idx], (b[0][sample_idx], b[1])) for m, s, b in net_out]
        o = get_train_pairs_single(
            model, labels[sample_idx, 0], sdf[sample_idx, 0], img[sample_idx], net_out_sample,
            model.mask_pixel_sizes, pos_sdf_threshold, neg_sdf_threshold,
            pos_iou_limit, neg_iou_limit,
            pos_samples, neg_to_pos_ratio
        )
        outputs.extend(o)

    if len(outputs) == 0:
        return None

    outputs = list(zip(*outputs))
    outputs, layer_idx = outputs[:-1], np.concatenate(outputs[-1])
    pred_masks, target_masks, pred_boxes, target_boxes, pred_scores, target_scores, img_crops = \
        [torch.cat(o, 0) for o in outputs]
    return pred_masks, target_masks, pred_boxes, target_boxes, pred_scores, target_scores, img_crops, layer_idx


def get_train_pairs_single(model, labels, sdf, img, net_out, pixel_sizes,
                           pos_sdf_threshold, neg_sdf_threshold,
                           pos_iou_limit, neg_iou_limit,
                           pos_samples, neg_to_pos_ratio):
    box_mask = get_object_boxes(labels, 2)
    resampled_layers = resample_data(labels, sdf, box_mask, img, pixel_sizes)
    outputs = []
    for layer_idx, layer_data in enumerate(zip(net_out, resampled_layers)):
        (out_masks, out_scores, out_boxes), (res_labels, res_sdf, res_boxes, res_img) = layer_data

        o = generate_samples_for_layer(
            model, out_masks, out_scores, out_boxes, res_labels, res_sdf, res_boxes, res_img,
            pos_sdf_threshold, neg_sdf_threshold,
            pos_iou_limit, neg_iou_limit,
            pos_samples, neg_to_pos_ratio
        )
        if o is not None:
            outputs.append((*o, o[0].shape[0] * [layer_idx]))
    return outputs


def generate_samples_for_layer(model, out_features, out_scores, out_boxes, labels, sdf, obj_boxes, img,
                               pos_sdf_threshold, neg_sdf_threshold,
                               pos_iou_limit, neg_iou_limit,
                               pos_samples, neg_to_pos_ratio, box_padding=0.15):
    out_boxes, anchor_boxes = out_boxes

    # border and stride for converting between image space and conv-center space
    stride = model.mask_size // model.region_size
    border = stride // 2

    # slice to select values from image at conv center locations
    mask_centers_slice = (
        slice(border, -border + 1, stride),
        slice(border, -border + 1, stride))
    # [fs, fs] - sdf at conv centers
    sdf_fs = sdf[mask_centers_slice]
    # [(y, x, h, w), fs, fs] - obj boxes at conv centers
    target_boxes_fs = obj_boxes[(slice(None), *mask_centers_slice)].contiguous()
    # target_boxes_fs[0] -= target_boxes_fs[2] * box_padding
    # target_boxes_fs[1] -= target_boxes_fs[3] * box_padding
    # target_boxes_fs[2:] *= 1 + 2 * box_padding
    target_boxes_fs = target_boxes_fs.float() / target_boxes_fs.new(2 * [labels.shape[0], labels.shape[1]]).view(-1, 1, 1)

    assert target_boxes_fs.shape[-1] == anchor_boxes.shape[-1], (target_boxes_fs.shape, anchor_boxes.shape, out_boxes.shape, out_features.shape, labels.shape, sdf.shape, img.shape)

    anchor_iou = aabb_iou(
        target_boxes_fs.unsqueeze(1).expand_as(anchor_boxes).contiguous().view(anchor_boxes.shape[0], -1).t(),
        anchor_boxes.view(anchor_boxes.shape[0], -1).t()
    ).view(*anchor_boxes.shape[1:])

    # assert sdf_fs.shape == out_masks.shape[-2:], (sdf_fs.shape, out_masks.shape)

    pos_centers_fmap = (anchor_iou > pos_iou_limit) & (sdf_fs > pos_sdf_threshold)
    neg_centers_fmap = (anchor_iou < neg_iou_limit) | (sdf_fs < neg_sdf_threshold)

    # TODO: allow zero negative centers
    if pos_centers_fmap.sum() == 0 or neg_centers_fmap.sum() == 0:
        return None

    pos_centers_fmap_idx_all = pos_centers_fmap.view(-1).nonzero().squeeze()
    neg_centers_fmap_idx_all = neg_centers_fmap.view(-1).nonzero().squeeze()
    pos_centers_fmap_perm = torch.randperm(len(pos_centers_fmap_idx_all))
    # neg_centers_fmap_perm = torch.randperm(len(neg_centers_fmap_idx_all))
    pos_centers_fmap_perm = pos_centers_fmap_perm[:pos_samples].contiguous().cuda()
    # neg_centers_fmap_perm = neg_centers_fmap_perm[:len(pos_centers_fmap_perm) * neg_to_pos_ratio].contiguous().cuda()
    pos_centers_fmap_idx = pos_centers_fmap_idx_all[pos_centers_fmap_perm]
    # neg_centers_fmap_idx = neg_centers_fmap_idx_all[neg_centers_fmap_perm]

    pred_pos_scores = out_scores.take(Variable(pos_centers_fmap_idx_all))
    pred_neg_scores = out_scores.take(Variable(neg_centers_fmap_idx_all))
    pred_scores = torch.cat([pred_pos_scores, pred_neg_scores])
    target_scores = out_scores.data.new(pred_scores.shape[0]).fill_(0)
    target_scores[:pred_pos_scores.shape[0]] = 1

    # ([y, x, h, w], NPos)
    pred_boxes = out_boxes.view(out_boxes.shape[0], -1)[:, pos_centers_fmap_idx]
    target_boxes = target_boxes_fs.unsqueeze(1).repeat(1, anchor_boxes.shape[1], 1, 1) \
                           .view(target_boxes_fs.shape[0], -1)[:, pos_centers_fmap_idx]

    pred_boxes, target_boxes = pred_boxes.t(), target_boxes.t()
    pred_boxes_pad = pad_boxes(pred_boxes.data, box_padding)

    pred_features = roi_align(out_features.unsqueeze(0), pred_boxes_pad, model.region_size)
    img_crops = roi_align(
        Variable(img.unsqueeze(0), volatile=True),
        pred_boxes_pad,
        model.mask_size
    ).data

    pos_center_label_nums = labels[mask_centers_slice].unsqueeze(0).repeat(anchor_boxes.shape[1], 1, 1)
    pos_center_label_nums = pos_center_label_nums.take(pos_centers_fmap_idx)

    target_masks = labels_to_mask_roi_align(labels, pred_boxes_pad, pos_center_label_nums, model.mask_size)

    # print([x.shape for x in (pred_masks, target_masks, pred_boxes, target_boxes, pred_scores, target_scores, img_crops)])

    return pred_features, target_masks, pred_boxes, target_boxes, pred_scores, target_scores, img_crops


def get_object_boxes(labels, downsampling=1):
    # [size, size]
    assert labels.dim() == 2

    count = labels.max()
    if count == 0:
        return torch.zeros(4, *labels.shape).cuda()

    if downsampling != 1:
        labels = downscale_nonzero(labels, downsampling)

    # [count] with [1, count]
    label_nums = torch.arange(1, count + 1).long().cuda()
    # [count, size, size] with [0, 1]
    masks = (labels.unsqueeze(0) == label_nums.view(-1, 1, 1)).float()

    # [new count] with [1, old count]
    nonzero_idx = (masks.view(masks.shape[0], -1).sum(-1) != 0).nonzero().squeeze()
    count = len(nonzero_idx)
    assert count != 0
    # [count, size, size] with [0, 1]
    masks = masks.index_select(0, nonzero_idx)

    # [size] with [0, size - 1] ascending
    size_range = torch.arange(labels.shape[0]).cuda()
    # [size] with [0, size - 1] descending
    size_range_rev = torch.arange(labels.shape[0] - 1, -1, -1).cuda()

    # [count, size, size] with [0, size), filtered by mask, ascending by 1 dim
    y_range_mask = masks * size_range.view(1, -1, 1)
    # [count, size, size] with [0, size), filtered by mask, descending by 1 dim
    y_range_mask_rev = masks * size_range_rev.view(1, -1, 1)
    # [count, size, size] with [0, size), filtered by mask, ascending by 2 dim
    x_range_mask = masks * size_range.view(1, 1, -1)
    # [count, size, size] with [0, size), filtered by mask, descending by 2 dim
    x_range_mask_rev = masks * size_range_rev.view(1, 1, -1)

    # [count] with [0, size)
    y_max = y_range_mask.view(count, -1).max(1)[0]
    x_max = x_range_mask.view(count, -1).max(1)[0]
    y_min = labels.shape[0] - 1 - y_range_mask_rev.view(count, -1).max(1)[0]
    x_min = labels.shape[0] - 1 - x_range_mask_rev.view(count, -1).max(1)[0]
    assert y_min.dim() == 1, y_min.shape

    # [count, 4] with [0, size], in format [y, x, h, w] at dim 1
    box_vec = torch.stack([y_min, x_min, y_max - y_min, x_max - x_min], 1)
    assert box_vec.shape == (count, 4)
    # [count, 4, size, size], filtered by mask, in format [y, x, h, w] at dim 1
    box_mask = masks.unsqueeze(1) * box_vec.view(count, 4, 1, 1)
    # [4, size, size], filtered by mask, in format [y, x, h, w] at dim 0
    box_mask = box_mask.sum(0)

    if downsampling != 1:
        box_mask = F.upsample(box_mask.unsqueeze(0), scale_factor=downsampling, mode='nearest').data.squeeze(0)
        box_mask *= downsampling

    return box_mask


def resample_data(labels, sdf, box_mask, img, pixel_sizes):
    # labels - [size, size]
    # sdf - [size, size]
    # box_mask - [4, size, size]
    # img - [3, size, size]
    # pixel_sizes - [layers count]
    assert labels.shape == sdf.shape
    assert labels.dim() == 2
    resampled = []
    for px_size in pixel_sizes:
        assert labels.shape[-1] % px_size == 0
        if px_size == 1:
            res_labels, res_sdf, res_boxes, res_img = labels, sdf, box_mask, img
        elif px_size < 1:
            assert round(1 / px_size, 3) == int(1 / px_size)
            factor = int(1 / px_size)
            res_labels = F.upsample(labels.view(1, 1, *labels.shape).float(), scale_factor=factor).data[0, 0].long()
            res_boxes = F.upsample(box_mask.unsqueeze(0), scale_factor=factor).data[0] / px_size
            res_sdf = F.upsample(sdf.view(1, 1, *sdf.shape), scale_factor=factor, mode='bilinear').data[0, 0]
            res_img = F.upsample(img.unsqueeze(0), scale_factor=factor, mode='bilinear').data[0]
        else:
            res_labels = downscale_nonzero(labels, px_size)
            res_boxes = downscale_nonzero(box_mask, px_size) / px_size
            res_sdf = F.avg_pool2d(Variable(sdf.view(1, 1, *sdf.shape), volatile=True), px_size, px_size).data[0, 0]
            res_img = F.avg_pool2d(Variable(img.unsqueeze(0), volatile=True), px_size, px_size).data[0]
        resampled.append((res_labels, res_sdf, res_boxes, res_img))
    return resampled


def downscale_nonzero(x, factor):
    assert x.shape[-1] == x.shape[-2]
    assert x.shape[-1] % factor == 0
    assert 2 <= x.dim() <= 4
    ns = x.shape[-1]
    ds = x.shape[-1] // factor
    # [norm size, norm size] - non zero column / pixel mask
    mask = x.view(-1, ns, ns).sum(0) != 0
    # [down size, down size, factor, factor]
    mask = mask.view(ds, factor, ds, factor).transpose(1, 2).float()
    zero_mask = (mask.view(ds, ds, -1).mean(-1) < 0.5).view(ds, ds, 1, 1).expand_as(mask)
    # inverting values of mask cells where max() should select zeros
    mask[zero_mask] *= -1
    # [-1, ds, ds, factor, factor]
    gx = x.view(-1, ds, factor, ds, factor).transpose(2, 3).contiguous()
    # [-1, ds, ds, factor * factor]
    gx = gx.view(*gx.shape[:3], -1)
    # [ds, ds] indices
    downscaled_indices = mask.view(ds, ds, -1).max(-1)[1]
    # [gx[0], ds, ds, 1]
    downscaled_indices = downscaled_indices.view(1, ds, ds, 1).expand(gx.shape[0], ds, ds, 1)
    gx = gx.gather(-1, downscaled_indices)
    gx = gx.sum(-1)
    gx = gx.view(*x.shape[:-2], ds, ds)
    assert gx.shape == (*x.shape[:-2], ds, ds), (x.shape, gx.shape)
    return gx


def labels_to_mask_roi_align(labels, boxes, label_nums, crop_size):
    assert labels.dim() == 2
    assert boxes.dim() == 2
    assert label_nums.dim() == 1
    masks = []
    for box, label_num in zip(boxes, label_nums):
        mask = (labels == label_num).view(1, 1, *labels.shape).float()
        mask = roi_align(Variable(mask, volatile=True), box.unsqueeze(0), crop_size).data
        masks.append(mask)
    return torch.cat(masks, 0)


def aabb_iou(a, b):
    assert a.dim() == 2 and a.shape[1] == 4
    assert a.shape == b.shape

    a, b = a.t(), b.t()

    inter_top = torch.max(a[0], b[0])
    inter_left = torch.max(a[1], b[1])
    inter_bot = torch.min(a[0] + a[2], b[0] + b[2])
    inter_right = torch.min(a[1] + a[3], b[1] + b[3])

    intersection = (inter_right - inter_left) * (inter_bot - inter_top)
    area_a = a[2] * a[3]
    area_b = b[2] * b[3]

    iou = intersection / (area_a + area_b - intersection)
    iou[(inter_right < inter_left) | (inter_bot < inter_top)] = 0
    return iou


# def center_crop(image, centers, border, centers_img_size):
#     """
#     Make several crops of image
#     Args:
#         image: Cropped image. Can have any nuber of channels, only last two are used for cropping.
#         centers: 1d indexes of crop centers.
#         border: tuple of (left-top, right-bottom) offsets from `centers`.
#         centers_img_size: Size of image used to convert centers from 1d to 2d format.
#
#     Returns: Tensor with crops [num crops, ..., crop size, crop size]
#
#     """
#     # get 2d indexes of `centers`
#     centers_y = centers / centers_img_size
#     centers_x = centers - centers_y * centers_img_size
#     centers = torch.stack([centers_y, centers_x], 1).cpu()
#     assert centers.shape == (centers_x.shape[0], 2), centers.shape
#     # crop `image` in +-border range from centers
#     crops = []
#     for c in centers:
#         crop = image[..., c[0] + border[0]: c[0] + border[1], c[1] + border[0]: c[1] + border[1]]
#         crops.append(crop)
#     return torch.stack(crops, 0)


# def mask_to_indexes(mask, stride, border, size):
#     """
#     Convert binary mask to indexes and upscale them
#     Args:
#         mask: Binary mask
#         stride: Stride between mask cells
#         border: Mask padding
#         size: Size of upscaled image
#
#     Returns: 1d indexes
#
#     """
#     # convert `mask` from binary mask to 2d indexes and upscale them from conv-center space to image space
#     idx = mask.nonzero() * stride + border
#     # convert to flat indexes
#     return idx[:, 0] * size + idx[:, 1]


# # custom weights initialization called on netG and netD
# def weights_init(m):
#     if isinstance(m, _ConvNd) or isinstance(m, nn.Linear):
#         m.weight.data.normal_(0.0, 0.02)
#         if m.bias is not None:
#             m.bias.data.fill_(0)
#     elif isinstance(m, _BatchNorm):
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)
#
#
# def calc_gradient_penalty(netD, real_data, fake_data):
#     LAMBDA = 2
#     alpha = torch.rand(real_data.shape[0], 1, 1, 1)
#     # alpha = alpha.expand(real_data.shape)
#     alpha = alpha.cuda()
#
#     interpolates = alpha * real_data + ((1 - alpha) * fake_data)
#     interpolates = interpolates.cuda()
#     interpolates = Variable(interpolates, requires_grad=True)
#
#     disc_interpolates, _ = netD(interpolates)
#
#     gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
#                               grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
#                               create_graph=True, retain_graph=True, only_inputs=True)[0]
#
#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
#     return gradient_penalty
#
#
# class GanD(nn.Module):
#     def __init__(self, nc=3, nf=128):
#         super().__init__()
#         self.head = nn.Sequential(
#             # input is (nc) x 32 x 32
#             nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
#             nn.ReLU(inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(nf * 2),
#             nn.ReLU(inplace=True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
#         )
#         self.tail = nn.Sequential(
#             nn.BatchNorm2d(nf * 4),
#             nn.ReLU(inplace=True),
#             # state size. (ndf*8) x 4 x 4
#             nn.Conv2d(nf * 4, 1, 4, 1, 0, bias=False),
#         )
#
#     def forward(self, input):
#         features = self.head(input)
#         output = self.tail(features)
#         return output.view(-1), features