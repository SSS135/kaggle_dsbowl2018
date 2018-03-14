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

from .dataset import NucleiDataset
from .feature_pyramid_network import FPN
from .iou import threshold_iou, iou
from .losses import dice_loss, soft_dice_loss, clipped_mse_loss
from .unet import UNet
from .ms_d_net import MSDNet
import numpy as np
import numpy.random as rng
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.batchnorm import _BatchNorm
import time
from collections import OrderedDict


def binary_cross_entropy_with_logits(x, z, reduce=True):
    bce = x.clamp(min=0) - x * z + x.abs().neg().exp().add(1).log()
    return bce.mean() if reduce else bce


def binary_focal_loss_with_logits(pred, target, lam=2, reduce=True):
    ce = binary_cross_entropy_with_logits(pred, target, False)
    loss = (target - F.sigmoid(pred)).abs().pow(lam) * ce
    return loss.mean() if reduce else loss


def mse_focal_loss(pred, target, lam=2, reduce=True):
    mse = F.mse_loss(pred, target, reduce=False)
    loss = (pred - target).clamp(-1, 1).abs().pow(lam) * mse
    return loss.mean() if reduce else loss


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


def train_preprocessor_rpn(train_data, epochs=15, pretrain_epochs=7, saved_model=None, return_predictions_at_epoch=None):
    dataset = NucleiDataset(train_data)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=4, pin_memory=True)

    model_gen = FPN(1, 3).cuda() if saved_model is None else saved_model.cuda()
    batch_to_instance_norm(model_gen)
    model_gen.freeze_pretrained_layers(False)
    # model_disc = GanD(4).cuda()

    # model_gen.apply(weights_init)
    # model_disc.apply(weights_init)

    # optimizer = torch.optim.SGD(get_param_groups(model), lr=0.05, momentum=0.9, weight_decay=5e-4)
    optimizer_gen = GAdam(get_param_groups(model_gen), lr=0.5e-4, betas=(0.9, 0.999), avg_sq_mode='tensor',
                          amsgrad=False, nesterov=0.9, weight_decay=1e-4)
    # optimizer_disc = GAdam(get_param_groups(model_disc), lr=1e-4, betas=(0.9, 0.999), avg_sq_mode='tensor',
    #                       amsgrad=False, nesterov=0.5, weight_decay=1e-4, norm_weight_decay=False)

    scheduler_gen = CosineAnnealingRestartParam(optimizer_gen, len(dataloader), 2, param_name='nesterov')
    # scheduler_disc = CosineAnnealingRestartLR(optimizer_disc, len(dataloader), 2)

    pad = dataloader.dataset.padding
    best_state_dict = copy_state_dict(model_gen)
    best_score = -math.inf

    # one = Variable(torch.cuda.FloatTensor([0.95]))
    # zero = Variable(torch.cuda.FloatTensor([0.05]))

    sys.stdout.flush()

    for epoch in range(epochs):
        with tqdm(dataloader) as pbar:
            optimizer_gen.zero_grad()
            model_gen.freeze_pretrained_layers(epoch < pretrain_epochs)
            score_fscore_sum, t_iou_sum, f_iou_sum = 0, 0, 0

            batch_masks = np.zeros(len(model_gen.mask_pixel_sizes))

            for batch_idx, data in enumerate(pbar):
                img, labels, sdf = [x.cuda() for x in data]
                x_train = Variable(img)
                sdf_train = Variable(sdf)
                mask_train = Variable((labels > 0).float().clamp(0.05, 0.95))

                cont_train = 1 - sdf_train.data ** 2
                cont_train = (cont_train.clamp(0.9, 1) - 0.9) * 20 - 1
                cont_train = Variable(cont_train)

                optimizer_gen.zero_grad()

                model_out_layers, model_out_img = model_gen(x_train, pad)
                train_pairs = get_train_pairs(
                    model_gen, labels, sdf, img[:, :, pad:-pad, pad:-pad], model_out_layers, model_gen.mask_pixel_sizes)

                if train_pairs is None:
                    continue

                pred_masks, target_masks, pred_scores, target_scores, img_crops, layer_idx = train_pairs

                pred_masks = model_gen.predict_masks(pred_masks)

                bm_idx, bm_count = np.unique(layer_idx, return_counts=True)
                batch_masks[bm_idx] += bm_count

                if return_predictions_at_epoch is not None and return_predictions_at_epoch == epoch:
                    return pred_masks.data.cpu(), target_masks.cpu(), img_crops.cpu(), x_train.data.cpu(), labels.cpu(), model_gen

                target_masks = Variable(target_masks.clamp(0.05, 0.95))
                target_scores = Variable(target_scores.clamp(0.05, 0.95))
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
                img_mask_out = F.sigmoid(img_mask_out)
                img_mask_loss = soft_dice_loss(img_mask_out, mask_train)
                img_sdf_loss = F.mse_loss(img_sdf_out, sdf_train)
                img_cont_loss = F.mse_loss(img_cont_out, cont_train)

                mask_loss = binary_focal_loss_with_logits(pred_masks, target_masks)
                score_loss = binary_focal_loss_with_logits(pred_scores, target_scores)
                loss = mask_loss + 0.1 * score_loss + mask_loss + (img_mask_loss + img_sdf_loss + img_cont_loss) / 3

                loss.backward()
                optimizer_gen.step()

                scheduler_gen.step()
                # scheduler_disc.step()

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
                pbar.set_postfix(
                    E=epoch, SF=score_fscore_sum / iter,
                    IoU=f_iou_sum / iter, IoU_T=t_iou_sum / iter,
                    MPS=np.round(batch_masks / iter / img.shape[0], 2), refresh=False)

            score = t_iou_sum
            if score > best_score:
                best_score = score
                best_state_dict = copy_state_dict(model_gen)
    model_gen.load_state_dict(best_state_dict)
    return model_gen


def get_train_pairs(
        model, labels, sdf, img, net_out, pixel_sizes,
        pos_sdf_threshold=0.2, neg_sdf_threshold=-0.3,
        pos_size_limits=(0.4, 0.9), neg_size_limits=(0.2, 1.2)):
    outputs = []
    for sample_idx in range(labels.shape[0]):
        net_out_sample = [(m[sample_idx], s[sample_idx, 0]) for m, s in net_out]
        o = get_train_pairs_single(
            model, labels[sample_idx, 0], sdf[sample_idx, 0], img[sample_idx], net_out_sample,
            pixel_sizes, pos_sdf_threshold, neg_sdf_threshold,
            pos_size_limits, neg_size_limits
        )
        outputs.extend(o)

    if len(outputs) == 0:
        return None

    outputs = list(zip(*outputs))
    outputs, layer_idx = outputs[:-1], np.concatenate(outputs[-1])
    pred_masks, target_masks, pred_scores, target_scores, img_crops = [torch.cat(o, 0) for o in outputs]
    return pred_masks, target_masks.unsqueeze(1).float(), pred_scores, target_scores, img_crops, layer_idx


def get_train_pairs_single(model, labels, sdf, img, net_out, pixel_sizes,
                           pos_sdf_threshold, neg_sdf_threshold,
                           pos_size_limits, neg_size_limits):
    box_mask = get_object_boxes(labels)
    resampled_layers = resample_data(labels, sdf, box_mask, img, pixel_sizes)
    outputs = []
    for layer_idx, layer_data in enumerate(zip(net_out, resampled_layers)):
        (out_masks, out_scores), (res_labels, res_sdf, res_boxes, res_img) = layer_data

        o = generate_samples_for_layer(
            model, out_masks, out_scores, res_labels, res_sdf, res_boxes, res_img,
            pos_sdf_threshold, neg_sdf_threshold,
            pos_size_limits, neg_size_limits
        )
        if o is not None:
            outputs.append((*o, o[0].shape[0] * [layer_idx]))
    return outputs


def get_object_boxes(labels):
    # [size, size]
    assert labels.dim() == 2

    count = labels.max()
    if count == 0:
        return torch.zeros(4, *labels.shape).cuda()

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


def center_crop(image, centers, border, centers_img_size):
    """
    Make several crops of image
    Args:
        image: Cropped image. Can have any nuber of channels, only last two are used for cropping.
        centers: 1d indexes of crop centers.
        border: tuple of (left-top, right-bottom) offsets from `centers`.
        centers_img_size: Size of image used to convert centers from 1d to 2d format.

    Returns: Tensor with crops [num crops, ..., crop size, crop size]

    """
    # get 2d indexes of `centers`
    centers_y = centers / centers_img_size
    centers_x = centers - centers_y * centers_img_size
    centers = torch.stack([centers_y, centers_x], 1).cpu()
    assert centers.shape == (centers_x.shape[0], 2), centers.shape
    # crop `image` in +-border range from centers
    crops = []
    for c in centers:
        crop = image[..., c[0] + border[0]: c[0] + border[1], c[1] + border[0]: c[1] + border[1]]
        crops.append(crop)
    return torch.stack(crops, 0)


def mask_to_indexes(mask, stride, border, size):
    """
    Convert binary mask to indexes and upscale them
    Args:
        mask: Binary mask
        stride: Stride between mask cells
        border: Mask padding
        size: Size of upscaled image

    Returns: 1d indexes

    """
    # convert `mask` from binary mask to 2d indexes and upscale them from conv-center space to image space
    idx = mask.nonzero() * stride + border
    # convert to flat indexes
    return idx[:, 0] * size + idx[:, 1]


def generate_samples_for_layer(model, out_masks, out_scores, labels, sdf, obj_boxes, img,
                               pos_sdf_threshold, neg_sdf_threshold,
                               pos_size_limits, neg_size_limits):
    # border and stride for converting between image space and conv-center space
    border = model.mask_size // 2
    stride = model.mask_size // model.mask_kernel_size

    # slice to select values from image at conv center locations
    mask_centers_slice = (
        slice(border, -border + 1, stride),
        slice(border, -border + 1, stride))
    # [fs, fs] - sdf at conv centers
    sdf_fs = sdf[mask_centers_slice]
    # [4, fs, fs] - obj boxes at conv centers
    obj_boxes_fs = obj_boxes[(slice(None), *mask_centers_slice)]
    # [fs, fs] - obj size at conv centers; scaled to [0, 1] range, where 1 is `model.mask_size`
    size_fs = torch.max(obj_boxes_fs[2], obj_boxes_fs[3]) / model.mask_size
    # [fs] - X or Y locations of conv centers in image
    fs_yx_range = torch.arange(border, labels.shape[-1] - border + 1, stride).cuda()
    # [2, fs, fs] - YX grid of conv centers in image space
    fs_img_conv_center_pos = torch.stack([
        fs_yx_range.view(-1, 1).expand(-1, len(fs_yx_range)),
        fs_yx_range.view(1, -1).expand(len(fs_yx_range), -1)
    ], 0)
    # [2, fs, fs] - top left corner offsets from obj centers
    obj_min_offsets = fs_img_conv_center_pos - obj_boxes_fs[:2]
    # [2, fs, fs] - bottom right corner offsets from obj centers
    obj_max_offsets = obj_boxes_fs[2:] - obj_min_offsets
    # [4, fs, fs]
    biggest_offsets = torch.cat([obj_min_offsets, obj_max_offsets], 0)
    # [fs, fs] with [0, 0.5] where 0.5 is half mask offset from center
    biggest_offsets = biggest_offsets.div(model.mask_size).max(0)[0]

    assert sdf_fs.shape == 2 * (out_masks.shape[1] - model.conv_size + 1,), (sdf_fs.shape, out_masks.shape)

    pos_centers_fmap = (biggest_offsets < pos_size_limits[1] / 2) & \
                       (size_fs > pos_size_limits[0]) & \
                       (sdf_fs > pos_sdf_threshold)
    neg_centers_fmap = (biggest_offsets > neg_size_limits[1] / 2) | \
                       (size_fs < neg_size_limits[0]) | \
                       (sdf_fs < neg_sdf_threshold)

    # TODO: allow zero negative centers
    if pos_centers_fmap.sum() == 0 or neg_centers_fmap.sum() == 0:
        return None

    pos_centers = mask_to_indexes(pos_centers_fmap, stride, border, labels.shape[-1])

    pos_centers_fmap_idx = pos_centers_fmap.view(-1).nonzero().squeeze()
    neg_centers_fmap_idx = neg_centers_fmap.view(-1).nonzero().squeeze()
    neg_centers_fmap_idx = neg_centers_fmap_idx[torch.randperm(len(neg_centers_fmap_idx))[:256].cuda()]
    pred_pos_scores = out_scores.take(Variable(pos_centers_fmap_idx))
    pred_neg_scores = out_scores.take(Variable(neg_centers_fmap_idx))

    pred_scores = torch.cat([pred_pos_scores, pred_neg_scores])
    target_scores = out_scores.data.new(pred_scores.shape[0]).fill_(0)
    target_scores[:pred_pos_scores.shape[0]] = 1

    label_crops = center_crop(labels, pos_centers, (-border, border), labels.shape[-1])
    pos_center_label_nums = labels.take(pos_centers)
    target_masks = label_crops == pos_center_label_nums.view(-1, 1, 1)
    # [num masks, conv channels, conv size, conv size]
    pred_masks = center_crop(out_masks, pos_centers_fmap_idx, (0, model.conv_size), pos_centers_fmap.shape[0])
    img_crops = center_crop(img, pos_centers, (-border, border), labels.shape[-1])

    return pred_masks, target_masks, pred_scores, target_scores, img_crops


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