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
from .feature_pyramid_network import FPN, MaskHead
from .iou import threshold_iou, iou
from .losses import dice_loss, soft_dice_loss, clipped_mse_loss
from .unet import UNet
from .ms_d_net import MSDNet
import numpy as np
import numpy.random as rng
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.batchnorm import _BatchNorm


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


def train_preprocessor_rpn(train_data, epochs=15, pretrain_epochs=7, saved_model=None, return_predictions_at_epoch=None):
    samples_per_image = 256

    dataset = NucleiDataset(train_data)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=4, pin_memory=True)

    model_gen = FPN(1).cuda() if saved_model is None else saved_model.cuda()
    model_gen.freeze_pretrained_layers(False)
    # model_disc = GanD(4).cuda()

    # model_gen.apply(weights_init)
    # model_disc.apply(weights_init)

    # optimizer = torch.optim.SGD(get_param_groups(model), lr=0.05, momentum=0.9, weight_decay=5e-4)
    optimizer_gen = GAdam(get_param_groups(model_gen), lr=2e-4, betas=(0.9, 0.999), avg_sq_mode='weight',
                          amsgrad=False, nesterov=0.5, weight_decay=5e-4, norm_weight_decay=False)
    # optimizer_disc = GAdam(get_param_groups(model_disc), lr=1e-4, betas=(0.9, 0.999), avg_sq_mode='tensor',
    #                       amsgrad=False, nesterov=0.5, weight_decay=1e-4, norm_weight_decay=False)

    scheduler_gen = CosineAnnealingRestartLR(optimizer_gen, len(dataloader), 2)
    # scheduler_disc = CosineAnnealingRestartLR(optimizer_disc, len(dataloader), 2)

    pad = dataloader.dataset.padding
    best_model = model_gen
    best_score = -math.inf

    # one = Variable(torch.cuda.FloatTensor([0.95]))
    # zero = Variable(torch.cuda.FloatTensor([0.05]))

    sys.stdout.flush()

    for epoch in range(epochs):
        with tqdm(dataloader) as pbar:
            model_gen.freeze_pretrained_layers(epoch < pretrain_epochs)
            score_fscore_ma, t_iou_ma, f_iou_ma = 0, 0, 0

            batch_masks = np.zeros(len(model_gen.mask_pixel_sizes))

            for i, data in enumerate(pbar):
                img, labels, sdf = [x.cuda() for x in data]
                x_train = torch.autograd.Variable(img)

                optimizer_gen.zero_grad()

                model_out = model_gen(x_train, pad)
                train_pairs = get_train_pairs(
                    labels, sdf, img[:, :, pad:-pad, pad:-pad], model_out, model_gen.mask_pixel_sizes, samples_per_image)

                if train_pairs is None:
                    continue

                pred_masks, target_masks, pred_scores, target_scores, img_crops, layer_idx = train_pairs

                pred_masks = model_gen.predict_masks(pred_masks)

                bm_idx, bm_count = np.unique(layer_idx, return_counts=True)
                batch_masks[bm_idx] += bm_count

                if return_predictions_at_epoch is not None and return_predictions_at_epoch == epoch:
                    return pred_masks.data.cpu(), target_masks.cpu(), img_crops.cpu(), x_train.data.cpu(), labels.cpu(), best_model

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

                mask_loss = binary_focal_loss_with_logits(pred_masks, target_masks)
                score_loss = binary_focal_loss_with_logits(pred_scores, target_scores)
                loss = mask_loss + 0.1 * score_loss

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

                bc = 1 - 0.99 ** (i + 1)
                score_fscore_ma = 0.99 * score_fscore_ma + 0.01 * score_fscore
                f_iou_ma = 0.99 * f_iou_ma + 0.01 * f_iou.mean()
                t_iou_ma = 0.99 * t_iou_ma + 0.01 * t_iou.mean()
                pbar.set_postfix(E=epoch, SF=score_fscore_ma / bc, IoU=f_iou_ma / bc, IoU_T=t_iou_ma / bc,
                                 MPS=np.round(batch_masks / (i + 1) / img.shape[0], 2), refresh=False)

            score = t_iou_ma
            if t_iou_ma > best_score:
                best_score = score
                best_model = copy.deepcopy(model_gen)
    return best_model


def get_train_pairs(
        labels, sdf, img, net_out, pixel_sizes, samples_count,
        neg_to_pos_ratio=3, pos_sdf_threshold=0.2, neg_sdf_threshold=-0.3,
        pos_size_limits=(0.4, 0.85), neg_size_limits=(0.2, 1.2)):
    outputs = []
    for sample_idx in range(labels.shape[0]):
        net_out_sample = [(m[sample_idx], s[sample_idx, 0]) for m, s in net_out]
        o = get_train_pairs_single(
            labels[sample_idx, 0], sdf[sample_idx, 0], img[sample_idx], net_out_sample,
            pixel_sizes, samples_count, neg_to_pos_ratio, pos_sdf_threshold, neg_sdf_threshold,
            pos_size_limits, neg_size_limits
        )
        outputs.extend(o)

    if len(outputs) == 0:
        return None

    outputs = list(zip(*outputs))
    outputs, layer_idx = outputs[:-1], np.concatenate(outputs[-1])
    pred_masks, target_masks, pred_scores, target_scores, img_crops = [torch.cat(o, 0) for o in outputs]
    return pred_masks, target_masks.unsqueeze(1).float(), pred_scores, target_scores, img_crops, layer_idx


def get_train_pairs_single(labels, sdf, img, net_out, pixel_sizes, samples_count,
                           neg_to_pos_ratio, pos_sdf_threshold, neg_sdf_threshold,
                           pos_size_limits, neg_size_limits):
    box_mask = get_object_boxes(labels)
    resampled_layers = resample_data(labels, sdf, box_mask, img, pixel_sizes)
    outputs = []
    for layer_idx, layer_data in enumerate(zip(net_out, resampled_layers)):
        (out_masks, out_scores), (res_labels, res_sdf, res_boxes, res_img) = layer_data

        num_samples_left = samples_count - len(outputs)
        num_layers_left = len(net_out) - layer_idx
        num_layer_total_samples = round(num_samples_left / num_layers_left)
        num_layer_pos_samples = math.ceil(num_layer_total_samples / (neg_to_pos_ratio + 1))

        o = generate_samples_for_layer(
            out_masks, out_scores, res_labels, res_sdf, res_boxes, res_img,
            num_layer_pos_samples, neg_to_pos_ratio,
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
    mask = mask.view(ds, factor, ds, factor).permute(0, 2, 1, 3).float()
    zero_mask = (mask.view(ds, ds, -1).mean(-1) < 0.5).view(ds, ds, 1, 1).expand_as(mask)
    # inverting values of mask cells where max() should select zeros
    mask[zero_mask] *= -1
    # [-1, ds, ds, factor, factor]
    gx = x.view(-1, ds, factor, ds, factor).permute(0, 1, 3, 2, 4).contiguous()
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


def generate_samples_for_layer(out_masks, out_scores, labels, sdf, obj_boxes, img,
                               max_pos_samples_count, neg_to_pos_ratio,
                               pos_sdf_threshold, neg_sdf_threshold,
                               pos_size_limits, neg_size_limits):
    # border and stride for converting between image space and conv-center space
    border = FPN.mask_size // 2
    stride = FPN.mask_size // FPN.mask_kernel_size

    def upscaled_indexes(mask, max_count):
        # convert `mask` from binary mask to 2d indexes and upscale them from conv-center space to image space
        idx = mask.nonzero() * stride + border
        # shuffle indexes and select no more than `max_count`
        perm = torch.randperm(len(idx))[:max_count].type_as(idx)
        idx = idx[perm]
        # convert to flat indexes
        idx = idx[:, 0] * labels.shape[-1] + idx[:, 1]
        return idx, perm

    def center_crop(image, centers, border=(-border, border), raw_size=labels.shape[-1]):
        # get 2d indexes of `centers`
        centers_y = centers / raw_size
        centers_x = centers - centers_y * raw_size
        centers = torch.stack([centers_y, centers_x], 1).cpu()
        assert centers.shape == (centers_x.shape[0], 2), centers.shape
        # crop `image` in +-border range from centers
        crops = []
        for c in centers:
            crop = image[..., c[0] + border[0]: c[0] + border[1], c[1] + border[0]: c[1] + border[1]]
            crops.append(crop)
        return torch.stack(crops, 0)

    # slice to select values from image at conv center locations
    mask_centers_slice = (
        slice(border, -border + 1, stride),
        slice(border, -border + 1, stride))
    # [fs, fs] - sdf at conv centers
    sdf_fs = sdf[mask_centers_slice]
    # [4, fs, fs] - obj boxes at conv centers
    obj_boxes_fs = obj_boxes[(slice(None), *mask_centers_slice)]
    # [fs, fs] - obj size at conv centers; scaled to [0, 1] range, where 1 is `FPN.mask_size`
    size_fs = torch.max(obj_boxes_fs[2], obj_boxes_fs[3]) / FPN.mask_size
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
    biggest_offsets = biggest_offsets.div(FPN.mask_size).max(0)[0]

    assert sdf_fs.shape == 2 * (out_masks.shape[1] - MaskHead.conv_size + 1,), (sdf_fs.shape, out_masks.shape)

    pos_centers_fmap = (biggest_offsets < pos_size_limits[1] / 2) & \
                       (size_fs > pos_size_limits[0]) & \
                       (sdf_fs > pos_sdf_threshold)
    neg_centers_fmap = (biggest_offsets > neg_size_limits[1] / 2) | \
                       (size_fs < neg_size_limits[0]) | \
                       (sdf_fs < neg_sdf_threshold)

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
    # pred_masks = out_masks.view(-1, FPN.mask_size, FPN.mask_size).index_select(0, Variable(pred_pos_scores_idx))
    # [num masks, conv channels, conv size, conv size]
    pred_masks = center_crop(out_masks, pred_pos_scores_idx, (0, MaskHead.conv_size), pos_centers_fmap.shape[0])
    img_crops = center_crop(img, pos_centers)

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