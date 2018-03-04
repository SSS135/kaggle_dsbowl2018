import copy
import math
import sys

import torch
import torch.nn.functional as F
import torch.utils.data
from optfn.cosine_annealing import CosineAnnealingRestartLR
from optfn.gadam import GAdam
from optfn.param_groups_getter import get_param_groups
from torch.autograd import Variable
from tqdm import tqdm
from torch import nn

from .dataset import NucleiDataset
from .feature_pyramid_network import FPN
from .iou import threshold_iou, iou
from .losses import dice_loss, soft_dice_loss, clipped_mse_loss
from .unet import UNet
from .ms_d_net import MSDNet
from scipy import ndimage
import numpy as np
import numpy.random as rng


def train_preprocessor(train_data, epochs=15, pretrain_epochs=7, hard_example_subsample=1):
    dataset = NucleiDataset(train_data, supersample=1)
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=4 * hard_example_subsample, pin_memory=True)

    model = FPN(3).cuda()
    optimizer = GAdam(get_param_groups(model), lr=0.0001, nesterov=0.0, weight_decay=5e-4,
                      avg_sq_mode='tensor', amsgrad=False)

    scheduler = CosineAnnealingRestartLR(optimizer, len(dataloader), 2)
    pad = dataloader.dataset.padding
    best_model = model
    best_score = -math.inf

    sys.stdout.flush()

    for epoch in range(epochs):
        with tqdm(dataloader) as pbar:
            model.freeze_pretrained_layers(epoch < pretrain_epochs)
            t_iou_ma, f_iou_ma, sdf_loss_ma, cont_loss_ma = 0, 0, 0, 0
            for i, (img, mask, sdf) in enumerate(pbar):
                x_train = torch.autograd.Variable(img.cuda())
                sdf_train = torch.autograd.Variable(sdf.cuda())
                mask_train = torch.autograd.Variable(mask.cuda())

                optimizer.zero_grad()
                out_mask, out_sdf, out_cont = model(x_train)[:, :, pad:-pad, pad:-pad].chunk(3, 1)
                out_mask, out_sdf = F.sigmoid(out_mask), out_sdf.contiguous()

                mask_target = (mask_train > 0).float().clamp(min=0.05, max=0.95)
                sdf_loss = clipped_mse_loss(out_sdf, sdf_train, -1, 1, reduce=False).view(x_train.shape[0], -1).mean(-1)
                cont_loss = clipped_mse_loss(out_cont, cont_train, -1, 1, reduce=False).view(x_train.shape[0], -1).mean(-1)
                if hard_example_subsample != 1:
                    keep_count = max(1, x_train.shape[0] // hard_example_subsample)
                    mse_mask_loss = F.mse_loss(out_mask, mask_target, reduce=False).view(x_train.shape[0], -1).mean(-1)
                    sort_loss = sdf_loss + cont_loss + mse_mask_loss
                    keep_idx = sort_loss.sort()[1][-keep_count:]
                    mask_loss = soft_dice_loss(out_mask[keep_idx], mask_target[keep_idx])
                    loss = mask_loss + sdf_loss[keep_idx].mean() + cont_loss[keep_idx].mean()
                else:
                    mask_loss = soft_dice_loss(out_mask, mask_target)
                    loss = mask_loss + sdf_loss.mean() + cont_loss.mean()
                loss.backward()
                optimizer.step()
                scheduler.step()

                f_iou = iou(out_mask, mask_train)
                t_iou = threshold_iou(f_iou)
                bc = 1 - 0.99 ** (i + 1)
                sdf_loss_ma = 0.99 * sdf_loss_ma + 0.01 * sdf_loss.data.mean()
                cont_loss_ma = 0.99 * cont_loss_ma + 0.01 * cont_loss.data.mean()
                t_iou_ma = 0.99 * t_iou_ma + 0.01 * t_iou
                f_iou_ma = 0.99 * f_iou_ma + 0.01 * f_iou
                pbar.set_postfix(epoch=epoch, T_IoU=t_iou_ma / bc, SDF=sdf_loss_ma / bc,
                                 Cont=cont_loss_ma / bc, IOU=f_iou_ma / bc, refresh=False)

            if t_iou_ma > best_score:
                best_score = t_iou_ma
                best_model = copy.deepcopy(model)
    return best_model


def generate_region_samples(labels_cpu, sdf_cpu, labels_cuda, sdf_cuda, samples_per_image, sample_size,
                            neg_to_pos_ratio=3, pos_sdf_threshold=0.4, neg_sdf_threshold=0.2):
    labels_np = labels_cpu.numpy().squeeze(1)
    sdf_np = sdf_cpu.numpy().squeeze(1)

    pad = sample_size // 2
    sdf_center = sdf_np[:, pad:-pad, pad: -pad]
    sdf_np = np.full_like(sdf_np, -1, dtype=sdf_np.dtype)
    sdf_np[:, pad:-pad, pad: -pad] = sdf_center

    samples = [generate_region_samples_single(*x, samples_per_image, sample_size, pos_sdf_threshold, neg_sdf_threshold)
               for x in zip(labels_np, sdf_np, labels_cuda, sdf_cuda)]


def generate_region_samples_single(labels_np, sdf_np, labels_cuda, sdf_cuda, samples_count, sample_size,
                                   neg_to_pos_ratio, pos_sdf_threshold, neg_sdf_threshold):
    assert samples_count % (neg_to_pos_ratio + 1) == 0
    pos_samples_count = samples_count // (neg_to_pos_ratio + 1)
    pos_sample_rect_pos, obj_sizes, label_nums = generate_positive_samples_info(
        labels_np, sdf_np, pos_samples_count, sample_size, pos_sdf_threshold)
    pos_samples_count = len(pos_sample_rect_pos)
    neg_samples_count = samples_count - pos_samples_count
    neg_sample_rect_pos = generate_negative_samples_info(sdf_np, neg_samples_count, sample_size, neg_sdf_threshold)


def generate_positive_samples_info(labels, sdf, samples_count, sample_size, sdf_threshold):
    indexes = np.nonzero(sdf > sdf_threshold)
    indexes = rng.choice(indexes, min(len(indexes), samples_count), replace=False)
    label_nums = labels[indexes]
    labels = labels * np.isin(labels, label_nums)
    objs = [x for x in ndimage.find_objects(labels) if x is not None]
    obj_sizes = [max(y.end - y.start, x.end - x.start) / sample_size for (y, x, _) in objs]
    sample_rect_pos = [(cy - sample_size // 2, cx - sample_size // 2) for cy, cx in np.transpose(indexes)]
    return sample_rect_pos, obj_sizes, label_nums


def generate_negative_samples_info(sdf, samples_count, sample_size, sdf_threshold):
    indexes = np.nonzero(sdf < sdf_threshold)
    indexes = rng.choice(indexes, samples_count, replace=False)
    sample_rect_pos = [(cy - sample_size // 2, cx - sample_size // 2) for cy, cx in np.transpose(indexes)]
    return sample_rect_pos
