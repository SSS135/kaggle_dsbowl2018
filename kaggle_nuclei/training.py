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

from .dataset import make_train_dataset
from .feature_pyramid_network import FPN
from .iou import threshold_iou, iou
from .losses import dice_loss, soft_dice_loss, clipped_mse_loss
from .unet import UNet


def train_preprocessor(train_data, epochs=15, pretrain_epochs=7, hard_example_subsample=1,
                       affine_augmentation=False, resnet=True):
    dataset = make_train_dataset(train_data, affine=affine_augmentation, supersample=hard_example_subsample)
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=4 * hard_example_subsample, pin_memory=True)
    if resnet:
        model = FPN(3).cuda()
        optimizer = GAdam(get_param_groups(model), lr=0.0001, nesterov=0.0, weight_decay=5e-4,
                          avg_sq_mode='tensor', amsgrad=False)
    else:
        model = UNet(3, 3).cuda()
        optimizer = torch.optim.SGD(get_param_groups(model), lr=0.03, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingRestartLR(optimizer, len(dataloader), 2)
    pad = dataloader.dataset.padding
    best_model = model
    best_score = -math.inf

    sys.stdout.flush()

    for epoch in range(epochs):
        with tqdm(dataloader) as pbar:
            if resnet:
                model.freeze_pretrained_layers(epoch < pretrain_epochs)
            t_iou_ma, f_iou_ma, sdf_loss_ma, cont_loss_ma = 0, 0, 0, 0
            for i, (img, mask, sdf) in enumerate(pbar):
                x_train = torch.autograd.Variable(img.cuda())
                sdf_train = torch.autograd.Variable(sdf.cuda())
                mask_train = torch.autograd.Variable(mask.cuda())

                cont_train = 1 - sdf_train.data ** 2
                cont_train = (cont_train.clamp(0.9, 1) - 0.9) * 20 - 1
                cont_train = Variable(cont_train)

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
