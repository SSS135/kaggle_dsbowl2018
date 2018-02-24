from .unet import UNet
import torch
from optfn.param_groups_getter import get_param_groups
from optfn.cosine_annealing import CosineAnnealingRestartLR
from tqdm import tqdm
import sys
import torch.nn.functional as F
from .dice_loss import soft_dice_loss
from .iou import threshold_iou, iou
import math
import copy


def train_unet(dataloader, epochs=65):
    model = UNet(3, 2).cuda()
    optimizer = torch.optim.SGD(get_param_groups(model), lr=0.03, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingRestartLR(optimizer, len(dataloader), 2)
    pad = dataloader.dataset.padding
    best_model = model
    best_score = -math.inf

    sys.stdout.flush()

    for epoch in range(epochs):
        with tqdm(dataloader) as pbar:
            t_iou_ma, d_iou_ma = 0, 0
            for i, (img, mask, sdf, name) in enumerate(pbar):
                x_train = torch.autograd.Variable(img.cuda())
                sdf_train = torch.autograd.Variable(sdf.cuda())
                mask_train = torch.autograd.Variable(mask.cuda())
                optimizer.zero_grad()
                out_mask, out_sdf = model(x_train)[:, :, pad:-pad, pad:-pad].chunk(2, 1)
                out_mask, out_sdf = F.sigmoid(out_mask), out_sdf.contiguous()
                mask_target = (mask_train > 0).float().clamp(min=0.05, max=0.95)
                loss = soft_dice_loss(out_mask, mask_target) + F.mse_loss(out_sdf, sdf_train)
                loss.backward()
                optimizer.step()
                scheduler.step()

                t_iou = threshold_iou(iou(out_mask, mask_train))
                d_iou = threshold_iou(iou(out_sdf, mask_train, bin_threshold=0))
                bc = 1 - 0.999 ** (i + 1)
                d_iou_ma = 0.999 * d_iou_ma + 0.001 * d_iou
                t_iou_ma = 0.999 * t_iou_ma + 0.001 * t_iou
                pbar.set_postfix(epoch=epoch, T_IoU=t_iou_ma / bc, D_IOU=d_iou_ma / bc, refresh=False)

            score = t_iou_ma + d_iou_ma
            if score > best_score:
                best_score = score
                best_model = copy.deepcopy(model)
    return best_model
