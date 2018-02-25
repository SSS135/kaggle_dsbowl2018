import copy
import math
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchsample.transforms as tst
import torchvision.transforms as tsf
from optfn.cosine_annealing import CosineAnnealingRestartLR
from optfn.param_groups_getter import get_param_groups
from torch.autograd import Variable
from torch.utils.data import Dataset
from tqdm import tqdm

from .dataset import make_train_dataset
from .dataset import pad, size
from .iou import threshold_iou, iou
from .losses import dice_loss
from .transforms import RandomCrop, Pad
from .unet import UNet
from .dataset import resnet_norm_mean, resnet_norm_std


class PostprocessingDataset(Dataset):
    def __init__(self, data, preds):
        self.photo_transform = tsf.Compose([
            # MeanNormalize(),
            tsf.Normalize(mean=resnet_norm_mean, std=resnet_norm_std),
            Pad(2 * (pad,), mode='reflect'),
            # *([tst.RandomAffine(rotation_range=180, zoom_range=(0.5, 2))] if affine else []),
            RandomCrop(2 * (size + pad * 2,)),
            tst.RandomFlip(True, True),
        ])
        self.mask_transform = tsf.Compose([
            Pad(2 * (pad,), mode='reflect'),
            # *([tst.RandomAffine(rotation_range=180, zoom_range=(0.5, 2))] if affine else []),
            RandomCrop(2 * (size + pad * 2,)),
            tst.RandomFlip(True, True),
        ])
        self.padding = pad
        self.data = data
        self.preds = preds

    def __getitem__(self, index):
        data = self.data[index]
        img = data['img'].float() / 255

        rstate = random.getstate()
        tseed = random.randrange(sys.maxsize)
        torch.manual_seed(tseed)
        random.setstate(rstate)
        img = self.source_transform(img)

        pad = self.padding
        mask = data['mask'].float().unsqueeze(0)
        sdf = data['distance_field'].unsqueeze(0)

        torch.manual_seed(tseed)
        random.setstate(rstate)
        mask = self.target_transform(mask).round().long()
        torch.manual_seed(tseed)
        random.setstate(rstate)
        sdf = self.target_transform(sdf)
        return img, mask[:, pad:-pad, pad:-pad], sdf[:, pad:-pad, pad:-pad]

    def __len__(self):
        return len(self.data) * self.supersample


def train_rl_unet(train_data, train_pred, epochs=7, hard_example_subsample=1, affine_augmentation=False):
    dataset = make_train_dataset(train_data, affine=affine_augmentation, supersample=hard_example_subsample)
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=4 * hard_example_subsample, pin_memory=True)
    model = UNet(3, 3).cuda()
    optimizer = torch.optim.SGD(get_param_groups(model), lr=0.03, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingRestartLR(optimizer, len(dataloader), 2)
    pad = dataloader.dataset.padding
    best_model = model
    best_score = -math.inf

    sys.stdout.flush()

    for epoch in range(epochs):
        with tqdm(dataloader) as pbar:
            t_iou_ma, sdf_loss_ma, cont_loss_ma = 0, 0, 0
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
                sdf_loss = F.mse_loss(out_sdf, sdf_train, reduce=False).view(x_train.shape[0], -1).mean(-1)
                cont_loss = F.mse_loss(out_cont, cont_train, reduce=False).view(x_train.shape[0], -1).mean(-1)
                if hard_example_subsample != 1:
                    keep_count = max(1, x_train.shape[0] // hard_example_subsample)
                    mse_mask_loss = F.mse_loss(out_mask, mask_target, reduce=False).view(x_train.shape[0], -1).mean(-1)
                    sort_loss = sdf_loss + cont_loss + mse_mask_loss
                    keep_idx = sort_loss.sort()[1][-keep_count:]
                    mask_loss = dice_loss(out_mask[keep_idx], mask_target[keep_idx])
                    loss = mask_loss + sdf_loss[keep_idx].mean() + cont_loss[keep_idx].mean()
                else:
                    mask_loss = dice_loss(out_mask, mask_target)
                    loss = mask_loss + sdf_loss.mean() + cont_loss.mean()
                loss.backward()
                optimizer.step()
                scheduler.step()

                t_iou = threshold_iou(iou(out_mask, mask_train))
                bc = 1 - 0.99 ** (i + 1)
                sdf_loss_ma = 0.99 * sdf_loss_ma + 0.01 * sdf_loss.data.mean()
                cont_loss_ma = 0.99 * cont_loss_ma + 0.01 * cont_loss.data.mean()
                t_iou_ma = 0.99 * t_iou_ma + 0.01 * t_iou
                pbar.set_postfix(epoch=epoch, T_IoU=t_iou_ma / bc, SDF=sdf_loss_ma / bc, Cont=cont_loss_ma / bc, refresh=False)

            if t_iou_ma > best_score:
                best_score = t_iou_ma
                best_model = copy.deepcopy(model)
    return best_model