import random
import sys

import numpy as np
import torch
import torch.utils.data
import torchsample.transforms as tst
import torchvision.transforms as tsf
from torch.utils.data import Dataset

from .dataset import pad, size
from .dataset import resnet_norm_mean, resnet_norm_std
from .transforms import RandomCrop, Pad, RandomAffineCrop


class PostprocessingDataset(Dataset):
    def __init__(self, data, preds):
        self.photo_transform = tsf.Compose([
            RandomAffineCrop(size + pad * 2, padding=size // 2, rotation=(-180, 180), scale=(0.25, 4), pad_mode='reflect'),
            tst.RandomFlip(True, True),
            tsf.Normalize(mean=resnet_norm_mean, std=resnet_norm_std),
        ])
        self.mask_transform = tsf.Compose([
            RandomAffineCrop(size + pad * 2, padding=size // 2, rotation=(-180, 180), scale=(0.25, 4), pad_mode='reflect'),
            tst.RandomFlip(True, True),
        ])
        self.padding = pad
        self.data = data
        self.preds = preds

    def __getitem__(self, index):
        data = self.data[index]
        pred = self.preds[index]
        img = data['img'].float() / 255
        mask = data['mask'].unsqueeze(0).float()
        pad = self.padding
        pred = torch.unbind(torch.from_numpy(pred).float(), 2)
        pred_mask, pred_sdf, pred_cont = [p.contiguous().unsqueeze(0) for p in pred]

        rstate = random.getstate()
        tseed = random.randrange(sys.maxsize)
        torch.manual_seed(tseed)
        random.setstate(rstate)
        img = self.photo_transform(img)

        torch.manual_seed(tseed)
        random.setstate(rstate)
        mask = self.mask_transform(mask).round().long()

        torch.manual_seed(tseed)
        random.setstate(rstate)
        pred_mask = self.mask_transform(pred_mask)

        torch.manual_seed(tseed)
        random.setstate(rstate)
        pred_sdf = self.mask_transform(pred_sdf)

        torch.manual_seed(tseed)
        random.setstate(rstate)
        pred_cont = self.mask_transform(pred_cont)

        return img, pred_mask, pred_sdf, pred_cont, mask[:, pad:-pad, pad:-pad]

    def __len__(self):
        return len(self.data)