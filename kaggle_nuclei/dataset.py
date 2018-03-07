import torch
import torchvision.transforms as tsf
from torch.utils.data import Dataset
import random
import sys
import numpy as np
import torchsample.transforms as tst
from .transforms import RandomCrop, Pad, RandomAffineCrop
import random
import sys

import numpy as np
import torch
import torchsample.transforms as tst
import torchvision.transforms as tsf
from torch.utils.data import Dataset

from .transforms import RandomCrop, Pad

bad_ids = {
    '19f0653c33982a416feed56e5d1ce6849fd83314fd19dfa1c5b23c6b66e9868a', # very many mistakes
    '12aeefb1b522b283819b12e4cfaf6b13c1264c0aadac3412b4edd2ace304cb40', # very many mistakes
    '7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80', # worst
    '193ffaa5272d5c421ae02130a64d98ad120ec70e4ed97a72cdcd4801ce93b066', # big white thing at side
    'b1eb0123fe2d8c825694b193efb7b923d95effac9558ee4eaf3116374c2c94fe', # many mistakes
    'adc315bd40d699fd4e4effbcce81cd7162851007f485d754ad3b0472f73a86df', # many mistakes
}


train_size = 256
train_pad = 64
resnet_norm_mean = [0.5, 0.5, 0.5]
resnet_norm_std = [0.5, 0.5, 0.5]


class NucleiDataset(Dataset):
    def __init__(self, data, supersample=1):
        self.has_mask = 'mask_compressed' in data[0]
        self.padding = train_pad
        self.supersample = supersample
        self.supersample_indexes = None
        self._transforming_object_size = False

        if self.has_mask:
            self.datas = [d for d in data if d['name'] not in bad_ids]
        else:
            self.datas = data

        crop_conf = dict(
            size=train_size + train_pad * 2, padding=train_pad, rotation={0, 90},
            scale=(0.5, 2), horizontal_flip=True, vertical_flip=True)

        self.source_transform = tsf.Compose([
            RandomAffineCrop(pad_mode='median', **crop_conf),
            tsf.Normalize(mean=resnet_norm_mean, std=resnet_norm_std),
        ])
        self.target_transform = tsf.Compose([
            RandomAffineCrop(pad_mode='minimum', callback=self.target_transform_callback, **crop_conf),
        ])

    def target_transform_callback(self, arr, crop_y, crop_x, rotation, scale, hflip, vflip, affine_matrix):
        if not self._transforming_object_size:
            return
        arr *= scale

    def __getitem__(self, index):
        if self.supersample != 1:
            if self.supersample_indexes is None or index + 1 == len(self):
                self.supersample_indexes = np.random.randint(len(self.datas), size=len(self))
            index = self.supersample_indexes[index]

        data = self.datas[index]
        img = data['img'].float() / 255

        rstate = random.getstate()
        tseed = random.randrange(sys.maxsize)
        torch.manual_seed(tseed)
        random.setstate(rstate)
        img = self.source_transform(img)

        if self.has_mask:
            pad = self.padding
            mask = data['mask_compressed'].float().unsqueeze(0)
            sdf = data['sdf_compressed'].float().div(127.5).sub(1).unsqueeze(0)
            obj_size = data['info_mask'][2:].float()

            torch.manual_seed(tseed)
            random.setstate(rstate)
            mask = self.target_transform(mask).round().long()

            torch.manual_seed(tseed)
            random.setstate(rstate)
            sdf = self.target_transform(sdf)

            torch.manual_seed(tseed)
            random.setstate(rstate)
            self._transforming_object_size = True
            obj_size = self.target_transform(obj_size)
            self._transforming_object_size = False

            unpad = (slice(None), slice(pad, -pad), slice(pad, -pad))
            return img, mask[unpad], sdf[unpad], obj_size[unpad]
        else:
            return img

    def __len__(self):
        return len(self.datas) * self.supersample
