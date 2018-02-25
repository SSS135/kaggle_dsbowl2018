import torch
import torchvision.transforms as tsf
from torch.utils.data import Dataset
import random
import sys
import numpy as np
import torchsample.transforms as tst
from .transforms import RandomCrop, Pad
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


size = 128
pad = 32
resnet_norm_mean = [0.485, 0.456, 0.406]
resnet_norm_std = [0.229, 0.224, 0.225]


class NucleiDataset(Dataset):
    def __init__(self, data, source_transform, target_transform, supersample=1):
        self.has_mask = 'mask' in data[0]
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.padding = pad
        self.supersample = supersample
        self.supersample_indexes = None
        if self.has_mask:
            self.datas = [d for d in data if d['name'] not in bad_ids]
        else:
            self.datas = data

    def __getitem__(self, index):
        if self.supersample != 1:
            if self.supersample_indexes is None or index + 1 == len(self):
                self.supersample_indexes = np.random.randint(len(self.datas), size=len(self))
            index = self.supersample_indexes[index]

        data = self.datas[index]
        img = data['img'].float() / 255
        name = data['name']

        rstate = random.getstate()
        tseed = random.randrange(sys.maxsize)
        torch.manual_seed(tseed)
        random.setstate(rstate)
        img = self.source_transform(img)

        if self.has_mask:
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
        else:
            return img

    def __len__(self):
        return len(self.datas) * self.supersample


def make_train_dataset(train_data, affine=False, supersample=1):
    s_transf = tsf.Compose([
        # MeanNormalize(),
        tsf.Normalize(mean=resnet_norm_mean, std=resnet_norm_std),
        Pad(2 * (pad,), mode='reflect'),
        *([tst.RandomAffine(rotation_range=180, zoom_range=(0.5, 2))] if affine else []),
        RandomCrop(2 * (size + pad * 2,)),
        tst.RandomFlip(True, True),
    ])
    t_transf = tsf.Compose([
        Pad(2 * (pad,), mode='reflect'),
        *([tst.RandomAffine(rotation_range=180, zoom_range=(0.5, 2))] if affine else []),
        RandomCrop(2 * (size + pad * 2,)),
        tst.RandomFlip(True, True),
    ])
    return NucleiDataset(train_data, s_transf, t_transf, supersample=supersample)


# def make_test_dataset(test_data):
#     s_transf = tsf.Compose([
#         MeanNormalize(),
#         Pad(2 * (pad,), mode='reflect'),
#         RandomCrop(2 * (size + pad * 2,)),
#     ])
#     return NucleiDataset(test_data, s_transf, None)
