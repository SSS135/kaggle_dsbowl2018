import PIL
import torch
import torchvision.transforms as tsf
from torch.utils.data import Dataset
import random
import sys
import numpy as np
import torchsample.transforms as tst
import torch.nn.functional as F
from .transforms import MeanNormalize, RandomCrop, Pad


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


class NucleiDataset(Dataset):
    def __init__(self, data, source_transform, target_transform):
        self.has_mask = 'mask' in data[0]
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.padding = pad
        if self.has_mask:
            self.datas = [d for d in data if d['name'] not in bad_ids]
        else:
            self.datas = data

    def __getitem__(self, index):
        data = self.datas[index]
        img = data['img'].permute(2, 0, 1).float() / 255
        name = data['name']

        rstate = random.getstate()
        tseed = random.randrange(sys.maxsize)
        torch.manual_seed(tseed)
        random.setstate(rstate)
        img = self.source_transform(img)

        if self.has_mask:
            pad = self.padding
            mask = data['mask'].unsqueeze(0).long()
            sdf = torch.from_numpy(data['distance_field']).float().unsqueeze(0)

            torch.manual_seed(tseed)
            random.setstate(rstate)
            mask = self.target_transform(mask)
            torch.manual_seed(tseed)
            random.setstate(rstate)
            sdf = self.target_transform(sdf)
            return img, mask[:, pad:-pad, pad:-pad], sdf[:, pad:-pad, pad:-pad], name
        else:
            return img, name

    def __len__(self):
        return len(self.datas)


def make_train_dataset(train_data):
    s_transf = tsf.Compose([
        # tsf.ToPILImage(),
        # tsf.Resize((size, size)),
        # tsf.RandomCrop(size),
        # tsf.RandomHorizontalFlip(),
        # tsf.ToTensor(),
        MeanNormalize(),
        Pad(2 * (pad,), mode='reflect'),
        RandomCrop(2 * (size + pad,)),
        tst.RandomFlip(True, True),
        # tsf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    t_transf = tsf.Compose([
        # tsf.ToPILImage(),
        # tsf.Resize((size, size), interpolation=PIL.Image.NEAREST),
        # tsf.RandomCrop(size),
        # tsf.RandomHorizontalFlip(),
        # tsf.ToTensor(),
        Pad(2 * (pad,)),
        RandomCrop(2 * (size + pad,)),
        tst.RandomFlip(True, True),
    ])
    return NucleiDataset(train_data, s_transf, t_transf)


def make_test_dataset(test_data):
    s_transf = tsf.Compose([
        # tsf.ToPILImage(),
        # tsf.Resize((size, size)),
        # tsf.ToTensor(),
        MeanNormalize(),
        Pad(2 * (pad,), mode='reflect'),
        RandomCrop(2 * (size + pad,)),
        # tsf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return NucleiDataset(test_data, s_transf, None)
