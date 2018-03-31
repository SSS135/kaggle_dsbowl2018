import random
import sys

import numpy as np
import torch
import torchvision.transforms as tsf
from torch.utils.data import Dataset

from .settings import train_pad, train_size, resnet_norm_std, resnet_norm_mean, box_padding
from .transforms import RandomAffineCrop


class NucleiDataset(Dataset):
    def __init__(self, data, normalize_scale_by_obj_area=True, normalize_image_sample_freq=True):
        self.has_mask = 'mask_compressed' in data[0]
        self.normalize_scale_by_obj_area = normalize_scale_by_obj_area
        self.normalize_image_sample_freq = normalize_image_sample_freq
        self.data = data
        self.target_obj_area = (58 * 1.333) ** 2
        self.base_scale_range = 1 / 2, 2

        self.index_map = np.arange(len(data))
        if normalize_image_sample_freq:
            assert normalize_scale_by_obj_area
            obj_areas_sq = np.array([s['median_mask_area'] for s in data]) ** 0.5
            img_areas_sq = np.array([np.prod(s['img'].shape[1:]) for s in data]) ** 0.5
            total_samples = 4 * len(data)
            sample_freqs = img_areas_sq / obj_areas_sq
            sample_freqs *= total_samples / sample_freqs.sum()
            sample_freqs = sample_freqs.round().clip(1, 8).astype(np.int32)
            self.index_map = np.repeat(self.index_map, sample_freqs)

        crop_conf = dict(
            size=train_size + train_pad * 2, padding=train_pad, rotation={(0, 360)},
            scale=self.base_scale_range, horizontal_flip=True, vertical_flip=True)

        self.source_random_affine_crop = RandomAffineCrop(pad_mode='median', affine_order=1, **crop_conf)
        self.target_random_affine_crop = RandomAffineCrop(pad_mode='minimum', affine_order=0, **crop_conf)

        self.source_transform = tsf.Compose([
            self.source_random_affine_crop,
            tsf.ToPILImage(),
            tsf.ColorJitter(0.3, 0.3, 0.3, 0.3),
            tsf.ToTensor(),
            tsf.Normalize(mean=resnet_norm_mean, std=resnet_norm_std),
        ])
        self.target_transform = tsf.Compose([
            self.target_random_affine_crop,
        ])

    def __getitem__(self, index):
        index = self.index_map[index]
        data = self.data[index]

        if self.normalize_scale_by_obj_area:
            med_area_sq = data['median_mask_area'] ** 0.5
            target_area_sq = self.target_obj_area ** 0.5
            scale_mul = target_area_sq / med_area_sq
            scale = scale_mul * self.base_scale_range[0], scale_mul * self.base_scale_range[1]
            self.source_random_affine_crop.scale = self.target_random_affine_crop.scale = scale

        img = data['img'].float() / 255
        rstate = random.getstate()
        tseed = random.randrange(sys.maxsize)
        torch.manual_seed(tseed)
        random.setstate(rstate)
        img = self.source_transform(img)

        if self.has_mask:
            mask = data['mask_compressed'].float().unsqueeze(0)
            sdf = data['sdf_compressed'].float().div(127.5).sub(1).unsqueeze(0)

            torch.manual_seed(tseed)
            random.setstate(rstate)
            mask = self.target_transform(mask).round().long()

            torch.manual_seed(tseed)
            random.setstate(rstate)
            sdf = self.target_transform(sdf)

            return img, mask, sdf #, obj_size[unpad]
        else:
            return img

    def __len__(self):
        return len(self.index_map)