import random
import math

import torch
import scipy
import scipy.misc
import scipy.ndimage
import torch.nn.functional as F
import numpy as np


class MeanNormalize:
    def __call__(self, input):
        return input.sub(input.mean())


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, input):
        h_idx = random.randint(0, input.shape[-2] - self.size[0])
        w_idx = random.randint(0, input.shape[-1] - self.size[1])
        return input[:, h_idx: (h_idx + self.size[0]), w_idx: (w_idx + self.size[1])]


class Pad:
    def __init__(self, size, mode='constant'):
        self.size = size
        self.mode = mode

    def __call__(self, x):
        return F.pad(x, pad=self.size, mode=self.mode).data


class RandomAffineCrop:
    def __init__(self, size, padding=0, rotation=(0,), scale=(1, 1),
                 horizontal_flip=False, vertical_flip=False, pad_mode='constant',
                 callback=None):
        self.size = size
        self.padding = padding
        self.rotation = rotation
        self.scale = scale
        self.pad_mode = pad_mode
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.callback = callback # (result, pixel_transform(y, x)) -> None or array

    def __call__(self, input):
        x = input.cpu().numpy()
        if input.dim() == 2:
            x = np.expand_dims(x, 0)
        x = x.transpose((1, 2, 0))

        rotation = random.choice(tuple(self.rotation))
        if not isinstance(rotation, list) and not isinstance(rotation, tuple):
            rotation = (rotation, rotation)
        rotation = math.radians(random.uniform(rotation[0], rotation[1]))
        scale = math.exp(random.uniform(math.log(self.scale[0]), math.log(self.scale[1])))
        hflip = self.horizontal_flip and random.random() > 0.5
        vflip = self.vertical_flip and random.random() > 0.5
        rot_padded_size = np.abs(self.rotate_vec_2d(self.size, self.size, rotation)).max()
        # FIXME: 1.05 will fix corners, too lazy for math
        if abs(rotation % (math.pi / 2)) > 0.05:
            rot_padded_size *= 1.05
        rot_padded_scaled_size = math.ceil(rot_padded_size / scale)
        rot_padded_size = math.ceil(rot_padded_size)

        rot_pad = round((rot_padded_size - self.size) / 2)
        pad = min(rot_padded_scaled_size - 1, self.padding)
        crop_y = round(random.uniform(-pad, x.shape[0] - rot_padded_scaled_size + pad))
        crop_x = round(random.uniform(-pad, x.shape[1] - rot_padded_scaled_size + pad))

        crop_rect = (crop_y, crop_x, rot_padded_scaled_size, rot_padded_scaled_size)
        x = self.padded_crop(x, crop_rect, self.pad_mode)
        x, transf = self.affine_transform(x, scale, rotation, hflip, vflip,
                                          (rot_padded_size, rot_padded_size, x.shape[2]))
        x = x[rot_pad: rot_pad + self.size, rot_pad: rot_pad + self.size]

        if self.callback is not None:
            v = self.callback(x, crop_y, crop_x, rotation, scale, hflip, vflip, transf)
            if v is not None:
                x = v

        x = x.transpose((2, 0, 1))
        if input.dim() == 2:
            x = x.squeeze(0)
        x = torch.from_numpy(x).type_as(input)

        # print(input.shape, x.shape, rotation, scale, (hflip, vflip), rot_padded_size, rot_padded_scaled_size, pad, (crop_y, crop_x))

        return x

    @staticmethod
    def affine_transform(x, scale, rot, hflip, vflip, out_shape):
        c_in = 0.5 * np.array(x.shape)
        c_out = 0.5 * np.array(out_shape)

        rot_transform = np.array([
            [math.cos(rot), math.sin(rot), 0],
            [-math.sin(rot), math.cos(rot), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        scale_transform = np.array([
            [1 / scale, 0, 0],
            [0, 1 / scale, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        flip_transform = np.array([
            [-1 if hflip else 1, 0, 0],
            [0, -1 if vflip else 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        transform = flip_transform @ rot_transform @ scale_transform
        offset = c_in - c_out @ transform.T
        offset[2] = 0

        dst = scipy.ndimage.interpolation.affine_transform(
            x,
            transform,
            order=0,
            offset=offset,
            output_shape=out_shape,
        )
        return dst, transform

    @staticmethod
    def padded_crop(x, rect, pad_mode='constant'):
        """ rect == (y, x, h, w) """
        rect = (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])
        allowed_rect = (
            min(max(rect[0], 0), x.shape[0]),
            min(max(rect[1], 0), x.shape[1]),
            min(max(rect[2], 0), x.shape[0]),
            min(max(rect[3], 0), x.shape[1]),
        )
        padding = (
            allowed_rect[0] - rect[0],
            allowed_rect[1] - rect[1],
            rect[2] - allowed_rect[2],
            rect[3] - allowed_rect[3],
        )

        atop, aleft, abot, aright = allowed_rect
        x = x[atop: abot, aleft: aright]
        ptop, pleft, pbot, pright = padding
        x = np.pad(x, ((ptop, pbot), (pleft, pright), (0, 0)), pad_mode)
        return x

    @staticmethod
    def rotate_vec_2d(y, x, radians):
        ca = math.cos(radians)
        sa = math.sin(radians)
        return ca * x - sa * y, sa * x + ca * y


# class RandomResize:
#     def __init__(self, scale):
#         self.size = size
#         self.mode = mode
#
#     def __call__(self, x):
#         s_shape = (np.array(img.shape) * scale / 16).round().astype(int) * 16
#         if np.max(s_shape) > 768:
#             continue
#         x = scipy.misc.imresize(img, s_shape).astype(np.float32) / 255
#         x = x - x.mean()
#         x = x.transpose(2, 0, 1)
#         x = torch.from_numpy(x).unsqueeze(0).cuda()
#         x = torch.cat([x, flip(x, 2), flip(x, 3), flip(flip(x, 3), 2)])
#         x = Variable(x, volatile=True)
#         x = F.pad(x, 4 * (pad,), mode='reflect')
#         out = model(x)
#         out = out.data[:, :, pad:-pad, pad:-pad]
#         out = out[0].add_(flip(out[1], 1)).add_(flip(out[2], 2)).add_(flip(flip(out[3], 2), 1)).div_(4 * tested_scales)
#         out = out.cpu().numpy()
#         out = np.stack([scipy.misc.imresize(o, img.shape[:2], mode='F') for o in out], 2)