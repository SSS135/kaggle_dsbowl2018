import random

import torch.nn.functional as F


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